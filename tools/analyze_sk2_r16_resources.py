#!/usr/bin/env python3
"""Offline, source-pinned R16 resources and conservative profile associations.

No polling, candidate import, CUDA, timing claims or automatic promotion.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import analyze_sk2_r16_result as performance
from analyze_decoder_selected_resources import visible_template, ROW_RE

ROOT=Path(__file__).resolve().parents[1]
JOB="28d3c46e68504ee4"
SOURCE="464115b2c871338f8d0f51768b82f6bd9090139aea13588ec668a9de587e44f8"
INVENTORY="experiments/iterations/20261104-sk2-r16-cpu.json"
INVENTORY_SHA="f5d75febd9e083563bd42e4edbd73cd93657ffe282e0bc7cc95bda32fbd54c57"
SCHEMA="R16R_SCHEMA phase=prepare_preflight t=MT,TOKR,NT,NWR,WSK,UPW,NS,COMP,HL,NACC,CL,OutBytes,PK,FUSE,BND x=M,N,K,pmax,threads,grid,dyn a=regs,local,static,maxdyn,maxthreads occ=blocks_per_SM err=device,optin,attrs_before,set,attrs_after,occ"
STOP="R16R_STOP coverage=incomplete reason=log_capacity"
TF="MT TOKR NT NWR WSK UPW NS COMP HL NACC CL OutBytes PK FUSE BND".split()
XF="M N K pmax threads grid dynamic_shared_bytes".split()
AF="registers_per_thread local_bytes_per_thread static_shared_bytes max_dynamic_shared_bytes max_threads_per_block".split()
EF="device optin attributes_before set_optin attributes_after occupancy".split()
NUM=r"-?\d{1,20}"
CSV=rf"{NUM}(?:,{NUM})*"
MARKER=re.compile(rf"R16R t=({CSV}) x=({CSV}) a=({CSV}) occ=({NUM}) min=({NUM}) err=({CSV})")


def require(ok,why):
    if not ok:raise ValueError(why)


def digest(data):return hashlib.sha256(data).hexdigest()


def bounded(path,root=ROOT,limit=4*1024*1024):
    p=Path(path);p=(Path(root)/p).resolve() if not p.is_absolute() else p.resolve()
    require(p.is_relative_to(Path(root).resolve()),"artifact outside root")
    require(p.stat().st_size<=limit,"artifact exceeds bound")
    return p,p.read_bytes()


def fields(csv,names):
    vals=list(map(int,csv.split(",")));require(len(vals)==len(names),"field count")
    return dict(zip(names,vals))


def case_key(t,x):return tuple(t[k] for k in TF)+(x["M"],x["N"],x["K"])


def geometry(t,x):
    nw=t["NWR"]*t["WSK"];r=t["NWR"]*t["NT"]*8;ups=t["WSK"]*t["UPW"]
    nesc=min(t["NS"],4) if t["COMP"] else 1
    wby=nw*t["UPW"]*t["NT"]*(768 if t["COMP"] else 2048 if t["HL"] else 1024)
    wbyr=nw*2048;xby=ups*t["TOKR"]*128
    sst=(xby+(max(wby,wbyr) if t["FUSE"] else wby)+1023)&~1023
    toks=16*t["MT"];cp=r+1;rp=r+((40-r%32)%32)
    dyn=(1024+t["NS"]*sst+16*t["NS"]+32+nesc*((toks*cp+3)&~3)*4
         +(t["CL"]-1)*t["WSK"]*toks*rp*4+x["pmax"]*8+128)
    return {"threads":(nw+1+nesc)*32,"grid":x["N"]//r*t["CL"],"dynamic_shared_bytes":dyn,
            "rows":r,"weight_payload_bytes":wby,"router_weight_payload_bytes":wbyr,"slot_bytes":sst}


def load_contract(root=ROOT):
    _,source=bounded(f"experiments/candidates/muse_decode_layer/{SOURCE}.py",root)
    require(digest(source)==SOURCE,"frozen source mismatch")
    _,raw=bounded(INVENTORY,root);require(digest(raw)==INVENTORY_SHA,"CPU inventory hash mismatch")
    proof=json.loads(raw);require(proof["candidate_sha256"]==SOURCE and proof["status"]=="PASS","inventory source/status")
    require('"'+SCHEMA+r'\n"' in source.decode(),"source marker schema mismatch")
    rows=proof["host_admission"]["pointers"]
    require(len(rows)==14 and len({tuple(r["t"]) for r in rows})==12,"inventory cardinality")
    inventory=[]
    for i,r in enumerate(rows):
        t=dict(zip(TF,r["t"]));x=dict(zip(XF,r["x"]))
        inventory.append({"index":i,"template":t,"M":x["M"],"N":x["N"],"K":x["K"],
                          "minimum":r["min"][0],"key":case_key(t,x)})
    # MOCK attributes, occupancy and pmax are intentionally not copied as GPU facts.
    return source.decode(),inventory


def parse_log(log,inventory):
    require(isinstance(log,str) and len(log.encode())<=512*1024,"log bound/type")
    lookup={r["key"]:r for r in inventory};rows=[];malformed=[];stops=[];schemas=[];exceptions=[]
    for number,line in enumerate(log.splitlines(),1):
        if ("R16 resource API failure:" in line or "R16 resource admission failure:" in line) and (
                line.startswith("bfp disabled: RuntimeError(") or line.startswith("RuntimeError:")):
            exceptions.append({"line":number,"text":line[:640]})
        if "R16R" not in line:continue
        if line==SCHEMA:schemas.append(number);continue
        if line==STOP:stops.append(number);continue
        try:
            require(len(rows)<128,"record bound")
            match=MARKER.fullmatch(line);require(match is not None,"unknown/truncated marker")
            ts,xs,ats,occ,minimum,es=match.groups();t=fields(ts,TF);x=fields(xs,XF);a=fields(ats,AF);e=fields(es,EF)
            require(all(t[k]>0 for k in TF[:7]+["NACC","CL"]),"nonpositive template")
            require(all(t[k] in (0,1) for k in ("COMP","HL","PK","FUSE","BND")),"boolean template")
            require(t["OutBytes"] in (2,4),"output type")
            require(all(x[k]>0 for k in XF) and x["pmax"]%2==0,"invalid runtime metadata")
            entry=lookup.get(case_key(t,x));require(entry is not None,"case not in14-case inventory")
            minimum=int(minimum);occ=int(occ);issues=[];derived=geometry(t,x)
            if minimum!=entry["minimum"]:issues.append("wrong_required_occupancy")
            if any(x[k]!=derived[k] for k in ("threads","grid","dynamic_shared_bytes")):issues.append("launch_geometry_mismatch")
            if any(v!=0 for v in e.values()):issues.append("CUDA_API_failure")
            if e["attributes_after"]==0:
                if any(v<0 for v in a.values()) or a["registers_per_thread"]==0:issues.append("invalid_attributes")
                if a["local_bytes_per_thread"]!=0:issues.append("local_allocation_nonzero")
                if x["threads"]>a["max_threads_per_block"]:issues.append("threads_exceed_binary_limit")
                if x["dynamic_shared_bytes"]>a["max_dynamic_shared_bytes"]:issues.append("dynamic_exceeds_binary_limit")
            elif any(v!=-1 for v in a.values()):issues.append("attribute_error_sentinel_mismatch")
            if e["occupancy"]==0 and occ<entry["minimum"]:issues.append("insufficient_block_admission")
            if e["occupancy"]!=0 and occ!=-1:issues.append("occupancy_error_sentinel_mismatch")
            rows.append({"id":len(rows),"line":number,"case_index":entry["index"],"template":t,"launch":x,
                "attributes":a,"blocks_per_SM":occ,"minimum":minimum,"API_status":e,"issues":issues,
                "observed_resource_gate":"FAIL" if issues else "PASS","derived_geometry":derived,
                "phase":"prepare_preflight_not_timed_activation"})
        except (ValueError,KeyError,TypeError) as ex:
            malformed.append({"line":number,"reason":str(ex),"text":line[:640]})
    # Report complete sequential preparation groups; never invent an epoch/weight binding.
    groups=[];current=[];orphans=[]
    for row in rows:
        i=row["case_index"]
        if i==0:
            if current:groups.append({"record_ids":[r["id"] for r in current],"complete":False})
            current=[row]
        elif current and i==len(current):current.append(row)
        else:
            if current:groups.append({"record_ids":[r["id"] for r in current],"complete":False});current=[]
            orphans.append(row["id"])
        if len(current)==14:
            groups.append({"record_ids":[r["id"] for r in current],"complete":True});current=[]
    if current:groups.append({"record_ids":[r["id"] for r in current],"complete":False})
    # Frozen preparation shares the packed table by layer/rows/R: all pooler
    # configurations have R16, as do all QKV configurations. Their N differs.
    for group in groups:
        if group["complete"]:
            for indices in (tuple(range(9)),tuple(range(9,14))):
                matched=[rows[group["record_ids"][i]] for i in indices]
                if len({r["launch"]["pmax"] for r in matched})!=1:
                    group.setdefault("issues",[]).append("within_configuration_pmax_conflict")
    static=defaultdict(list);launches=defaultdict(list);conflicts=[]
    for row in rows:
        identity=tuple(row["template"][k] for k in TF)
        static[identity].append(row)
        launches[identity+(row["launch"]["threads"],row["launch"]["dynamic_shared_bytes"])].append(row)
    for label,sets in (("same_pointer_attributes",static),("same_launch_occupancy",launches)):
        for group in sets.values():
            good=[r for r in group if all(v==0 for v in r["API_status"].values())]
            values={tuple(r["attributes"].values()) if label=="same_pointer_attributes" else r["blocks_per_SM"] for r in good}
            if len(values)>1:conflicts.append({"kind":label,"record_ids":[r["id"] for r in group]})
    failures=[r["id"] for r in rows if r["issues"]]
    complete=bool(groups) and all(g["complete"] and not g.get("issues") for g in groups) and not (orphans or malformed or stops or conflicts)
    gate="FAIL" if failures or exceptions else "PASS_RETAINED_PREFLIGHT_CASES" if complete and schemas else "UNKNOWN_INCOMPLETE_EVIDENCE"
    return {"gate":gate,"records":rows,"preparation_sequence_groups":groups,"orphan_record_ids":orphans,
        "schemas":schemas,"stops":stops,"malformed":malformed,"conflicts":conflicts,"exception_lines":exceptions,
        "failed_record_ids":failures,"case_count":14,"unique_pointer_count":12,
        "observed_case_indices":sorted({r["case_index"] for r in rows}),
        "unobserved_case_indices":sorted(set(range(14))-{r["case_index"] for r in rows}),
        "coverage_limit":"Groups are retained sequence matches, not identified weight sets or timed invocations. pmax is kept per runtimeN/config, never borrowed acrossB32/QKV32."}


def block(text,start):
    opening=text.index("{",start);level=1
    for end in range(opening+1,len(text)):
        level+=(text[end]=="{")-(text[end]=="}")
        if level==0:return text[start:end+1]
    raise ValueError("unbalanced source wrapper")


def generated_catalog(source):
    require(digest(source.encode())==SOURCE,"catalog must use exact source")
    names={"bfp_SK2_CONFIGS","bfp_SK2_PICK","bfp_SK2_K","bfp_SK2_SRC","SK2B","SK2B_UN_TOKR","SK2B1","SK2B_O_TOKR",
           "_bfp__sk2_src_unfused","bfp__sk2_geom","bfp__sk2_src"}
    def key(n):return n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
    env={};exec(compile(ast.Module(body=[n for n in ast.parse(source).body if key(n) in names],type_ignores=[]),"<pinned-pure-source-generators>","exec"),env)
    generated=env["bfp__sk2_src"](True);catalog=[]
    for wrapper in ("sk2","sk2f","sk2t","sk2ft","sk2b"):
        body=block(generated,generated.index(("int64_t " if wrapper=="sk2b" else "void ")+wrapper+"("))
        for line in body.splitlines():
            match=re.search(r"launch_sk2<([^>]+)>",line)
            if not match:continue
            raw=[x.strip() for x in match.group(1).split(",")]
            values=[int(v=="true") if v in ("true","false") else 2 if v=="__nv_bfloat16" else 4 if v=="float" else int(v) for v in raw]
            values+= [0]*(14-len(values));tokr=values[1]
            cfg=re.search(r"cfg == (\d+)",line)
            if cfg:
                c=env["bfp_SK2_CONFIGS"][int(cfg.group(1))];name,layer,n=c[:3]
            elif wrapper=="sk2b":name,layer,n="unpooler_BND","unpooler",8192
            else:name,layer,n="pooler","pooler",4096
            for pk in ((1,) if tokr==8 else (1,0)):
                full=values[:12]+[pk]+values[12:14]
                catalog.append({"template":full,"wrapper":wrapper,"name":name,"layer":layer,"N":n,
                    "K":env["bfp_SK2_K"][layer],"UV_mode":2 if wrapper=="sk2ft" else 1 if wrapper=="sk2t" else 0})
    return catalog


def activation(profiles,catalog,resources):
    captures=[];global_unknown=[];global_fail=[]
    for capture in profiles:
        case=capture["case"];T=case["batch"]*case.get("q_len",1);rows=[]
        report=capture.get("report","");require(isinstance(report,str) and len(report)<=131072,"profile report bound")
        for line in report.splitlines():
            if "sk2_kernel<" not in line and "ub2ns::ub2_kernel" not in line:continue
            match=ROW_RE.fullmatch(line)
            if not match:rows.append({"status":"UNKNOWN","reason":"unparsed_relevant_symbol","text":line[:512]});continue
            symbol,_,_,calls=match.groups();calls=int(calls)
            if "ub2ns::ub2_kernel" in symbol:
                rows.append({"kind":"UB2","calls":calls,"symbol":symbol,"status":"VISIBLE_UB2_SYMBOL_UV_MODE_UNKNOWN"});continue
            visible=visible_template(symbol);compatible=[]
            for entry in catalog:
                vals=visible["values"]
                if visible.get("error") or not vals or entry["template"][:len(vals)]!=vals:continue
                M=2*T if entry["UV_mode"]==2 else T
                tokr=8 if M<=8 else 16 if M<=16 else 32
                if M>32 or entry["template"][1]!=tokr:continue
                if entry["UV_mode"] and T>8:continue
                pk=int(M<tokr or tokr==8)
                if entry["template"][12]!=pk:continue
                matching=[r["id"] for r in resources["records"] if list(r["template"].values())==entry["template"]
                    and r["launch"]["N"]==entry["N"] and r["launch"]["K"]==entry["K"]]
                compatible.append({**entry,"physical_M":M,"compatible_preflight_record_ids":matching})
            rows.append({"kind":"SK2","symbol":symbol,"calls":calls,"visible_template":visible,
                "compatible_source_roles":compatible,"status":"UNAMBIGUOUS_SOURCE_ROLE" if len(compatible)==1 else "UNKNOWN_AMBIGUOUS_ROLE" if compatible else "UNKNOWN_NO_SOURCE_ROLE",
                "timed_pmax_known":False,"preflight_not_bound_to_timed_weight_set":True})
        requirements={};fail=[];unknown=[]
        if T<=32:
            sk2=[r for r in rows if r.get("kind")=="SK2"]
            for layer,required_name in (("qkv_gate",{1:"qkv8_r176",8:"qkv8_r176",16:"qkv16_r176",32:"qkv32_r208"}.get(T)),
                                        ("pooler","pooler32_r144" if T==32 else "pooler")):
                possible=[r for r in sk2 if any(c["layer"]==layer for c in r["compatible_source_roles"])]
                def intended(c):
                    return c["layer"]==layer and c["name"]==required_name and (layer!="pooler" or T==32 or c["template"][13]==1)
                definite=[r for r in possible if r["compatible_source_roles"] and all(intended(c) for c in r["compatible_source_roles"])]
                wrong=[r for r in possible if all(c["layer"]==layer for c in r["compatible_source_roles"]) and not any(intended(c) for c in r["compatible_source_roles"])]
                status="PASS_VISIBLE_SOURCE_ROLE" if definite else "FAIL_WRONG_SELECTED_PATH" if wrong else "UNKNOWN"
                requirements[layer]={"status":status,"required":required_name,"matching_symbols":[r["symbol"] for r in definite]}
                if status.startswith("FAIL"):fail.append(layer)
                elif status=="UNKNOWN":unknown.append(layer)
            # Preserve component observations without treating a hidden runtime ta.mode as known.
            for label,predicate in (("O",lambda c:c["layer"]=="o"),
                                    ("unpooler_BND",lambda c:c["layer"]=="unpooler" and c["template"][14]==1)):
                possible=[r for r in sk2 if any(predicate(c) for c in r["compatible_source_roles"])]
                definite=[r for r in possible if all(predicate(c) for c in r["compatible_source_roles"])]
                required=label=="O" or T==16
                requirements[label]={"status":"PASS_VISIBLE_SOURCE_ROLE" if definite else "UNKNOWN" if required else "NOT_REQUIRED_AT_THIS_T"}
                if required and not definite:unknown.append(label)
            ub2=any(r.get("kind")=="UB2" for r in rows)
            requirements["unpooler_UB2"]={"status":"VISIBLE_SYMBOL" if ub2 else "UNKNOWN" if T<=8 else "NOT_REQUIRED_AT_THIS_T"}
            if T<=8:
                if not ub2:unknown.append("unpooler_UB2")
                uv=[r for r in sk2 if r["compatible_source_roles"] and all(c["UV_mode"]==2 for c in r["compatible_source_roles"])]
                requirements["UV_pooler"]={"status":"PASS_SOURCE_CONSTRAINED_UV2_ROLE" if uv else "UNKNOWN_RUNTIME_MODE_HIDDEN"}
                if not uv:unknown.append("UV_pooler")
        else:requirements["large_T_control"]={"status":"NOT_CHANGED_BY_R16_NO_SMALL_TEMPLATE_REQUIREMENT"}
        global_fail +=[(capture["index"],x) for x in fail];global_unknown +=[(capture["index"],x) for x in unknown]
        captures.append({"index":capture["index"],"logical_T":T,"case":case,"rows":rows,"requirements":requirements})
    return {"gate":"FAIL" if global_fail else "UNKNOWN" if global_unknown else "PASS_VISIBLE_ACTIVATION_ONLY",
        "failures":global_fail,"unknown_requirements":global_unknown,"captures":captures,
        "limits":["Profile prefixes can hidePK/FUSE/BND; role sets preserve every compatible generated path.",
                  "B32 andQKV32 share fullkernel pointers; symbol/callcount alone does not reveal runtimeN orpmax.",
                  "Resource markers describe preparation; actual timedpmax and weight-set binding are absent.",
                  "UV/O modes are runtime values, not kernel template arguments; ambiguous modes remain unknown."]}


def analyze(root=ROOT):
    source,inventory=load_contract(root)
    run=performance.read_run(JOB,SOURCE,"check",root)
    base={"schema":1,"job":JOB,"source_sha256":SOURCE,"inventory_sha256":INVENTORY_SHA,
          "automatic_promotion_allowed":False,"metrics":{},"status":run["status"],"provenance":run["provenance"]}
    if run["status"]=="PENDING":return {**base,"resource_gate":"PENDING","activation_gate":"PENDING"}
    _,raw=bounded(f"experiments/runs/{JOB}.json",root,16*1024*1024);receipt=json.loads(raw);response=receipt["response"];result=response.get("result") or {}
    tail=result.get("log_tail") or "";require(isinstance(tail,str),"logtail type")
    log=tail;log_relation="receipt_tail_only";warnings=[]
    path=(response.get("output") or {}).get("log")
    if path:
        _,data=bounded(path,root,512*1024);external=data.decode()
        if external==tail:log_relation="identical"
        elif external and (not tail or external.endswith(tail) or external.startswith(tail)):log=external;log_relation="saved_log_contains_tail"
        elif tail and (tail.startswith(external) or tail.endswith(external)):log_relation="tail_contains_saved_log"
        else:warnings.append("log_artifacts_disagree");log_relation="disagree_use_receipt_tail"
    parsed=parse_log(log,inventory)
    base.update(resources=parsed,resource_gate=parsed["gate"],log_sha256=digest(log.encode()),log_relation=log_relation,artifact_warnings=warnings)
    if run["status"]!="PASS":return {**base,"activation_gate":"UNAVAILABLE_OFFICIAL_GATE_FAILED",
        "decision":"REJECT_MECHANISM" if parsed["gate"]=="FAIL" else "REJECT_OFFICIAL_GATE",
        "official_failure":run.get("reason"),
        "evidence_limit":"Preparation failed; unobserved resource cases, correctness, timing and timed activation cannot be inferred."}
    # read_run already binds external profiles and their source, exact13/12/12 records.
    for job,mode in performance.PARENTS:
        parent=performance.read_run(job,performance.PARENT_SHA,mode,root)
        require(parent["status"]=="PASS","parent proof missing")
        require(result["evaluator"]==parent["result"]["evaluator"],"full evaluator identity mismatch")
        nb=[r for r in result["records"] if r["event"]=="benchmark"]
        pb=[r for r in parent["result"]["records"] if r["event"]=="benchmark"]
        require(all(n["index"]==p["index"] and {k:v for k,v in n["case"].items() if k!="seed"}==
                    {k:v for k,v in p["case"].items() if k!="seed"} for n,p in zip(nb,pb)),"parent shape mismatch")
    profiles=[r for r in result["records"] if r["event"]=="profile"]
    benches={r["index"]:r for r in result["records"] if r["event"]=="benchmark"}
    require({p["index"] for p in profiles}==set(range(12)),"profile index coverage")
    require(all(p["case"]==benches[p["index"]]["case"] for p in profiles),"profile/benchmark shape mismatch")
    active=activation(profiles,generated_catalog(source),parsed)
    base.update(activation=active,activation_gate=active["gate"])
    if warnings:base["resource_gate"]="UNKNOWN_ARTIFACT_CONFLICT"
    base["decision"]="REJECT_MECHANISM" if parsed["gate"]=="FAIL" or active["gate"]=="FAIL" else "HOLD_UNRESOLVED_EVIDENCE" if warnings or parsed["gate"]!="PASS_RETAINED_PREFLIGHT_CASES" or active["gate"]!="PASS_VISIBLE_ACTIVATION_ONLY" else "RESOURCE_AND_VISIBLE_ACTIVATION_ONLY_NO_PROMOTION"
    return base


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path);args=parser.parse_args()
    try:report=analyze()
    except (OSError,ValueError,KeyError,TypeError,StopIteration) as error:report={"status":"PROVENANCE_OR_INPUT_REJECTED","error":str(error),"metrics":{},"automatic_promotion_allowed":False}
    text=json.dumps(report,indent=2)+"\n"
    if args.output:args.output.write_text(text)
    print(text,end="")
