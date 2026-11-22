#!/usr/bin/env python3
"""Offer one final private check after a confirmed terminal control failure."""
import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from pathlib import Path

import nov20_check_slots as slots
import submit_nov17_integrated_checks as measured
import submit_nov20_v129_check as first
import submit_nov20_v132_check as second
import submit_nov20_v141_check as third

ROOT, RUNS, V64 = measured.ROOT, measured.RUNS, measured.V64
START, END, RESET = first.START, first.END, first.RESET
V154C = measured.Candidate("v154c", "v154c_v64_hicache_backup",
    "7bc2eca145404dd077273440e2d18b9c0b9c5da83ff8bdcc5c10907733896b23", V64.flags)
NAME = "v154c_nov20_check16"
REFERENCE = "ac89fddc4ee147ab"
REFERENCE_HASH = "52ad423aa93fb6900bf921b75f28bebe08b6275aa4fd9ce09058a7dad4ff5241"
FIRST = third.REQUIRED[0]
V132 = third.REQUIRED[1]
V141 = second.PEER
V139 = third.REQUIRED[2]
V120B = (*second.PARENTS[1], second.previous.START, second.previous.END)
BRANCHES = ((V141, V132, V120B), (V132, V141, V139))
OWN = (V154C,16,NAME,START,END)


def validated_plan(root=ROOT,runs=RUNS):
    base=root/"serving_v0523/candidates"
    for c in (V64, FIRST[0], V132[0], V141[0], V139[0], V120B[0], V154C):
        if first.digest(base/c.directory/"patch.diff") != c.sha256:
            raise ValueError("frozen patch mismatch for "+c.name)
    val=json.loads((base/V154C.directory/"validation.json").read_text())
    if (val["patch_sha256"]!=V154C.sha256 or val["launch_flags"]!=V64.flags
            or val["parent_sha256"]!=V64.sha256 or val["tree"]!="ce86ba9a3c3f182021ddbf14c3ea40852119d8e9"
            or val["fixed_flag_patterns"]!=76 or val["strict_apply_full_delta_and_reverse"] is not True
            or val["changed_from_parent"]!=["python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh"]):
        raise ValueError("V154c source lineage, flags or validation mismatch")
    ref=json.loads((runs/(REFERENCE+".json")).read_text())
    check=(V64,16,"v64_nov17_check16",measured.START,measured.END)
    if (slots.dated_receipt(runs,check)!=REFERENCE or type(ref.get("created_at")) not in (int,float)
            or not measured.START.timestamp()<=ref["created_at"]<measured.END.timestamp()
            or ref.get("users") is not None and (type(ref["users"]) is not int or ref["users"]!=16)
            or (ref["id"],ref["board"],ref["mode"],ref["status"],ref["duplicate_of"],ref["serving_flags"])
               !=(REFERENCE,"serving","check","done",None,V64.flags)):
        raise ValueError("V64 matched private reference mismatch")
    r=ref["result"]
    if (r["passed"] is not True or r["correctness"]["passed"] is not True or r["status"]!="ok"
            or r["patch"]["sha256"]!=V64.sha256 or r["serving_flags"]!=V64.flags or r["concurrency"]!=[16]
            or len(r["rungs"])!=1 or r["rungs"][0]["concurrency"]!=16 or r["rungs"][0]["truncated"] is not False
            or r["rungs"][0]["tok_s_per_gpu"]!=17349.9 or r["rungs"][0]["reps"]!=1):
        raise ValueError("V64 matched private rung mismatch")
    return dict(mode="check",users=16,candidate=V154C.name,sha256=V154C.sha256,flags=V154C.flags,
                comparison_job=REFERENCE,comparison_sha256=V64.sha256,comparison_tokens_s_gpu=17349.9,
                archived_reference_document_sha256=REFERENCE_HASH,
                start_utc=START.isoformat(),end_utc_exclusive=END.isoformat(),resets_at=RESET,
                required_quota=slots.THIRD,reserve_checks=0,maximum_private_checks=1,
                first=FIRST[2],alternatives=[dict(accepted=a[2],unattempted=b[2],terminal_control=c[2])
                                            for a,b,c in BRANCHES],
                arbitration_lock=slots.LOCK,receipt=NAME+".submit.txt")


def unique_receipt(runs,req):
    candidate,_,name,*_=req
    job=slots.dated_receipt(runs,req)
    if not job:
        return None
    canonical=runs/(name+".submit.txt")
    tag=re.compile(r"(?<![a-z0-9])"+re.escape(candidate.name)+r"(?![a-z0-9])",re.I)
    for path in runs.rglob("*"):
        if not path.is_file() or path==canonical:
            continue
        filename=path.name
        submitted=filename.endswith((".submit.txt",".submit.json",".submitted_at.txt"))
        if tag.search(filename) and (submitted or filename.endswith(".skip.json")):
            return None
        if first.JUDGE_FILE.fullmatch(filename):
            if filename.split(".")[0]==job:
                continue
            text=path.read_text()
        elif submitted:
            first_line,_,body=path.read_text().partition("\n")
            try:
                h=json.loads(first_line)
            except ValueError:
                h=None
            if isinstance(h,dict) and h.get("candidate") and h.get("sha256"):
                if h["sha256"]==candidate.sha256 or tag.search(str(h["candidate"])) or tag.search(str(h.get("directory",""))):
                    return None
                # A different submitted candidate can correctly cite this prerequisite.
                text=body
            else:
                text=first_line+"\n"+body
        else:
            continue
        if candidate.sha256 in text or tag.search(text) or submitted and re.search(r"\bjob "+job+r"\b",text):
            return None
    return job


def local_position(runs):
    first_job=unique_receipt(runs,FIRST)
    if not first_job:
        return None
    for taken,missing,control in BRANCHES:
        if slots.prior_attempt(runs,missing[0],missing[2]):
            continue
        peer=unique_receipt(runs,taken)
        terminal=unique_receipt(runs,control)
        if peer and terminal and len({first_job,peer,terminal,REFERENCE})==4:
            return (FIRST,taken,control),(first_job,peer,terminal)
    return None


def official_identity(req,job,okc,execute,*,runs=None):
    response=execute([okc,"serve","status",job,"--json"],capture_output=True,text=True,timeout=60)
    if response.returncode:
        return None
    try:
        data=json.loads(response.stdout)
        state=slots.official_state(req,job,okc,lambda *a,**kw:response,runs=runs)
        if state not in ("pending","passed","failed"):
            return None
        candidate,users,*_=req
        status=data["status"]
        r=data.get("result")
        if status=="done" or r is not None:
            if (not isinstance(r,dict) or not isinstance(r.get("patch"),dict)
                    or r["patch"].get("sha256")!=candidate.sha256 or r.get("serving_flags")!=candidate.flags
                    or r.get("concurrency")!=[users] or type(r["concurrency"][0]) is not int
                    or type(r.get("passed")) is not bool or state=="pending"):
                return None
            quality=r.get("correctness")
            quality_pass=quality.get("passed") if isinstance(quality,dict) else None
            if state=="failed":
                negative=r.get("status") in ("failed","crashed","error","timeout","incorrect")
                if r["passed"] or not (negative or r.get("status")=="ok" and quality_pass is False):
                    return None
            elif not r["passed"] or r.get("status")!="ok" or quality_pass is False:
                return None
        # Explicit runtime failure is required from the competing control. Other
        # terminal states still prove consumption by either of the first two checks.
        failure=state=="failed" and data.get("status") in ("done","failed")
        return state,failure,data["created_at"]
    except (ValueError,KeyError,TypeError,IndexError):
        return None


def live_position(position,okc,execute,*,runs=None):
    if position is None:
        return False
    checks,jobs=position
    control=official_identity(checks[2],jobs[2],okc,execute,runs=runs)
    if control is None or not control[1]:
        return False
    first_state=official_identity(checks[0],jobs[0],okc,execute,runs=runs)
    if first_state is None:
        return False
    peer_state=official_identity(checks[1],jobs[1],okc,execute,runs=runs)
    return peer_state is not None and first_state[2]<=peer_state[2]


def run_once(*,okc,root=ROOT,runs=RUNS,clock=first.utc_now,execute=subprocess.run):
    def inside():
        return START<=clock()<END
    def denied(command,reason):
        return subprocess.CompletedProcess(command,1,"",reason)
    def bounded(cmd,**kwargs):
        now=clock()
        if not START<=now<END:
            return denied(cmd,"submission window closed")
        return execute(cmd,**(kwargs|{"timeout":min(kwargs["timeout"],(END-now).total_seconds())}))
    if not inside():
        return "outside submission window"
    with slots.arbitrate(runs) as held:
        if not held:
            return "another allocation poll is active"
        def prior(own=False):
            return slots.prior_attempt(runs,V154C,NAME,exclude_own=own)
        if prior():
            return "attempt recorded"
        validated_plan(root,runs)
        position=local_position(runs)
        if not live_position(position,okc,bounded,runs=runs):
            return "waiting for two confirmed November 20 checks and terminal competitor"
        patch=root/"serving_v0523/candidates"/V154C.directory/"patch.diff"
        expected=[okc,"serve","check",str(patch),"--users","16",*first.FLAGS,"--no-wait"]
        status=[okc,"serve","status","--json"]
        def quota():
            rsp=bounded(status,capture_output=True,text=True,timeout=60)
            if rsp.returncode:
                return rsp
            try:
                data=json.loads(rsp.stdout)
                counts=data["limits"]["checks"]
                if data["resets_at"]==RESET and all(type(counts[k]) is int and counts[k]==v
                                                    for k,v in slots.THIRD.items()):
                    return rsp
            except (KeyError,TypeError,ValueError):
                pass
            return denied(status,"last-slot quota unavailable")
        def eligible(own=False):
            return (inside() and not prior(own) and local_position(runs)==position
                    and live_position(position,okc,bounded,runs=runs) and inside())
        def guarded(cmd,**kwargs):
            if cmd==status:
                # Complete all transient remote checks before submit_once writes its
                # exclusive attempt. A transport failure leaves this slot retryable.
                rsp=quota()
                if rsp.returncode or not eligible():
                    return denied(cmd,"last-slot evidence changed")
                validated_plan(root,runs)
                rsp=quota()
                return rsp if not rsp.returncode and eligible() else denied(cmd,"last-slot evidence changed")
            if cmd!=expected:
                raise ValueError("unexpected non-private check command")
            if not inside() or prior(True) or local_position(runs)!=position or first.digest(patch)!=V154C.sha256:
                return denied(cmd,"frozen local identity or window changed")
            return bounded(cmd,**kwargs)
        requests,ids=position
        evidence={"comparison_job":REFERENCE,"comparison_sha256":V64.sha256,"slot":3,
                  "nov20_jobs":dict(zip([r[0].name for r in requests[:2]],ids[:2],strict=True)),
                  "failed_control":requests[2][0].name,"failed_control_job":ids[2]}
        result=measured.submit_once((NAME,V154C,16,evidence),okc=okc,root=root,runs=runs,now=clock(),
                execute=guarded,start=START,end=END,reset=RESET,reserve_checks=0)
        if result.startswith("submitted ") and slots.dated_receipt(runs,OWN)!=result.split()[1]:
            return "attempt needs manual reconciliation; no retry"
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    plan=validated_plan()
    if not args.run:
        print(json.dumps(plan,indent=2))
        return
    okc=os.environ.get("OKC",str(Path.home()/".okc/bin/okc"))
    with (RUNS/"nov20_v154c_runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        previous=None
        print(json.dumps(dict(armed_at=first.utc_now().isoformat(),**plan)),flush=True)
        while first.utc_now()<END:
            if first.utc_now()>=START:
                try:
                    state=run_once(okc=okc)
                except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
                    state=f"{type(exc).__name__}: {exc}"
                if state!=previous:
                    print(json.dumps(dict(at=first.utc_now().isoformat(),outcome=state)),flush=True)
                    previous=state
                if state.startswith(("attempt ","submitted ","prior attempt","outside submission window")):
                    break
            time.sleep(min(60,max(0,(END-first.utc_now()).total_seconds())))
        print(json.dumps(dict(finished_at=first.utc_now().isoformat())),flush=True)


if __name__=="__main__":
    main()
