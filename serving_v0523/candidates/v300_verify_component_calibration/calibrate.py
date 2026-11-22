"""Compare homogeneous one-user 8-H200 rounds to independent Hopper layer measurements."""
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
from statistics import median
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAND = HERE.parent
V297 = CAND / "v297_wide_fused_decode"
V288 = runpy.run_path(str(CAND / "v288_paired_cold_and_verify_intervals/check.py"))
HASH = {}


def payload(path):
    b = path.read_bytes()
    HASH[str(path.relative_to(ROOT))] = sha256(b).hexdigest()
    return json.loads(b)


def reconstruct():
    repo = ROOT / "serving_v0523/sglang"
    freeze = payload(V297/"validation.json")
    with tempfile.TemporaryDirectory() as td:
        obj = Path(td)/"objects"; obj.mkdir()
        gd = subprocess.check_output(["git","-C",repo,"rev-parse","--absolute-git-dir"],text=True).strip()
        env = dict(os.environ,GIT_INDEX_FILE=td+"/index",GIT_OBJECT_DIRECTORY=str(obj),GIT_ALTERNATE_OBJECT_DIRECTORIES=gd+"/objects")
        def git(*a,data=None):
            return subprocess.check_output(["git","-C",repo,*a],env=env,input=data)
        git("read-tree",freeze["base"])
        patch=(V297/"patch.diff").read_bytes()
        assert sha256(patch).hexdigest()==freeze["patch_sha256"]
        git("apply","--cached","--whitespace=error-all","-",data=patch)
        assert git("write-tree").decode().strip()==freeze["tree"]
        def bind(path,terms,ordered=False):
            data=git("show",":"+path)
            HASH["V297:"+path]=sha256(data).hexdigest()
            src=data.decode(); positions=[]
            for x in terms:
                positions.append(src.index(x,positions[-1]+1 if ordered and positions else 0))
            return dict(zip(terms,[src[:p].count("\n")+1 for p in positions]))
        src="python/sglang/"
        worker=src+"srt/speculative/dflash_worker_v2.py"
        return dict(tree=freeze["tree"],
            worker=bind(worker,["proposal = self.propose_decode_block(batch)","def propose_decode_block(", "draft_out = self.draft_model_runner.forward(forward_batch)","def is_current_decode_proposal(","def verify_proposed_decode_block(","target_out = self.target_worker.forward_batch_generation("],True),
            output=bind(worker,["# which the next target forward overwrites","verify_fc_shard=logits_output.dflash_verify_fc_shard","logits_output.dflash_verify_fc_shard = None"],True),
            relay=bind(src+"srt/managers/overlap_utils.py",["def resolve_seq_lens_cpu(","self.publish_ready.synchronize()","def publish(","self.publish_ready.record()"],True),
            attention=bind(src+"kernels/ops/attention/paged_verify_attention.py",["def _tile_geometry(","def _window_tile_run(","def paged_verify_attention(","_partition_attention[(partitions, kv_heads)]","_merge_partitions[(batch, kv_heads, max_q * groups)]"],True))


def windows():
    allrows={}
    for tag in ("v41","v44"):
        result=V288["decode_windows"](V288["run_inputs"](tag))
        groups=defaultdict(list)
        for (rep,turn,ordinal),r in result.items():groups[rep].append(r)
        allrows[tag]=[dict(rep=rep, windows=len(rs), median_context=median((r["first_context_report"]+r["last_context_report"])/2 for r in rs),
                         min_context=min(r["first_context_report"] for r in rs), max_context=max(r["last_context_report"] for r in rs),
                         median_ms=median(r["step_ms"] for r in rs), median_accept=median(r["mean_accepted"] for r in rs)) for rep,rs in sorted(groups.items())]
    HASH.update(V288["SOURCE_HASHES"])
    pairs=payload(CAND/"v288_paired_cold_and_verify_intervals/check.json")["decode"]["pairs"]
    measured={name:median(r[name]["step_ms"] for r in pairs) for name in ("v41","v44")}
    assert len(pairs)==9 and all(sum(r["windows"] for r in a)==25 for a in allrows.values())
    return dict(all_complete_windows=allrows,matched_pairs=len(pairs),matched_median_ms=measured)


def checks(problem):
    by_load=defaultdict(list)
    found=set()
    for p in sorted((ROOT/"experiments/candidates"/problem/"profiles").glob("*.json")):
        obj=payload(p); rows=obj.get("benchmarks",[])
        if len(rows)!=12:continue
        job=obj["job_id"]
        assert job not in found and obj["problem"]==problem
        found.add(job)
        for load in ("verify 1","verify 16","verify 32"):
            pair=[r for r in rows if r.get("status")=="pass" and r["case"]["load"]==load]
            if len(pair)!=2:continue
            pair.sort(key=lambda x:x["case"]["layer"])
            assert [(r["case"]["layer"],r["case"]["weight"],r["case"]["page"]) for r in pair]==[("full",16,1),("sliding",46,1)]
            estimates={key:sum(r[field]*r["case"]["weight"] for r in pair)/1e6 for key,field in
                       [("stock_ms","baseline_ns"),("candidate_ms","median_ns"),("published_floor_ms","speed_of_light_ns")]}
            estimates.update({r["case"]["layer"]+"_stock_ms":r["baseline_ns"]*r["case"]["weight"]/1e6 for r in pair})
            estimates.update({r["case"]["layer"]+"_candidate_ms":r["median_ns"]*r["case"]["weight"]/1e6 for r in pair})
            by_load[load].append(dict(job=job,**estimates))
    result={}
    for load,values in by_load.items():
        assert len(values)==len(found)
        result[load]=dict(unique_jobs=len(values),fastest_job=min(values,key=lambda r:r["candidate_ms"]),
            distribution={k:dict(min=min(x[k] for x in values),median=median(x[k] for x in values),max=max(x[k] for x in values)) for k in values[0] if k!="job"})
    return result


def thresholds(period,target,attention):
    return dict(period_ms=period,host_or_draft_gap_needed_ms={k:period*(1-1/f) for k,f in (("2x",2),("3x",3),("4x",4))},
                remaining_after_fastest_synthetic_target_ms={k:period/f-target for k,f in (("2x",2),("3x",3),("4x",4))},
                if_all_stock_attention_cost_deleted=period/(period-attention["stock_ms"]["max"]),
                if_all_fastest_attention_cost_deleted=period/(period-attention["candidate_ms"]["min"]))


def main():
    source=reconstruct(); obs=windows(); layer=checks("muse_decode_layer"); attention=checks("muse_attention_decode")
    assert layer["verify 1"]["unique_jobs"]==27 and attention["verify 1"]["unique_jobs"]==9
    p=obs["matched_median_ms"]["v41"]
    standalone=layer["verify 1"]["distribution"]; a=attention["verify 1"]["distribution"]
    assert standalone["stock_ms"]["min"]>p
    rounds=payload(CAND/"v140_decode_overlap_feasibility/check.json")["model"]["rungs"]
    budgets={"V41_verify1":thresholds(p,standalone["candidate_ms"]["min"],a)}
    for row in rounds:
        u=row["users"]; key="verify "+str(u)
        if key in layer: budgets["V64_"+key]=thresholds(row["estimated_period_ms"],layer[key]["distribution"]["candidate_ms"]["min"],attention[key]["distribution"])
    print(json.dumps(dict(source=source,service=obs,layer=layer,attention=attention,budgets=budgets,sha256=HASH),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
