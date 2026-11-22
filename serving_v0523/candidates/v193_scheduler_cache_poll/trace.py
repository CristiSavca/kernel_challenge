"""Rebuild aggregate cache and critical-path evidence from public results and our own 8-H200 load traces."""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
modal = ROOT/"experiments/modal_node"
sched = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
    r"full token usage: ([\d.]+), swa token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)"
)


def info(path):
    return {"source": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def replay(rows, official_wall):
    by_user = defaultdict(list)
    for row in rows:
        assert 0 <= row["ttft"] <= row["dur"] and row["cached"] <= row["prompt"]
        by_user[row["user"]].append(row)
    totals = []
    first = []
    later = []
    for user_rows in by_user.values():
        user_rows.sort(key=lambda r: r["turn"])
        assert [r["turn"] for r in user_rows] == list(range(13))
        end = max(r["t0"] + r["dur"] for r in user_rows)
        totals.append(end)
        first.append(user_rows[0]["ttft"])
        later.append(sum(r["ttft"] for r in user_rows[1:]))
    wall = max(totals)
    assert abs(official_wall - wall) < .03
    def counterfactual(savings):
        return wall / max(t - s for t,s in zip(totals,savings))
    return dict(concurrent_users=len(by_user), wall_s=wall, requests=len(rows),
                prompt_rows=sum(r["prompt"] for r in rows),
                cached_rows=sum(r["cached"] for r in rows),
                real_prefill_rows=sum(r["prompt"]-r["cached"] for r in rows),
                first_real_prefill_rows=sum(r["prompt"]-r["cached"] for r in rows if r["turn"]==0),
                all_later_ttft_free_fixed_other_times=counterfactual(later),
                all_first_ttft_free_fixed_other_times=counterfactual(first))


def archive(file, server_log=None):
    data=json.loads(file.read_bytes())
    result=[]
    for rung in data["rungs"]:
        users=rung["concurrency"]
        for rep,wall in enumerate(rung["wall_s"]):
            rows=[r for r in data["requests"] if r["load"]==users and r["rep"]==rep]
            result.append(dict(rep=rep,**replay(rows,wall)))
    if server_log:
        batches=[[] for _ in result]
        round_id=-1
        for line in server_log.read_text().splitlines():
            if "Cache flushed successfully!" in line:
                round_id+=1
            match=sched.search(line)
            if match and round_id>=0:
                assert "TP0 EP0" in line and "cuda graph: True" in line
                batches[round_id].append(tuple(map(float,match.groups())))
        assert round_id+1 == len(result)
        for report,samples in zip(result,batches):
            assert sum(v[1] for v in samples) == report["real_prefill_rows"]
            report["prefill_scheduler"]=dict(passes=len(samples),
                full_16384_passes=sum(v[1]==16384 for v in samples),
                passes_with_queue=sum(v[6]>0 for v in samples), queue_peak=int(max(v[6] for v in samples)),
                passes_with_running_decode=sum(v[5]>0 for v in samples),
                full_usage_peak=max(v[3] for v in samples), swa_usage_peak=max(v[4] for v in samples))
    return info(file) | {"rounds": result}


def official(job):
    path=ROOT/"serving_v0523/runs"/(job+".json")
    result=json.loads(path.read_bytes())["result"]
    assert result["passed"] is True and all(not r["truncated"] for r in result["rungs"])
    return info(path) | {"rungs": [{k:r[k] for k in ("concurrency","ttft_first_p90_s","ttft_later_p90_s",
        "tok_s_per_gpu","tok_s_per_user_p50","acceptance","cache_hit")} for r in result["rungs"]]}


if __name__=="__main__":
    v41=next(modal.rglob("*v41__load.json"))
    v44=next(modal.rglob("*v44__load.json"))
    logs=next(v41.parent.glob("*v41__server.log"))
    print(json.dumps({"V64":official("abd217baf0ed49fd"),"V68":official("04b01f958aa5487c"),
        "V41":archive(v41,logs), "V41_scheduler_log":info(logs),"V44":archive(v44),
        "counterfactual":"Subtract the indicated recorded client delay independently from each user's timeline, hold every other duration and interturn gap fixed, then take the maximum end. Queue or compute redistribution can change other durations.",
        "cache_caveat":"Cached includes GPU hits and any host-tier hits. Logged usage excludes evictable rows. No copy engine timing or actual restored-row count is exposed."},indent=2))
