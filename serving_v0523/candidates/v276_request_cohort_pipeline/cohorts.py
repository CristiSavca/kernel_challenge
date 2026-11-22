"""Price independent DFlash request cohorts and a TP4 replica against recorded serving work."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess

root = Path(__file__).resolve().parents[3]
runs = root/"serving_v0523/runs"
node = root/"experiments/modal_node/20261113T204343Z-bench"
hashes = {}

def read(p):
    raw=p.read_bytes()
    hashes[str(p.relative_to(root))]=sha256(raw).hexdigest()
    return raw

def archive(glob):
    return json.loads(read(next(node.glob(glob))))

def from_pin(path):
    pin="297e54fd105e974cdf401e5b4d20253a22c16f90"
    data=subprocess.check_output(["git","-C",root/"serving_v0523/sglang","show",pin+":python/sglang/"+path])
    hashes[pin+":"+path]=sha256(data).hexdigest()
    return data.decode()

def overlaps(receipt):
    windows=defaultdict(list)
    for r in receipt["requests"]:
        windows[r["load"],r["rep"]].append(r)
    out=[]
    for (load,rep), rows in sorted(windows.items()):
        first=min(r["t0"] for r in rows)
        end=max(r["t0"]+r["dur"] for r in rows)
        events=defaultdict(list)
        for r in rows:
            events[r["t0"]+r["ttft"]].append((r["user"],1))
            events[r["t0"]+r["dur"]].append((r["user"],-1))
        events[first];events[end]
        ticks=sorted(events); active=set(); hist=[0.]*(load+1)
        for i,t in enumerate(ticks):
            for user,change in sorted(events[t],key=lambda e:e[1]):
                if change==1:
                    assert user not in active
                    active.add(user)
                else:
                    active.remove(user)
            if i+1<len(ticks):
                hist[len(active)]+=ticks[i+1]-t
        assert not active and abs(sum(hist)-(end-first))<1e-6
        logged=sum(r["dur"]-r["ttft"] for r in rows)
        assert abs(sum(k*d for k,d in enumerate(hist))-logged)<1e-6
        cold=[r for r in rows if r["turn"]==0]
        out.append(dict(users=load,rep=rep,wall_s=end-first,streamer_count_wall_s=hist,
            at_least_two_streaming_fraction=sum(hist[2:])/(end-first),
            all_streaming_fraction=hist[-1]/(end-first),
            first_wave_new=sum(r["prompt"]-r["cached"] for r in cold),
            all_cold_active_after_last_first_text=min(r["t0"]+r["dur"] for r in cold)
              >max(r["t0"]+r["ttft"] for r in cold)))
    return out

def cold_pipeline(windows):
    # Align server-side cold batch sizes by actual cache flush, then require their
    # exact cumulative work to equal the independently recorded client-side misses.
    log=read(next(node.glob("*__v41__server.log"))).decode().splitlines()
    runs=[]
    for line in log:
        if "TP0 EP0]" not in line:
            continue
        if "Cache flushed successfully!" in line:
            runs.append([])
        elif runs and "Prefill batch" in line:
            runs[-1].append(int(re.search(r"#new-token: (\d+)",line)[1]))
    assert len(runs)==len(windows)
    out=[]
    for batches,window in zip(runs,windows):
        total=0
        weights=[]
        for n in batches:
            total+=n
            weights.append(n)
            if total>=window["first_wave_new"]:
                break
        assert total==window["first_wave_new"]
        # Two equal half-model stages, disjoint 4-GPU groups. Each stage takes
        # half the TP8 full-model cost per chunk times the TP4 layer slowdown.
        # This assumes no cross-stage overhead and work proportional to new rows.
        end1=end2=0.
        for w in weights:
            end1+=w/2
            end2=max(end2,end1)+w/2
        assert end2==(sum(weights)+max(weights))/2
        out.append(dict(users=window["users"],rep=window["rep"],chunks=len(weights),
            total_rows=total,max_rows=max(weights),conditional_cold_gain={
                str(r):total/(end2*r) for r in (1,1.5,2)},
            max_TP4_cost_ratio_to_TP8_for_cold_2x=total/(2*end2),
            max_TP4_cost_ratio_to_TP8_for_cold_4x=total/(4*end2)))
    return out


def gains(run):
    rungs={r["concurrency"]:r for r in run["result"]["rungs"]}
    opts={}
    for full,r in rungs.items():
        cases=[]
        for small,half in rungs.items():
            if full<=small or full%small:
                continue
            groups=full//small
            rate_ratio=half["tok_s_per_user_p50"]/r["tok_s_per_user_p50"]
            period=lambda row:1000*row["acceptance"]/row["tok_s_per_user_p50"]
            # Two disjoint, balanced stages produce one cohort completion every
            # half small-batch period in an indefinitely full round robin.
            best=2*rate_ratio/groups
            limit=rate_ratio/groups
            cases.append(dict(small_users=small,groups=groups,full_period_proxy_ms=period(r),
                small_period_proxy_ms=period(half),balanced_stream_gain=best,
                gain_at_one_quarter_draft=rate_ratio/(groups*.75),
                draft_fraction_break_even_open_interval=([1-limit,limit] if 2*limit>1 else None)))
        opts[full]=cases
    return opts

def frontier(scenarios,trace,official):
    candidates=root/"serving_v0523/candidates"
    scorelib=next(candidates.glob("v265*/audit.py"))
    replay=next(candidates.glob("v195*/score_headroom.py"))
    for p in (scorelib,replay,root/"tools/okc_node_loadgen.py"):
        read(p)
    model=runpy.run_path(scorelib)["calibrate"](official)[0]
    replaylib=runpy.run_path(replay)
    rounds=replaylib["measured_intervals"](trace)
    unchanged=replaylib["rerate"](rounds,(1,1,1))
    out={}
    for name,mult in scenarios.items():
        modified={}
        for key,users in rounds.items():
            f=mult[key[0]]
            modified[key]=[(u[0],u[1],u[2]/f,*u[3:]) for u in users]
        rates=replaylib["rerate"](modified,(1,1,1))
        ys=[rates[n][0]/unchanged[n][0] for n in sorted(unchanged)]
        xs=[rates[n][1]/unchanged[n][1] for n in sorted(unchanged)]
        out[name]=dict(stream_factors=mult,e2e_gpu_factors=ys,e2e_user_factors=xs,estimated_score=model(xs,ys))
    return out

def stage_paths():
    worker=from_pin("srt/speculative/dflash_worker_v2.py")
    assert "self.target_worker.forward_batch_generation(" in worker and "commit_lens=commit_lens" in worker
    backend=from_pin("srt/model_executor/runner_backend/full_cuda_graph_backend.py")
    assert "self._graphs[shape_key] = graph" in backend and "self._graphs[shape_key].replay()" in backend
    spec=next((root/"serving_v0523/sglang/python/sglang/srt/arg_groups").glob("*spec*py"))
    # Locate the pinned validator even when other speculative arg-groups exist.
    folder=spec.parent
    match=next(p for p in folder.glob("*spec*py") if "Currently DFLASH speculative decoding only supports pp_size == 1." in p.read_text())
    txt=from_pin(str(match.relative_to(root/"serving_v0523/sglang/python/sglang")))
    assert "if cfg.pp_size != 1:" in txt
    stats=from_pin("srt/utils/common.py")
    assert "return free_gpu_memory / (1 << 30)" in stats

def replica(official,windows):
    gib=2**30
    logs=read(next(node.glob("*__v41__server.log"))).decode()
    hardware=archive("*session.json")
    assert "143771 MiB" in str(hardware)
    start=float(re.search(r"TP0 EP0\] Load weight begin. avail mem=([0-9.]+)",logs)[1])
    dr=float(re.search(r"TP0 EP0\] Load weight end.*?DFlash2DraftModel,.*?mem usage=([0-9.]+)",logs)[1])
    target=next(i["build"] for i in official["result"]["launch_settings"]["derived"] if i["setting"].endswith(".memory_usage.weight"))
    new=next(w["first_wave_new"] for w in windows if w["users"]==32)
    # Ideal balanced 16+16 assignment of newly computed cold rows. Omit existing
    # shared prefixes. Double per-token KV head bytes for TP4 vs TP8; per-rank
    # target and draft window footprints use only 2,048 live positions/request.
    bank=62*32*27*2**20/gib
    qkvo=62*2*(8192*2304+1024*8192)/gib
    full=(new/2)*8192/gib
    swa=16*2048*23552/gib
    return dict(physical_h200_gib=143771/1024,v41_preload_available_gib=start,
        target_V64_from_internal_weight_gib=target,added_packed_expert_gib=bank,
        added_sharded_qkvo_gib=qkvo,retained_draft_V41_gib=dr,
        cold_rank_full_KV_gib=full,cold_rank_window_KV_gib=swa,
        TP4_full_draft_KV_gib=new/2*5120/gib,TP4_window_draft_KV_gib=16*2048*5120/gib,
        model_total_full_draft_gib=target+bank+qkvo+dr+full+swa+new/2*5120/gib,
        model_total_window_draft_gib=target+bank+qkvo+dr+full+swa+16*2048*5120/gib,
        omitted="additional sharded draft/vocabulary weights, graph capture, workspaces, allocator overhead, pending verification slots; historical footprints and one-copy cache inputs transferred across systems")

def main():
    stage_paths()
    trace=archive("*__v41__load.json"); windows=overlaps(trace)
    controls={}
    for p in sorted(runs.glob("????????????????.json")):
        j=json.loads(p.read_bytes())
        if j["mode"]=="run" and j["result"].get("passed"):
            read(p);controls[p.stem]=gains(j)
    assert len(controls)==5
    baseline=json.loads(read(runs/"abd217baf0ed49fd.json"))
    cur=controls["abd217baf0ed49fd"]
    best={n:max([1]+[c["balanced_stream_gain"] for c in cases]) for n,cases in cur.items()}
    balanced_if_quarter={n:max([1]+[c["gain_at_one_quarter_draft"] for c in cases]) for n,cases in cur.items()}
    outcomes=frontier(dict(base={n:1 for n in cur},ideal_dual_stage=best,
                           draft_quarter=balanced_if_quarter),trace,baseline)
    assert abs(outcomes["base"]["estimated_score"]-1.4119)<5e-5
    return dict(score_controls=controls,V41_stream_exposure=windows,conditional_frontier=outcomes,
        two_stage_cold_pipeline=cold_pipeline(windows),
        two_TP4_model_replicas=replica(baseline,windows),hashes=hashes,
        scope="Rung median output rates are not physical per-stage CUDA times. Equal simultaneous stages on disjoint resources, no contention, unchanged demand and observed rates transferred by cohort; a scenario, not a universal achievable bound.")
if __name__=="__main__":
    print(json.dumps(main(),indent=2))
