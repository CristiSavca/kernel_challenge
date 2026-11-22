"""Recompute hardware comparisons and fixed-timeline counterfactuals using archived receipts."""
from collections import defaultdict
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "serving_v0523/runs"
SCORED = {"V64":"abd217baf0ed49fd", "V48b":"8ebed2bced8942ff"}
PRIVATE = {"V64":"ac89fddc4ee147ab", "V68":"04b01f958aa5487c"}


def collect(jobs, scored):
    result = {}
    for label, job in jobs.items():
        path = RUNS / (job+".json")
        doc = json.loads(path.read_text())
        run = doc["result"]
        assert doc["status"] == "done" and run["passed"] and run["correctness"]["passed"] and run["later_turns"]["passed"]
        assert all(not rung["truncated"] for rung in run["rungs"])
        result[label] = dict(id=job, mode=doc["mode"], cluster=doc["cluster"], node=run["node"],
                             flags=doc["serving_flags"], shape=run["shape"], rungs=run["rungs"],
                             source=str(path.relative_to(ROOT)), gpu_memory=run["gpu_memory"],
                             comparison=run["frontier" if scored else "vs_stock"])
    return result


def timelines(path):
    data = json.loads(path.read_text())
    user_chains = defaultdict(list)
    for r in data["requests"]:
        user_chains[r["load"],r["rep"],r["user"]].append(r)
    shapes = defaultdict(list)
    for (load, rep, user), chain in user_chains.items():
        chain.sort(key=lambda x:x["turn"])
        assert len(chain)==13 and all(r["turn"]==i for i,r in enumerate(chain))
        assert all(a["t0"]+a["dur"] <= b["t0"]+0.00001 for a,b in zip(chain,chain[1:]))
        first = sum(r["ttft"] for r in chain)
        after = sum(r["dur"]-r["ttft"] for r in chain)
        total = max(r["t0"]+r["dur"] for r in chain)
        gap = total-first-after
        assert gap >= -1e-6 and min(after, first) > 0
        shapes[load, rep].append((gap, first, after))
    out = {}
    scenarios = {"measured":(1,1), "twice_ttft":(0.5,1), "twice_after_first":(1,0.5),
                 "free_ttft":(0,1), "free_after_first":(1,0)}
    for load in sorted({load for load,_ in shapes}):
        groups = [vals for (n,rep),vals in shapes.items() if n==load]
        assert all(len(vals)==load for vals in groups)
        walls = {key:[max(g+ap*a+bp*b for g,a,b in vals) for vals in groups] for key,(ap,bp) in scenarios.items()}
        rung = next(r for r in data["rungs"] if r["concurrency"]==load)
        assert all(abs(a-b)<0.025 for a,b in zip(walls["measured"],rung["wall_s"],strict=True))
        out[load] = dict(reps=len(groups), requests=sum(len(x)*13 for x in groups), measured_makespans=walls["measured"],
                         at_fixed_user_order_and_gaps={k:sum(walls["measured"])/sum(v) for k,v in walls.items() if k!="measured"})
    return dict(source=str(path.relative_to(ROOT)), recorded_requests=len(data["requests"]), by_load=out)


def main():
    scored, private = collect(SCORED, True), collect(PRIVATE, False)
    assert scored["V64"]["node"]==scored["V48b"]["node"]
    assert private["V64"]["node"]!=private["V68"]["node"] and private["V64"]["flags"]==private["V68"]["flags"]
    score = scored["V64"]["comparison"]["score"]
    target_factor = 4/score
    private_ratio = {k:v["comparison"]["rungs"][0]["at_stock_point"] for k,v in private.items()}
    traces = {}
    base = ROOT/"experiments/modal_node"
    for candidate in ("v41","v44"):
        path, = base.glob("*/*__"+candidate+"__load.json")
        traces[candidate] = timelines(path)
    v175 = json.loads((HERE.parent/"v175_decode_collective_pipeline/check.json").read_text())
    answer = dict(scored=scored, private=private, recorded_client_traces=traces,
          target=dict(common_required_per_rung=target_factor, fixed_work_time_reduction_if_uniform=1-1/target_factor,
                 per_rung_for_each_stock_ratio_four={p["concurrency"]:4/p["ratio"] for p in scored["V64"]["comparison"]["points"]},
                 if_k_rungs_change_equally={k:target_factor**(4/k) for k in range(1,5)}),
          private_mixed_node_normalized_v68_over_v64=private_ratio["V68"]/private_ratio["V64"],
          private_mixed_node_raw_v68_over_v64=private["V68"]["rungs"][0]["tok_s_per_gpu"]/private["V64"]["rungs"][0]["tok_s_per_gpu"],
          v175_scoped_decode_pool_unpool_idealized=[v["proxy_ratio_if_both_projections_free"] for v in v175["per_rung"]])
    print(json.dumps(answer,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
