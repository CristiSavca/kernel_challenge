"""Estimate grouped launches from per-token rank occupancy and archived public-shape controls."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

here = Path(__file__).resolve().parent
root = here.parents[2]
source = Path(sys.argv[1])/"python/sglang/srt/layers/muse_spark/humming_prefill_waves.py"
node = next(n for n in ast.parse(source.read_bytes()).body if getattr(n,"name",None) == "wave_capacity")
env = {"triton": SimpleNamespace(cdiv=lambda x,y: (x+y-1)//y, next_power_of_2=lambda x: 1<<(x-1).bit_length())}
exec(compile(ast.Module([node],[]),str(source),"exec"),env)
prior = json.loads(next(here.parent.glob("v115*/geometry.json")).read_bytes())
trace = json.loads(next(here.parent.glob("v197*/independent_reach.json")).read_bytes())
samples = json.loads((root / trace["H200"]["source"]).read_bytes())
rng = np.random.default_rng(prior["seed"])
def geometry(tokens):
    hidden = intermediate = 4096
    topk = 8
    slots = tokens * topk
    legacy_cap = env["wave_capacity"](tokens,topk,32,hidden,intermediate)
    rows = min(slots,legacy_cap+topk-1)
    elements = rows * (max(hidden,intermediate) + max(hidden,2*intermediate))
    compact_rows = min(slots,elements//(hidden+intermediate))
    cap = compact_rows if compact_rows == slots else compact_rows-topk+1
    return (legacy_cap,rows),(cap,compact_rows),elements*2
report = []
for ref in prior["shapes"]:
    t = ref["tokens"]
    old,new,arena_bytes = geometry(t)
    counts = (Counter(), Counter())
    for i in range(512):
        local = rng.hypergeometric(32,224,8,size=t)
        last = int(local[:-1].sum())
        for (capacity,_),count in zip((old,new),counts):
            count[min((8*t+capacity-1)//capacity,1+last//capacity)] += 1
    assert {int(k):v for k,v in ref["iterations"].items()} == counts[0]
    avg = [sum(k*v for k,v in c.items())/512 for c in counts]
    assert counts[1] == {1:512}
    report.append(dict(tokens=t,legacy_capacity_rows=old,compact_capacity_rows=new,
        bf16_workspace_bytes_before_and_after=arena_bytes,max_waves=[(8*t+c-1)//c for c,_ in (old,new)],
        synthetic_wave_counts=counts,mean_waves=avg,
        conditional_mean_projection_calls_per_rank_per_62_layers=[2*n*62 for n in avg],
        conditional_mean_separate_SiLU_calls_per_rank_per_62_layers=[avg[0]*62,0]))
exposure = []
for row in trace["archive"]:
    calls = 62 * row["logical_grouped_passes"]
    exposure.append(dict(users=row["users"],rep=row["rep"],eligible_prompt_passes=row["logical_grouped_passes"],
        modelled_eager_count_readbacks_avoided_per_rank=calls,
        average_net_saving_ms_per_eligible_layer_for_5pct_run=row["net_ms_per_layer_call_for_5pct_raw"],
        average_net_saving_ms_per_eligible_layer_for_2_833x_run=row["net_ms_per_layer_call_for_2_833x_raw"]))
hardware = []
for row in samples["cases"]:
    t = row["T"]
    if t not in (8192,12288,16384): continue
    n = row["local_routes"]
    old,new,_ = geometry(t)
    # A count smaller than capacity guarantees one body regardless of final-token routing.
    hardware.append(dict(tokens=t,synthetic_H200_local_routes=n,
        fits_without_overflow_V115=n < old[0],fits_without_overflow_V306=n < new[0]))
print(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),shapes=report,
    archived_H200_synthetic_route_counts=hardware,eligibility_envelope=exposure,
    conditional_assumptions="512 independent draws per shape; eight distinct uniform choices, 32 of 256 experts local. Counterfactual kernel calls, counting only eligible conditional replays on each rank.",
    eligibility_assumptions="Prior scheduler-shape reconstruction with 62 expert layers per eligible pass. Includes no measurement of actual calls, graph capture, routing skew, sync latency or device overlap.",
    baseline_note="V197 already fuses W13 on supported prompt geometry. Against V197 the new benefit is removal of the host route-count read and eager graph break. V115 supplies the wave-workspace comparison.",
    physical_V306_timing=False),indent=2))
