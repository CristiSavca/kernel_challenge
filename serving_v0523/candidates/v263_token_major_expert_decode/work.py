"""Estimate logical work for source-token W13 using observed H200 route count and real tuning."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(p):
    return json.loads(p.read_bytes())


h200 = ROOT / "experiments/modal_serving/20261112T191039Z/result.json"
tuning = HERE.parent / "v191_decode_gate_up_epilogue/config.json"
trace = HERE.parent / "v261_fused_decode_route_staging/work.json"
score = ROOT / "serving_v0523/runs/abd217baf0ed49fd.json"
control = read(h200)
case = next(r for r in control["cases"] if r["T"] == 8)
assert "H200" in control["gpu"] and case["local_routes"] == 10
config = read(tuning)["configs"]["8192_8"]
assert config["block"] == [8,256,128] and all(config["tma"])
snapshots = read(trace)["snapshot_counts_by_running"]
period = HERE.parent / "v262_decode_phase_budget/check.json"
one = next(a for a in read(period)["cases"] if a["users"] == 1)
live = case["local_routes"]
layers, width, bytes_per_word = 62, 4096, 2
logical = 2 * live * width * bytes_per_word
print(json.dumps(dict(input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (h200, tuning, trace, score, period)}, source=read(HERE/"validation.json")["patch_sha256"],
    kernel_shape=config["block"], eligible_native_DFlash_request_counts=[1],
    archived_scheduler_snapshots={"one_request": sum(r["samples"] for r in snapshots if r["captured_bs"]==1),
                                 "all":sum(r["samples"] for r in snapshots)},
    observed_one_rank={"rows":8,"routed_slots":64,"live":live,"per_layer_removed_staging_bytes":logical,
                       "across_62_layers_bytes":logical*layers,"per_rank_removed_staging_CTAs_per_62_layers":64*layers,
                       "before_after_plan_CTAs":[65,1],"before_after_staging_launches":[1,1],
                       "original_inputs_bytes":8*width*bytes_per_word,"routed_slot_arena_bytes":64*width*bytes_per_word,
                       "retained_W2_output_arena":True,"isolated_native_Marlin_us":case["stock_us"]},
    matrix_issue="Unchanged M8,N256,K128 CTA count and HGMMA count per existing expert job. Eight original rows replace eight staged tile rows. The direct epilogue selects the routed token. All two-way K partials for eight rows are reduced in direct mode.",
    wall_time_scale={"service_period_proxy_ms":one["estimated_period_ms"],"at_4_8_TB_s_staging_payload_us_per_layer":logical/4.8e6,
                     "saving_per_layer_us_needed_for_5pct_period_throughput":1000*one["estimated_period_ms"]*(1-1/1.05)/layers},
    limitations="The H200 histogram comes from a separate one-rank Marlin control. Unique and L2-resident bytes, masked reduction savings forgone, TMA bandwidth, and graph execution time are not measured."),indent=2))
