"""Price equal-output decode tiles on the archived routing fixture at matched geometry."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
source_hashes = {}
def read(pattern):
    path, = here.parent.glob(pattern)
    source_hashes[str(path.relative_to(here.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_bytes())

exposure = read("v297_*/independent_exposure.json")
tiles = read("v275_*/work.json")
cache = read("v278_*/cache_sensitivity.json")
phase = read("v262_*/check.json")
r = next(case for case in tiles["recorded"] if case["input_rows"] == 256)
assert r["M"] == 24 and r["expert_row_tiles"] == 33
cases = {}
for name, count in (("neither_projection_widens",0),("one_projection_widens",1),("both_projections_widen",2)):
    native, wide = r["native_MN_jobs_per_projection"], r["selected_MN_jobs_per_projection"]
    avoided = count * r["saved_activation_TMA_MiB_per_projection"]
    cases[name] = dict(total_tile_jobs=(2-count)*native+count*wide,
        total_tile_jobs_saved=count*(native-wide), logical_input_TMA_MiB_saved=avoided,
        unique_expert_weights_or_scales_skipped=0, all_expert_dot_products_retained=True)
samples = [r["avoided_cache_miss_MiB"] for r in cache["cases"] if r["capacity_MiB"] == 50]
period = next(r["estimated_period_ms"] for r in phase["cases"] if r["users"] == 32)
layers = phase["target_layers"]
assert layers == 62
print(json.dumps(dict(initial_32_user_fixture=dict(expert_row_tiles=r["expert_row_tiles"],shape_rows=256,
    original_tile_jobs_across_W13_W2=2*r["native_MN_jobs_per_projection"],cases=cases),
    caches=dict(assumed_L2_MiB=50,logical_A_miss_MiB_avoided_range=[min(r["A"] for r in samples),max(r["A"] for r in samples)],
                logical_weight_scale_miss_MiB_avoided_range=[min(r["bank"] for r in samples),max(r["bank"] for r in samples)],
                assumptions="V278 conventional equal geometry, serial issue orders and 128-byte lines. The fused pipeline and physical H200 cache behavior can differ."),
    per_load_V41_scheduler_reports=exposure["scheduler_reports"],
    high_load_sensitivity=dict(V64_32user_period_proxy_ms=period,layers=layers,
        net_us_saved_per_layer_for_five_percent_faster_decode=period*(1-1/1.05)*1000/layers),
    scope="All tile counts require successful compilation, exact supported geometry, and actual dispatch. A scheduler sample is neither a time fraction nor a throughput measurement. The copy figures count TMA operand requests before caches. No H200 timing or score for this candidate.",
    sources_sha256=source_hashes),indent=2))
