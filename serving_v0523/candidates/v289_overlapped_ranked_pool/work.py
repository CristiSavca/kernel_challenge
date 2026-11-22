"""Price the V289 dependency change using explicit duration assumptions."""
import hashlib
import itertools
import json
from pathlib import Path

here = Path(__file__).resolve().parent
inputs = {
    "standalone_projection_and_v64_period": here.parent/"v228_sharded_verify_projections/work.json",
    "fixed_interactivity_frontier": here.parent/"v283_rank_major_unpool_boundary/score_target.json",
}
def read(key):
    return json.loads(inputs[key].read_bytes())

def saved(pool_and_gather, route, plan):
    return max(pool_and_gather, route) + plan - max(pool_and_gather, route+plan)

for pool, route, plan in itertools.product(range(17), repeat=3):
    assert saved(pool, route, plan) == min(plan, max(0, pool-route))
control = read("standalone_projection_and_v64_period")
target = read("fixed_interactivity_frontier")
layers = control["sparse_layers"]
rows = []
for rung in control["rungs"]:
    period = rung["v64_approximate_verify_period_ms"]
    def us_per_layer(factor):
        return 1000 * period * (1-1/factor) / layers
    slots = rung["rows"]*8
    rows.append(dict(users=rung["users"], rows=rung["rows"], plan_integer_payload_bytes=(3*slots+33)*4,
                     V64_proxy_verify_ms=period, exposed_overlap_us_per_layer_for_1_63pct_verify=us_per_layer(1.0163),
                     exposed_overlap_us_per_layer_for_uniform_4x_score_verify_scenario=us_per_layer(target["further_uniform_gain"]),
                     native_single_gpu_pool_us=rung["pool_us"], native_single_gpu_router_us=rung["router_us"]))
print(json.dumps(dict(parent="V284",candidate="V289",phase_definition="P = main pool projection and gather, R = gate and route, Q = route planning. The fused norm/staging waits for both input and plan.",
    baseline_segment="max(P,R)+Q",early_segment="max(P,R+Q)",ideal_saved_per_layer="min(Q,max(0,P-R))",
    verified_duration_triplets=17**3,sparse_layers=layers,by_rung=rows,
    scope="Native standalone pool and router measurements are context only. V289 shards its projection and gathers; no trace measures P,R,Q in this graph. Contention can erase overlap. The same four int32 tensors are allocated earlier on eligible graphs. Whole allocator peaks and hardware effects are unknown.",
    input_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in inputs.items()}),indent=2))
