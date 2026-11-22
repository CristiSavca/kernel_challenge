"""Quantify V173's disjoint routing and wave assumptions without profiling challenge requests."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CAND=HERE.parent
E=32
BANDS=32
SMS=132


def work(pairs):
    def total(calc):
        return sum(n*calc(k) for k,n in pairs)
    active=total(lambda k:k>0);small=total(lambda k:1<=k<=8);large=total(lambda k:k>8)
    small_rows=total(lambda k:k if 1<=k<=8 else 0)
    total_rows=total(lambda k:k)
    sweeps=[total(lambda k:math.ceil(k/16)),total(lambda k:math.ceil(k/32))]
    fraction=3*small/(2*sweeps[0]+sweeps[1])
    return dict(local_experts=E,expected_nonempty=active,expected_tiny=small,expected_hot=large,
         local_routes=total_rows,tiny_routes=small_rows,tiny_fraction_of_local_rows=small_rows/total_rows,
         tiny_ctas_per_bank=BANDS*small,hot_ctas_per_bank=BANDS*large,
         nominal_weight_sweeps_w13_w2=sweeps,nominal_weight_bytes_per_rank=9*(2**20)*(2*sweeps[0]+sweeps[1]),
         fraction_nominal_packed_bytes_in_tiny=fraction,
         hypothetical_equally_bandwidth_bound_pair_speedups={
            f"{s}x_tiny":1/(1-fraction+fraction/s) for s in (1.25,2.0)},
         tiny_speed_needed_if_overhead_is_five_percent_of_prior_pair=fraction/(fraction-.05),
         cta_capacity_equivalents_no_ceiling={
            "old_active_ctas_at_8_resident_per_sm":active*BANDS/(SMS*8),
            "new_tiny_ctas_at_4_resident_per_sm":small*BANDS/(SMS*4),
            "new_hot_ctas_at_8_resident_per_sm":large*BANDS/(SMS*8)})


def main():
    patch=(HERE/"patch.diff").read_bytes()
    src=patch.decode()
    assert "const dim3 grid(p.output_width / 128, experts);" in src
    assert "streamed_projection<Gate, true>, p, maps, stride" in src
    assert "muse_medium_route_pair::projection<Gate, true>, p, stride" in src
    assert "constexpr uint32_t kWarps = 2;" in src
    assert "constexpr uint32_t tile_rows = Gate ? 16 : 32;" in src
    diagnostic_path=ROOT/"experiments/iterations/20261104-moe-hardware-profile-recovered.json"
    d=json.loads(diagnostic_path.read_text())
    diag=next(c for c in d["cases"] if c["M"]==256)
    real=diag["routing"]["local_counts"]
    assert len(real)==32 and all(isinstance(x,int) for x in real)
    observed=work([(k,1) for k in real])
    assert observed["tiny_routes"]==59 and observed["expected_hot"]==10 and observed["local_routes"]==354
    uniform={}
    for rows in (128,256):
        # Each target row selects eight DISTINCT experts uniformly from 256.
        # A fixed expert is Bernoulli(8/256) per row; linearity is exact, no i.i.d. expert assumption needed.
        p=8/256
        pairs=[(k,E*math.comb(rows,k)*p**k*(1-p)**(rows-k)) for k in range(rows+1)]
        uniform[str(rows*8)]=work(pairs)
    v64=json.loads((ROOT/"serving_v0523/runs/abd217baf0ed49fd.json").read_text())["result"]
    raw=json.loads((CAND/"v165_streamed_expert_stages/sm90.json").read_text())["targets"]["sm_90a"]["kernels"]
    med=json.loads((CAND/"v141_adaptive_expert_projection/sm90.json").read_text())["release"]["architectures"]["sm_90a"]
    def register_limit(threads,regs):
        regs_per_warp=math.ceil(32*regs/256)*256
        return 65536//((threads//32)*regs_per_warp)
    res={}
    for gate in ("w13","w2"):
        t=raw["streamed_projection_"+gate];m=med[gate.upper()]
        res[gate]={"tiny_inherited_registers":t["registers"],"tiny_static_shared_bytes":t["shared_bytes"],
                   "tiny_register_ceiling_unpartitioned":register_limit(288,t["registers"]),
                   "medium_inherited_registers":m["registers"],
                   "medium_register_ceiling_unpartitioned":register_limit(64,m["registers"])}
    return dict(patch_sha256=hashlib.sha256(patch).hexdigest(),published_v64=dict(id="abd217baf0ed49fd",score=v64["frontier"]["score"],
                        rungs=[{k:r[k] for k in ("concurrency","acceptance","ttft_first_p90_s","ttft_later_p90_s")} for r in v64["rungs"]],
                        observed_serving_expert_histogram=False),
         nominal_layer=dict(grid_per_bank=[BANDS,E],prior_projection_launches=2,mixed_projection_launches=4,
              additional_fast_exit_ctas=2*E*BANDS,identical_routing_weight_tile_counts=True),
         uniform_distinct_top8_synthetic=uniform,
         recorded_standalone_h200={"source_sha256":hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
             "different_kernel":diag["counter_launch"]["kernel"],"diagnostic_only":d["diagnostic_only"],
             "measured_traffic_time":diag["counter_launch"]["metrics"],"geometry":observed},
         compiled_parent_resource_models=res,
         caveats=["These are nominal weight sweep counts, not measured V168/V173 device traffic.",
                  "Resource models give register ceilings. Partitioning, shared memory and occupancy APIs determine achievable residency.",
                  "Fractional CTA capacity equivalents are geometric ratios, not phases of equal time.",
                  "The public diagnostic uses a generated input and a different kernel. V64 serving reports no route histogram."])
if __name__=="__main__":
    print(json.dumps(main(),indent=2))
