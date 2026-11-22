"""Count nominal phase work for the archived standalone routed histogram and uniform routing."""
from collections import Counter
import hashlib
import json
from math import ceil, comb
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
E, WIDTH, BANDS, SMS = 32,4096,32,132


def bucket_work(counts):
    def total(fn):
        return sum(n * fn(k) for k, n in counts)
    tiny = total(lambda r: 0 < r <= 8)
    hot = total(lambda r: r > 8)
    rows = total(lambda r:r)
    hotrows = total(lambda r:r if r > 8 else 0)
    # The old two-warp vector CTA loads the 128-K input independently for
    # each 64-channel warp. TMA stages it once for both output bands.
    old_input = 2 * WIDTH * BANDS * (rows + hotrows)
    new_input = 2 * WIDTH * BANDS * rows
    def tails(r):
        return 1 if 17 <= r % 32 <= 24 else 0
    saved_down = total(tails)
    new_down = total(lambda r:ceil(r / 8))
    old_down = new_down + saved_down
    gate_sweeps=total(lambda r:ceil(r/16))
    down_sweeps=total(lambda r:ceil(r/32))
    bank_bytes = 9 * 2**20 * (2*gate_sweeps + down_sweeps)
    old_vector_bytes = 9 * 2**20 * (2*(gate_sweeps-tiny)+(down_sweeps-tiny))
    pair_fraction=old_vector_bytes/bank_bytes
    return dict(local_routes=rows,active=tiny+hot,light=tiny,hot=hot,
        hot_route_fraction=hotrows/rows,
        v173_v174_nominal_weight_bytes_per_layer_rank=bank_bytes,
        former_vector_hot_fraction_nominal_weight_bytes=pair_fraction,
        swept_full_banks_per_projection={"W13":gate_sweeps,"W2":down_sweeps},
        nominal_input_bytes_per_projection=dict(V173=old_input,V174=new_input,saved=old_input-new_input),
        relative_input_traffic_saved=(old_input-new_input)/old_input,
        down_8row_mma_groups=dict(V173=old_down,V174=new_down),
        relative_down_mma_groups_saved=saved_down/old_down,
        resident_block_wave_equivalents_per_bank=dict(
            V173_summed_ceilings=tiny*BANDS/(4*SMS) + hot*BANDS/(8*SMS),
            V174=tiny*BANDS/(4*SMS)+hot*BANDS/(4*SMS)),
        hypothetical_pair_throughput_if_only_former_vector_sweeps_double=1/(1-pair_fraction/2))


def main():
    historical=ROOT/"experiments/iterations/20261104-moe-hardware-profile-recovered.json"
    sample=next(x for x in json.loads(historical.read_text())["cases"] if x["M"]==256)
    uniform={}
    for n in (64,128,256):
        p=8/256
        uniform[str(n*8)]=bucket_work([(k,E*comb(n,k)*p**k*(1-p)**(n-k)) for k in range(n+1)])
    return dict(per_rank_shape=dict(experts=32,hidden=4096,inner=4096,bands=32),
        eligible_projection_pair=dict(previous_grid_launches=4,new_grid_launches=2,ctas_before=4096,ctas_after=2048,
                                      nominal_full_weight_sweeps_same=True),
        compiled_unpartitioned_residency_upper_bound=dict(V173_tiny=4,V173_hot=8,V174=4,
                 assumptions="132 SMs, 65536 registers and 228 KiB shared per SM, 256-register warp allocation"),
        uniform_top8_without_replacement=uniform,
        standalone_h200_diagnostic=dict(source=historical.name,sha256=hashlib.sha256(historical.read_bytes()).hexdigest(),
                  counts=sample["routing"]["local_counts"],model=bucket_work(Counter(sample["routing"]["local_counts"]).items())),
        limitations=["Input bytes count source requests from the projection's own lanes; caches and transaction coalescing can reduce DRAM bytes.",
            "The two former grids have different work per resident CTA. Wave-equivalent sums are not elapsed time.",
            "New W2 uses two ring buffers, whereas the parent tiny W2 used three. Lost lookahead can outweigh launch or warp improvements.",
            "The observed histogram uses a different standalone kernel; the scored serving receipt contains no expert histogram."])
if __name__=="__main__":
    print(json.dumps(main(),indent=2))
