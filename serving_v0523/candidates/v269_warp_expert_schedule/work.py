"""Replay route geometry and persistent CTA assignments for the two expert projections."""
from bisect import bisect_right
import hashlib
from itertools import accumulate
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROFILE = ROOT / "experiments/iterations/20261104-moe-hardware-profile-recovered.json"
N_TILES = 32
SMS = 132
EXPERT_MIB = 27


def replay(counts, width):
    tiles = [(c + width - 1) // width for c in counts]
    ends = list(accumulate(tiles))
    active = [t for t in tiles if t]
    native = [0] * SMS
    grouped = [0] * SMS
    traversals = 0
    lookups = 0
    for cta in range(SMS):
        old_expert = 0
        for job in range(cta, sum(tiles) * N_TILES, SMS):
            expert = bisect_right(ends, job // N_TILES)
            traversals += expert - old_expert
            lookups += 1
            old_expert = expert
            native[cta] += 1
        for job in range(cta, len(active) * N_TILES, SMS):
            grouped[cta] += active[job // N_TILES]
    assert sum(native) == sum(grouped)
    return dict(live_routes=sum(counts), live_experts=len(active), native_tiles=sum(tiles),
                native_width=width, longest_expert_tiles=max(tiles), jobs_per_projection=lookups,
                incumbent_expert_advances_per_projection=traversals,
                incumbent_shared_loop_condition_checks=traversals+lookups,
                V269_ballots_per_projection_per_warp=lookups,
                total_warps_per_CTA=20, native_max_serial_row_tiles=max(native),
                all_rows_per_expert_single_owner_round_robin_max_serial_tiles=max(grouped),
                all_rows_per_expert_independent_jobs=len(active)*N_TILES,
                V269_theoretical_CTA_occupancy_same=True,
                incumbent_weight_requests_mib=sum(tiles)*EXPERT_MIB,
                all_rows_reused_weight_requests_mib=len(active)*EXPERT_MIB,
                all_rows_reused_saved_weight_requests_mib=(sum(tiles)-len(active))*EXPERT_MIB,
                savings_additional_to_v216_pairing_mib=(sum(tiles)-len(active)-sum(t//2 for t in tiles))*EXPERT_MIB)


def model(users):
    rows = users * 8
    width = {1:8,4:8,16:16,32:24}[users]
    # Marginal probability for a given expert under independent distinct top-8 routing.
    p = 8 / 256
    pmf = [math.comb(rows,k)*p**k*(1-p)**(rows-k) for k in range(rows+1)]
    tiles = 32 * sum((k+width-1)//width * prob for k,prob in enumerate(pmf))
    active = 32 * (1-pmf[0])
    return dict(users=users, verify_rows=rows, total_route_slots=rows*8,
                native_tile_width=width, expected_live_experts=active, expected_native_tiles=tiles,
                expected_jobs=tiles*N_TILES, expected_excess_weight_request_mib=(tiles-active)*EXPERT_MIB)


recorded = []
for case in json.loads(PROFILE.read_bytes())['cases']:
    rows = case['M']
    metrics = case['counter_launch']['metrics']
    data = replay(case['routing']['local_counts'], {32:8,256:24}[rows])
    data['standalone_input_rows'] = rows
    data['separate_kernel_H200_dram_read_mib'] = metrics['dram__bytes_read.sum']['value'] / 2**20
    data['other_kernel_over_unique_mib'] = data['separate_kernel_H200_dram_read_mib'] - data['all_rows_reused_weight_requests_mib']
    recorded.append(data)

print(json.dumps(dict(physical_input=PROFILE.relative_to(ROOT).as_posix(),
    sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),recorded=recorded,
    independent_uniform=[model(i) for i in (1,4,16,32)], kernel_uses_runtime_histogram=True,
    full_CTA_barriers_saved_per_projection=2*SMS, full_CTA_barriers_saved_per_62_layer_target_step_per_rank=2*SMS*2*62),indent=2))
