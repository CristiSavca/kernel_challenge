"""Expected per-rank costs for independent uniform top-k, before memory caching."""

import json
import math
from pathlib import Path

MIB = 1024**2
RANKS, EXPERTS, TOPK = 8, 256, 8
HIDDEN, INNER, CLUSTER, TILE, WARP_COLUMNS = 4096, 4096, 8, 16, 64
PEAK_BYTES_PER_SECOND = 4.8e12


def geometry(tokens):
    p = TOPK / EXPERTS
    probabilities = [
        math.comb(tokens, count) * p**count * (1 - p)**(tokens - count)
        for count in range(tokens + 1)
    ]
    live = tokens * TOPK / RANKS
    active = EXPERTS / RANKS * (1 - probabilities[0])
    tiles = EXPERTS / RANKS * sum(
        ((count + TILE - 1) // TILE) * prob for count, prob in enumerate(probabilities)
    )
    # Each block of 16 NVFP4 weights occupies eight code bytes and one scale byte.
    bank = 3 * HIDDEN * INNER * 9 // 16
    row_bytes = live * INNER * 2
    assert abs(sum(probabilities) - 1) < 1e-12
    warps_per_rank_band = HIDDEN // (CLUSTER * WARP_COLUMNS)
    return {
        "target_rows": tokens, "expected_local_rows": live, "expected_distinct_experts": active,
        "expected_16row_tiles": tiles, "expected_worst_case_no_reuse_weight_mib_by_tile": tiles * bank / MIB,
        "expected_distinct_packed_banks_mib": active * bank / MIB,
        "cold_bank_floor_us_at_peak_hbm": 1e6 * active * bank / PEAK_BYTES_PER_SECOND,
        "one_gate_and_activation_store_and_read_mib": 6 * row_bytes / MIB,
        "remote_dsm_operand_reads_mib_before_transaction_effects": row_bytes / MIB * (CLUSTER - 1) * warps_per_rank_band,
        "parent_reserved_auxiliary_arenas_saved_mib": tokens * TOPK * 2 * INNER * 2 / MIB,
        "cluster_barriers_per_live_tile": 2,
        "w13_activation_w2_launches_per_eligible_layer": [3, 1],
    }


def main():
    root = Path(__file__).resolve().parents[3]
    scored = json.loads((root / "serving_v0523/runs/abd217baf0ed49fd.json").read_text())
    score = scored["result"]["frontier"]["score"]
    assert score == 1.4119
    result = {"uniform_route_assumption": True, "hardware_measured": False,
              "cold_bank_assumption": "one read per touched expert, no cache residency on entry",
              "remote_read_unit": "valid BF16 payload requested by down_rows warps; not hardware transactions",
              "bandwidth_reference": "https://www.nvidia.com/en-us/data-center/h200/",
              "shape": [geometry(rows) for rows in (128, 256)],
              "four_load_sensitivity_if_only_two_high_load_ratios_change": {
                  "V64_score": score, "4x_required_factor_at_each_high_load": (4/score)**2,
                  "if_both_high_load_ratios_double": score * math.sqrt(2)}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
