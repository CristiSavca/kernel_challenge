"""Reproduce the lossless-FP8 checks and H200 prefill crossover estimates."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "experiments/iterations/20261104-moe-hardware-profile-recovered.json"
BF16_TFLOPS = 989.5
HBM_TBPS = 4.8
SMS = 132
H = I = 4096
TOPK = 8
EXPERTS = 256
LOCAL_EXPERTS = 32
BM = 176
BN = 256
MIB = 1024**2


def fp8_positive(mantissa_bits, bias, last_code):
    unit = 2**mantissa_bits
    return {
        math.ldexp((code % unit) / unit + (code >= unit), max(code // unit, 1) - bias)
        for code in range(last_code + 1)
    }


def significand_bits(value):
    if value == 0:
        return 0
    n = abs(value).as_integer_ratio()[0]
    return n.bit_length() - ((n & -n).bit_length() - 1)


def dot_f32(a, b):
    total = np.float32(0)
    for x, y in zip(a, b):
        # Products in the witness are exactly representable in FP32.
        total = np.float32(total + np.float32(x * y))
    return float(total)


def representation():
    e4 = fp8_positive(3, 7, 126)
    e5 = fp8_positive(2, 15, 123)
    raw_scales = sorted(e4)
    magnitudes = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    raw_codes = [sign * m for sign in (-1, 1) for m in magnitudes]
    # V64 and V102 flush input E4M3 subnormal block scales, matching Marlin.
    w = [v * (s if s >= 2**-6 else 0.0) for v in raw_codes for s in raw_scales]
    precision = {str(b): sum(significand_bits(v) == b for v in w) for b in range(7)}
    a = [73.5, 2**-9, -73.5]
    hi = [72., 2**-9, -72.]
    lo = [1.5, 0., -1.5]
    weights = [448., 1., 448.]
    assert all(abs(v) in e4 for v in hi + lo + weights)
    assert [x + y for x, y in zip(hi, lo)] == a
    assert all(v in w for v in weights)
    assert 1.375 in raw_scales and 2.0625 not in e4
    reference = dot_f32(a, weights)
    split = float(np.float32(dot_f32(hi, weights)) + np.float32(dot_f32(lo, weights)))
    assert reference == 0 and split == 2**-9
    assert max(significand_bits(v) for v in w) == 6
    return {
        "signed_code_times_original_scale_pairs": len(w),
        "prepared_subnormal_scales": "zero, as in pinned V64",
        "significand_bits_by_pair": precision,
        "single_unscaled_e4m3_pairs": sum(abs(v) in e4 for v in w),
        "single_unscaled_e5m2_pairs": sum(abs(v) in e5 for v in w),
        "outside_e4m3_precision_at_any_power_of_two_scale": sum(significand_bits(v) > 4 for v in w),
        "outside_e5m2_precision_at_any_power_of_two_scale": sum(significand_bits(v) > 3 for v in w),
        "single_e4m3_BF16_mantissas_in_binade": sum(1 + j / 128 in e4 for j in range(128)),
        "BF16_mantissas_in_binade": 128,
        "weight_precision_witness": {"weight_code_value": 1.5, "block_scale": 1.375, "product": 2.0625},
        "illustrative_sequential_full_K_fp32_witness": {
            "nonzero_k": [0, 16, 32], "input": a, "hi": hi, "lo": lo, "weights": weights,
            "full_K_fp32": reference, "two_separate_fp32_dots": split,
            "both_results_exactly_BF16": True,
        },
    }


def uniform_routes(rng, n):
    draws = rng.integers(EXPERTS, size=(n, TOPK), dtype=np.int16)
    while True:
        repeated = (np.diff(np.sort(draws, axis=1), axis=1) == 0).any(axis=1)
        if not repeated.any():
            return draws
        draws[repeated] = rng.integers(EXPERTS, size=(repeated.sum(), TOPK), dtype=np.int16)


def model(rows, trials):
    rng = np.random.default_rng(112000 + rows)
    out_cols = (2 * I // BN, H // BN)
    totals = []
    for _ in range(trials):
        hist = np.bincount(uniform_routes(rng, rows).ravel(), minlength=EXPERTS)
        for ep_rank in range(8):
            routes = hist.reshape(8, LOCAL_EXPERTS)[ep_rank]
            expert_tiles = (routes + BM - 1) // BM
            mt = int(expert_tiles.sum())
            paired_mt = int(((expert_tiles + 1) // 2).sum())
            groups = int((routes > 0).sum())
            tokens = int(routes.sum())
            ctas = [n * mt for n in out_cols]
            wave_count = [math.ceil(count / SMS) for count in ctas]
            active_rows = tokens * 6 * H * I
            issued_rows = mt * BM * 6 * H * I
            packed_B_per_expert = H * 3 * I * (1 / 2 + 1 / 16)
            totals.append([
                tokens, mt, *ctas, *wave_count, groups * packed_B_per_expert,
                mt * packed_B_per_expert, tokens * (H + I) * 2,
                tokens * 2 * (H * out_cols[0] + I * out_cols[1]),
                mt * BM * 2 * (H * out_cols[0] + I * out_cols[1]),
                tokens * (2 * I + H) * 2, active_rows, issued_rows, paired_mt,
            ])
    (tokens, tiles, c1, c2, waves1, waves2, unique_b, request_b, unique_a,
     useful_request_a, padded_request_a, writes, useful_flops, issued_flops,
     paired_tiles) = np.mean(totals, axis=0)
    compute_ms = issued_flops / (BF16_TFLOPS * 1e9)
    # Use live A reads; TMA's logical row tails can request additional neighboring rows.
    def time_ms(multicast, miss_a, miss_b, compute_eff, memory_eff):
        requested_a = useful_request_a / multicast
        hbm = (unique_a + unique_b + writes + miss_a * (requested_a - unique_a)
               + miss_b * (request_b - unique_b))
        return max(compute_ms / compute_eff, hbm / (HBM_TBPS * memory_eff * 1e9))

    scenarios = {}
    for compute_eff in (0.35, 0.6, 0.9, 1):
        for miss_b in (0, 1):
            for miss_a in (0, 0.5, 1):
                before = time_ms(1, miss_a, miss_b, compute_eff, 0.75)
                after = time_ms(2, miss_a, miss_b, compute_eff, 0.75)
                k = f"tensor_eff_{compute_eff}_B_duplicate_miss_{miss_b}_A_duplicate_miss_{miss_a}"
                scenarios[k] = round(before / after, 4)
    return {
        "rows": rows, "rank_samples": len(totals), "mean_routed_rows_rank": round(tokens, 2),
        "mean_M_tiles": round(tiles, 3),
        "mean_W13_W2_CTAs": [round(c1, 2), round(c2, 2)],
        "mean_W13_W2_waves": [round(waves1, 2), round(waves2, 2)],
        "ideal_final_wave_packing_speedup_only": [
            round(waves1 / (c1 / SMS), 4), round(waves2 / (c2 / SMS), 4)
        ],
        "ideal_elimination_of_all_M_padding_ratio": round(issued_flops / useful_flops, 4),
        "useful_compute_peak_floor_ms": round(useful_flops / (BF16_TFLOPS * 1e9), 4),
        "padded_compute_peak_floor_ms": round(compute_ms, 4),
        "A_unique_W13_plus_W2_MiB": round(unique_a / MIB, 2),
        "A_requested_live_only_before_multicast_MiB": round(useful_request_a / MIB, 2),
        "A_requested_full_tiles_before_multicast_MiB": round(padded_request_a / MIB, 2),
        "B_unique_packed_MiB": round(unique_b / MIB, 2),
        "B_logical_requested_MiB": round(request_b / MIB, 2),
        "hypothetical_safe_B_multicast_pair_requested_MiB": round(request_b * paired_tiles / tiles / MIB, 2),
        "hypothetical_B_pair_allocated_CTA_slots_per_real_tile": round(2 * paired_tiles / tiles, 4),
        "C_MMA_output_bytes_MiB": round(writes / MIB, 2),
        "A_multicast_2x_roofline_ratios_no_sync_overhead": scenarios,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=32)
    args = ap.parse_args()
    profile = json.loads(PROFILE.read_text())
    assert profile["validation"]["profiled_sizes"] == [32, 256]
    counters = {}
    for case in profile["cases"]:
        values = case["counter_launch"]["metrics"]
        counters[case["M"]] = {
            "cold_DRAM_read_over_nominal_packed_requests": round(
                values["dram__bytes_read.sum"]["value"] / case["routing"]["nominal_request_bytes"], 4
            ),
            "reported_DRAM_pct": values["dram__throughput.avg.pct_of_peak_sustained_elapsed"]["value"],
        }
    result = {
        "status": "CPU exact representation audit and synthetic roofline model",
        "FLOP_model": "two FLOPs per FMA, 989.5 dense BF16 TFLOP/s, 1979 FP8 TFLOP/s",
        "HBM_model": "4.8 TB/s H200 SXM, 132 SMs, 1 CTA/SM, 176x256x64",
        "FP8_break_even": "FP8 effective fraction > dense_BF16_effective_fraction * number_of_FP8_MMA_products / 2, excluding all conversion and rescaling",
        "representation": representation(),
        "controlled_cold_standalone_H200": {
            "profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(), "data": counters,
        },
        "geometry": [model(rows, args.trials) for rows in (8192, 12288, 16320, 16384)],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
