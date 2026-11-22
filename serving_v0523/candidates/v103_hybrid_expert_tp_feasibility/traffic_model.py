"""Estimate ideal per-rank routed work and payload traffic for TP8 expert placements."""

import json

import numpy as np


def distinct_rows(rng, rows, count, population):
    draws = rng.integers(population, size=(rows, count), dtype=np.int16)
    for _ in range(100):
        duplicates = (np.diff(np.sort(draws, axis=1), axis=1) == 0).any(axis=1)
        if not duplicates.any():
            return draws
        draws[duplicates] = rng.integers(
            population, size=(duplicates.sum(), count), dtype=np.int16
        )
    raise AssertionError("duplicate draws did not converge")


def model(rows, trials, seed):
    rng = np.random.default_rng(seed)
    names = ("uniform", "four_hot_contiguous", "same_draws_hot_striped")
    tensor_widths = (1, 2, 4, 8)
    stats = np.zeros((len(names), len(tensor_widths), 2))
    for _ in range(trials):
        uniform = distinct_rows(rng, rows, 8, 256)
        hot = np.hstack(
            (distinct_rows(rng, rows, 4, 32), 32 + distinct_rows(rng, rows, 4, 224))
        )
        # Redistribute identical logical draws across ranks via a physical ID permutation.
        striped = (hot % 8) * 32 + hot // 8
        for i, routes in enumerate((uniform, hot, striped)):
            weights = np.bincount(routes.ravel(), minlength=256)
            for j, width in enumerate(tensor_widths):
                groups = weights.reshape(8 // width, -1)
                active = (groups > 0).sum(axis=1) / width
                work = groups.sum(axis=1) / width
                assert active.mean() == (weights > 0).sum() / 8
                assert work.mean() == rows
                stats[i, j] += (active.max(), work.max())
    stats /= trials
    result = {}
    for i, name in enumerate(names):
        result[name] = {}
        for j, width in enumerate(tensor_widths):
            banks, work = stats[i, j]
            result[name][f"EP{8 // width}_expertTP{width}"] = {
                "busiest_unique_weight_MiB": round(27 * banks, 3),
                "busiest_ideal_weight_ratio_vs_EP8": round(stats[i, 0, 0] / banks, 5),
                "busiest_full_expert_equivalent_routes": round(work, 3),
                "busiest_ideal_compute_ratio_vs_EP8": round(stats[i, 0, 1] / work, 5),
            }
    return result


def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def main():
    # Every nonzero weight is +1, representable as E2M1 times an E4M3 unit scale.
    x = np.array([256, 1, -256, 1], np.float32)
    full = bf16(x.sum())
    partial = bf16(bf16(x[:2].sum()) + bf16(x[2:].sum()))
    assert float(full) == 2 and float(partial) == 1
    result = {
        "synthetic": True,
        "seed_by_shape": "103000+rows",
        "unit_weight_MiB": 27,
        "bf16_W2_partial_rounding": {"full_K": float(full), "premature_BF16": float(partial)},
        "geometry": {},
    }
    for rows in (8, 32, 128, 256, 16384):
        trials = 64 if rows == 16384 else 1024
        traffic = {}
        for width in (1, 2, 4, 8):
            # A balanced expert group sees rows * width valid routes at global top-8.
            traffic[f"EP{8 // width}_expertTP{width}"] = {
                "intact_K_activation_allgather_incoming_MiB":
                    rows * (width - 1) * 4096 * 2 / 2**20,
                "standard_FP32_W2_ring_allreduce_incoming_MiB":
                    rows * 2 * (width - 1) * 4096 * 4 / 2**20,
            }
        result["geometry"][str(rows)] = {
            "trials": trials,
            "routing": model(rows, trials, 103000 + rows),
            "communication": traffic,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
