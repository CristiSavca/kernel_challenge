"""Check token-owned wave bounds and packed route finalization against a direct CPU model."""

import itertools
import json
import math

import numpy as np


def bf16(x):
    bits = np.asarray(x, np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.view(np.float32)


def source_ranges(counts, wave_size):
    prefix = np.cumsum(counts, dtype=np.int64) - counts
    return prefix // wave_size


def reference_ranges(counts, wave_size):
    result = []
    total = 0
    for n in counts:
        result.append(total // wave_size)
        total += int(n)
    return result


def check_bounds():
    enumerated = 0
    largest_slack = 0
    max_bound_seen = False
    for topk, tokens in ((1, 5), (2, 6), (3, 5), (4, 4), (8, 3)):
        for counts in itertools.product(range(topk + 1), repeat=tokens):
            for wave_size in (1, 2, topk, tokens, 2 * tokens):
                waves = reference_ranges(counts, wave_size)
                loads = {}
                for wave, count in zip(waves, counts):
                    if count:
                        loads[wave] = loads.get(wave, 0) + count
                assert sum(loads.values()) == sum(counts)
                assert all(w < math.ceil(topk * tokens / wave_size) for w in loads)
                assert all(v <= wave_size + topk - 1 for v in loads.values())
                largest_slack = max(largest_slack, max(loads.values(), default=0) - wave_size)
                max_bound_seen |= topk == 8 and largest_slack == 7
                enumerated += 1
    assert max_bound_seen
    return enumerated, largest_slack


def route_values(token, slot, expert, hidden):
    cols = np.arange(hidden, dtype=np.int64)[None, :]
    return bf16((((token[:, None] * 1949 + slot[:, None] * 103 + expert[:, None] * 31
                   + cols * 743) % 2081) - 1040).astype(np.float32) * np.float32(2 ** -7))


def final_terms(values, expert, score, weights, weight_first, round_product):
    scales = bf16((expert.astype(np.float32) % 7 + 1) * np.float32(0.1875))
    y = bf16(values * scales[:, None])
    if weight_first:
        y = y * weights
    y = y * (1 / np.sqrt(np.mean(y * y, axis=1, keepdims=True) + np.float32(1e-8)))
    if not weight_first:
        y = y * weights
    v = y * score[:, None]
    return bf16(v) if round_product else v


def check_plan(ids, experts, wave_size, hidden=9, numerical=True):
    rows, topk = ids.shape
    wave_capacity = wave_size + topk - 1
    live = ids >= 0
    counts = np.sum(live, axis=1)
    owners = source_ranges(counts, wave_size)
    assert np.array_equal(owners, reference_ranges(counts, wave_size))
    routed = np.zeros_like(live)
    scores = ((np.arange(rows * topk).reshape(rows, topk) * 397 % 1501) - 750).astype(np.float32) / 997
    weights = bf16((np.arange(hidden, dtype=np.float32) + 3) * np.float32(0.375))
    expected = np.zeros((rows, hidden), np.float32)
    outputs = np.zeros_like(expected)
    writes = np.zeros(rows, np.int32)
    weight_first, round_product = wave_size % 2 == 0, wave_size % 3 == 0
    if numerical:
        for k in range(topk):
            t = np.flatnonzero(live[:, k])
            e = ids[t, k]
            raw = route_values(t, np.full_like(t, k), e, hidden)
            expected[t] += final_terms(raw, e, scores[t, k], weights, weight_first, round_product)

    max_waves = math.ceil(rows * topk / wave_size)
    for wave in range(max_waves):
        assigned = live & (owners == wave)[:, None]
        t, s = np.where(assigned)
        assert len(t) <= wave_capacity
        e = ids[t, s]
        order = np.argsort(e, kind="stable")
        t, s, e = t[order], s[order], e[order]
        rank = wave_capacity - len(t)
        physical = rank + np.arange(len(t))
        offsets = rank + np.searchsorted(e, np.arange(experts + 1))
        # Humming uses the host shape for the end of its last expert.
        vendor_counts = np.diff(np.append(offsets[:-1], wave_capacity))
        np.testing.assert_array_equal(vendor_counts, np.bincount(e, minlength=experts))
        assert np.all(offsets[e] <= physical) and np.all(physical < offsets[e + 1])
        assert not np.any(routed[t, s])
        routed[t, s] = True
        selected = np.unique(t)
        assert np.all(writes[selected] == 0)
        writes[selected] += 1

        if not numerical or not len(t):
            continue
        arena = np.full((wave_capacity, hidden), np.nan, np.float32)
        arena[physical] = route_values(t, s, e, hidden)
        inverse = np.full(ids.shape, -1, np.int64)
        inverse[t, s] = physical
        # The common finalizer reduces every live slot for one token in its original order.
        for k in range(topk):
            token = np.flatnonzero(inverse[:, k] >= 0)
            values = arena[inverse[token, k]]
            outputs[token] += final_terms(values, ids[token, k], scores[token, k],
                                          weights, weight_first, round_product)
    np.testing.assert_array_equal(routed, live)
    np.testing.assert_array_equal(writes, counts > 0)
    if numerical:
        np.testing.assert_array_equal(bf16(outputs).view(np.uint32), bf16(expected).view(np.uint32))
    return int(np.sum(live)), int(writes.sum()), max_waves


def run():
    exact, overlap = check_bounds()
    rng = np.random.default_rng(289013)
    total_routes = finalized = cases = 0
    for topk, experts in ((1, 1), (3, 7), (8, 32), (8, 64)):
        for rows in (1, 7, 31, 32, 33, 65, 129):
            for mode in ("empty", "remote_tail", "last_expert", "balanced", "random", "all_local"):
                x = rng.integers(0, experts, size=(rows, topk), dtype=np.int32)
                if mode == "empty":
                    x.fill(-1)
                elif mode == "remote_tail":
                    x[rows // 2:] = -1
                elif mode == "last_expert":
                    x.fill(experts - 1)
                elif mode == "balanced":
                    x[:, 1:] = -1
                elif mode == "random":
                    x[rng.random(x.shape) < 0.47] = -1
                for chunk in sorted({1, max(1, rows // 2), rows, rows + topk - 1}):
                    n, u, _ = check_plan(x, experts, chunk)
                    total_routes += n
                    finalized += u
                    cases += 1
    large = []
    for tokens in (8192, 16384):
        for mode in ("near_uniform", "hot_rank", "all_remote", "boundary_crossing"):
            ids = rng.integers(0, 32, size=(tokens, 8), dtype=np.int32)
            if mode == "near_uniform":
                ids[rng.random(ids.shape) < 7/8] = -1
            elif mode == "all_remote":
                ids[:] = -1
            elif mode == "boundary_crossing":
                ids[tokens // 8 - 1, 1] = -1
            n, u, w = check_plan(ids, 32, tokens, numerical=False)
            large.append(dict(tokens=tokens, mode=mode, live_routes=n, tokens_with_live_routes=u, static_waves=w))
    return dict(bound_cases=exact, maximum_observed_overshoot=overlap,
                numerical_plans=cases, numerical_live_routes=total_routes,
                numerical_tokens_finalized=finalized, large_plans=large)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
