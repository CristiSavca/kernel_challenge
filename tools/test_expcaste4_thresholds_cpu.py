#!/usr/bin/env python3
"""Bounded CPU evidence for an exact-code alternative to ExpCast-FP8.

This is research, not a CUDA implementation or a GPU equivalence certificate.
Use NumPy from /tmp/okc-kperm-cpu-20261104. No input/model distributions are used.
"""
import json
import math
import numpy as np


def values():
    c = np.arange(127, dtype=np.int32)
    e, m = c >> 3, c & 7
    return np.where(e == 0, m * 2.0**-9, (1 + m / 8.0) * np.exp2(e - 7))


V = values()
MID = (V[:-1] + V[1:]) * 0.5
LOG = np.log2(MID)
GUARD = 2.0**-18
# Normal-binade midpoint below each of its eight codes. Code 8 is the
# subnormal boundary and is handled by the x >= -6 fast-path restriction.
B = np.log2(np.array([31 / 32] + [(15 + 2 * m) / 16 for m in range(1, 8)]))


def quantize(y):
    c = np.searchsorted(MID, y, side="left")
    at_tie = (c < len(MID)) & (y == MID[np.minimum(c, len(MID) - 1)])
    return c + (at_tie & ((c & 1) != 0))


def examine(x):
    # For float32 x in this interval, float64 evaluates 8*x+56 exactly,
    # so the cast models a single correctly rounded FFMA before F2I.RN.
    affine = (x.astype(np.float64) * 8 + 56).astype(np.float32)
    c0 = np.rint(affine).astype(np.int32)
    threshold = ((c0 >> 3).astype(np.float32) - np.float32(7) + B[c0 & 7].astype(np.float32)).astype(np.float32)
    out = c0 - (x < threshold)
    slow = (np.abs(x.astype(np.float64) - threshold) <= GUARD) | (x < -6) | (x > 8)
    y = np.exp2(x.astype(np.float64))
    q = quantize(y)
    assert np.all((c0 >= q) & (c0 <= q + 1))
    assert np.all(out[~slow] == q[~slow])
    # PTX ex2.approx.f32 permits 2 ulp from a correctly rounded result.
    # A conservative 3 ulp envelope around real exp2 includes initial FP32
    # rounding as well. Outside guard every value in that envelope must
    # round to the same E4M3 code. CPU cannot reproduce actual MUFU output.
    yf = y.astype(np.float32)
    ulp = (np.nextafter(yf, np.float32(np.inf)).astype(np.float64) - yf)
    qlo = quantize(y - 3 * ulp)
    qhi = quantize(y + 3 * ulp)
    assert np.all(qlo[~slow] == qhi[~slow])
    return int(x.size), int(slow.sum()), int(np.count_nonzero(c0 != q))


def main():
    xs = [np.linspace(-6, 8, 1_000_001, dtype=np.float32)]
    # All mathematical E4M3 transitions in the fast range and every affine
    # integer-round transition, with 64 float32 neighbors on each side.
    centers = list(LOG[(LOG >= -6) & (LOG <= 8)])
    centers += [(c + 0.5 - 56) / 8 for c in range(8, 120)]
    boundary = np.array(centers, dtype=np.float32)
    xs.append(boundary)
    lo, hi = boundary.copy(), boundary.copy()
    for _ in range(64):
        lo = np.nextafter(lo, np.float32(-np.inf))
        hi = np.nextafter(hi, np.float32(np.inf))
        xs.extend([lo.copy(), hi.copy()])
    # Include guard boundaries themselves, where the fast path resumes.
    for sign in [-1, 1]:
        edge = (np.array(centers, dtype=np.float32) + sign * GUARD).astype(np.float32)
        xs.extend([edge, np.nextafter(edge, np.float32(-np.inf)), np.nextafter(edge, np.float32(np.inf))])
    x = np.unique(np.concatenate(xs))
    x = x[(x >= -6) & (x <= 8)]
    n, fallback, dec = examine(x)
    print(json.dumps({
        "status": "pass", "samples": n, "guarded_samples": fallback,
        "decrements": dec, "fast_interval": [-6, 8], "guard": GUARD,
        "normal_log_midpoints": B.tolist(),
        "gpu_equivalence_verified": False,
        "scope": "uniform grid plus all threshold/affine boundaries and 64 neighboring float32 inputs; 3-ulp exp2 envelope outside guard",
        "limits": "CPU evidence only, not exhaustive float32 enumeration, no CUDA/SASS/resource/performance test; fallback must execute original ex2/cast",
    }, indent=2))


if __name__ == "__main__":
    main()
