# Attention research reset, November 4, 2026

Decision: do not allocate a GPU check to an approximate ExpCast port or another
attention role rewrite. A mathematical exact-code variant is possible, but the
additional instruction and fallback costs do not support prioritizing it over
the broad MoE/dense work needed to close the roughly 15% decoder latency gap.
This is a prioritization decision, not a measured claim that the variant is slow.

The inspected standalone incumbent is `muse_attention_decode/submission.py`,
SHA `87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c`.
No candidate, active submission or official evaluator was modified by this work.

## Recent primary evidence

**VC-Attention**, arXiv 2609.15810v1, September 14, 2026:
https://arxiv.org/html/2609.15810v1

Section 3.3 maps a base-2 log score directly to an E4M3 byte using an affine
approximation. Its match rate against exponentiation plus casting is 79.6%
within a normal binade; the remaining values differ by one code. Its normalized
row bound is 3.64% plus an underflow contribution. That is additional numerical
approximation, not an equivalent replacement. The paper evaluates video DiTs,
and its H200 speedups combine mechanisms against a different baseline. V-Smooth
reorders and requantizes values, while our input cache has prescribed contents.
The useful lead is avoiding scalar exponential/conversion work when the exact
required output code can instead be established cheaply. No reported paper
speedup is used as a prediction for this decoder.

**FoldAttention**, arXiv 2609.33410v1, September 27, 2026:
https://arxiv.org/html/2609.33410v1

Section 6.2/Table 2 attributes its improvement to gated refinement/value reads.
The both-plane variants are approximately 0.98–1.01x the fastest BF16 kernels;
reading every key still permits omitting a refinement plane. That is not our
required fixed E4M3 cache computation. Appendix D.7 also reports 0.58–0.84x
against the fastest library for eight-draft verification, relevant negative
evidence for our q_len=8 workload. Its fixed-reference formulation is useful
background, but our parent already avoids routine maximum reductions with a
lazy reference and packed-probability overflow test. No FoldAttention kernel
rewrite is recommended from these results.

**NVIDIA PTX ISA**, ex2.approx.f32 specification, read November 4:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2

The hardware instruction has a specified maximum error of two ulps from the
correctly rounded result. Therefore mathematical exp2-to-E4M3 equivalence is
insufficient to claim bit equivalence to the existing hardware sequence near a
rounding boundary. Actual CUDA compilation and a device boundary oracle would
still be required.

## Exact-code derivation and bounded CPU experiment

For a normal positive E4M3 code `c=8e+m`, the value is
`2^(e-7)*(1+m/8)`. On a binade with `f=frac(x)`, convexity gives
`8*(2^f-1) <= 8*f`, with a maximum gap below one. The unbiased affine estimate
`c0 = round_to_even(8*x+56)` can consequently overestimate the true rounded
E4M3 code of `2^x` by at most one, away from the special exponent endpoints.

The log midpoint below candidate code `c0` is
`t = (c0>>3)-7 + b[c0&7]`, with

```
b = log2([31/32, 17/16, 19/16, 21/16, 23/16, 25/16, 27/16, 29/16])
```

Decrement `c0` if `x<t`. Exact ties need the even-code rule. A conservative
candidate design would execute the original hardware exponent-and-cast in a
`2^-18` neighborhood of `t`, and outside `-6 <= x <= 8`. This preserves the
subnormal regime, special values, saturation, and the parent's detection of
probabilities above 256. It also avoids claiming that a CPU library reproduces
MUFU's approximation.

`tools/test_expcaste4_thresholds_cpu.py` tests the affine/code bracket and
correction on 1,028,417 float32 inputs: a dense uniform grid, every mathematical
E4M3 transition and affine integer transition, 64 adjacent float32 values on
each side, and guard boundary neighbors. Outside the guard, all outputs in a
conservative three-FP32-ulp envelope around mathematical exp2 round to the same
E4M3 code. There were zero assertion failures; 5,011 deliberately boundary-heavy
samples were guarded, and 476,396 samples needed a decrement. Those fractions
are not estimates of a model's activation distribution.

Evidence: `experiments/iterations/20261104-expcaste4-thresholds-cpu.json`.
This is bounded CPU evidence, not exhaustive float32 enumeration, a formal proof
of the compiled helper, GPU bit-equivalence, a resource check, or a speed test.
It does not authorize replacing the fallback with zero or skipping any key.

## Why this is not the next GPU experiment

The parent generates FP8 probability tiles in the q_len=8 PV8 path around
lines 3436–3523. Q_len=1 uses FP16 probability operands, so an E4M3 replacement
does not cover all attention workloads. Both PK2 and non-PK2 paths must preserve
their existing rescale order. The parent exponent input is
`fmaf(score, scale, -offset)`, so a new mapper must consume that exact rounded
float; folding its affine operation into the score FMA could move boundaries.

The parent starts the reference at the first tile's maximum, hence initial
probabilities peak at 1, whereas VC-Attention scales its maximum to 256. The
`x<-6` fallback can therefore be common. A per-lane fallback does not avoid a
warp's MUFU issue if any lane needs it. Raising the probability scale would be
a separate numerical/range modification and cannot be quietly bundled in.

Estimated common-path work includes a float FMA and integer conversion,
mantissa mask and exponent extraction, indexed table lookup or shuffle,
integer-to-float conversion, threshold addition, comparison and decrement,
guard/range checks, and byte packing. This is roughly 9–12 scalar instructions
before fallback/packing details, versus one MUFU plus half a packed conversion
per element in the current implementation. This is an instruction inventory,
not measured SASS. The parent already records EXPPOLY as slower because of
issue pressure (line 687). Extra lookup and register lifetimes could also cause
spills in the current heavily specialized consumer.

The exact-code mechanism remains a secondary research option if a future
device profile establishes a scalar-exponential bottleneck and adequate issue
headroom. Its first necessary device test would compare all boundary-neighbor
codes against the original helper and inspect generated instructions/resources,
before one full private attention check with identical parent flags. Abort on
any changed code outside the deliberately preserved fallback, new spills, or
no useful q8 gain. No such device test was run here. The current recommendation
is to use the next official check on the separately reviewed MoE/dense mechanism.
