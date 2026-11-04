# Exact NVFP4 decode: no supported broad improvement beyond KPERM

2026-11-04, CPU/source audit only. Parent standalone is exact `ef7b210b…`;
decoder uses the corresponding `MOE_MULDEC=True` path. No candidate, GPU work,
submission, or polling. Decision: **do not implement a scale/3-scale or universal
table rewrite from this evidence**. The algebra works, but no concrete mapping
has fewer credible instructions without extra memory traffic/register pressure.
This does not prove the current decoder instruction sequence globally optimal.

## What is already implemented

In `muse_moe_w4a16/candidates/public_cuda_v6.py`:

- `dq4` (675–707) converts one losslessly bit-packed 32-bit word into four
  BF16x2 register-A operands. MAIN selects **MULDEC=1** through `_prepare_layout`
  and `_ext`; the DQ7 branch is bypassed even though DQ7 is also defined. MULDEC
  already uses two funnel rotations and an integer multiply by 4228 to spread
  the remaining code bits. Reintroducing DQ7 is not a new decoder design.
- `wgs_seg` (3065–3076) performs **four packed BF16 multiplies for eight codes**.
  WGSSC2 already combines two rows' E4M3 scale conversion into one PRMT plus
  multiply-add expression, with half broadcasts folded into the multiplies.
- The scale product is not repeatedly rounded: decoded code operands represent
  FP4×2^-126, scales represent E4M3×2^64, and the epilogue restores 2^62 together
  with the original FP32 global scale. Keep that scaling/accumulation behavior.
- KPERM3 already reuses four scale pairs across the four load groups, removing
  three scale-load instructions and twelve scale-pair constructions per stage
  at source level. It deliberately retains `dq4` and all four HMUL2 operations.
  Recommending scale-conversion reuse again would duplicate that experiment.
- `_scales_prmt_exact` (3833) accepts only positive normal bytes 0x08–0x7E.
  Other scale encodings select the parent's LUT/non-WGS fallback. The current
  LUT converts the scale only; it does not tabulate every code×scale product.

The active MULDEC expressions contain eight elementary integer operations before
the four packed multiplies; compiler fusion/scheduling can change the generated
count. The source's DQ7 ALU comments are not measured instruction attribution.

## Exactness of the proposed two-base factorization

Magnitude codes are `[0, .5, 1, 1.5, 2, 3, 4, 6]`. With bases S and 3S, their
outputs are `[0, S/2, S, (3S)/2, 2S, 3S, 4S, 2*(3S)]`, followed by code sign.
This is mathematically exact, including finite E4M3 subnormals: the largest odd
significand is 15×3=45, requiring only six significant bits, below BF16's eight.
The parent's nonzero scaled operands range from 2^-72 to 2688×2^-62, safely
normal BF16. Intermediate FP4×2^-126 contains BF16 subnormals, which must survive.

`python3 tools/audit_nvfp4_decode_math.py` independently checks **4,064 products**
(all 254 finite E4M3 byte patterns ×16 FP4 codes), including **224 products with
nonzero subnormal scales**, signed zero handling, the current exponent trick,
and the S/3S factorization with exact rational arithmetic. It finds zero BF16
rounding discrepancies. Evidence is
`experiments/iterations/20261104-nvfp4-decode-math.json`; this proves operand
arithmetic, not generated CUDA instructions, GPU results, or end-to-end speed.

The normal-scale PRMT shortcut cannot simply be generalized. For E4M3 byte
0x01, the correct scale is 1/512, while the normal rebias produces 9/1024;
byte0 produces 1/128 instead of zero. Signed scales and NaN encodings likewise
must retain appropriate handling. The two NaN byte patterns are outside the
finite-value proof; preserve the existing fallback instead of claiming equality.
Also, 79 of 119 positive normal scale codes have 3S outside representable E4M3
values. Storing 3S in another E4M3 byte is not a generally exact transformation.

The primary [PTX half-precision multiply specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-mul)
defines packed BF16 multiplication, nearest-even rounding, and default subnormal
support. Exact finite products make rounding benign here; introducing flush-to-zero
would break the parent's internal representation.

## Why the proposed replacements are not a credible win yet

**Register S/3S.** The current packed multiply resolves magnitude scaling, sign,
and zero for two independent values in one operation. Replacing it needs each
half to choose S versus 3S, apply its exponent offset, preserve sign, and zero
code0. Code1 is a special case for an odd-code selector: it means S/2, not 3S/4.
These operations can share masks or fuse, but no reviewed sequence saves issue
work over the existing multiply plus unpack. Four bases cached alongside
KPERM's four scale pairs add live registers; fully tabulating eight magnitudes
for both row scales and all four reused steps takes 32 registers, versus four
scale-pair registers. This competes with accumulators and asynchronous A operands
in the already constrained kernel. These are design costs, not a lower-bound
proof against all future encodings.

**Universal lookup.** A fixed format table would not cache input results, but
still needs selection/load work. Its exact storage options are:

| Table, BF16 entries | Bytes | Current-ring implication |
| --- | --- | --- |
| 256 scale bytes ×16 code values | 8,192 | Exceeds parent dynamic headroom |
| Signs factored: 128 scale magnitudes ×8 code magnitudes | 2,048 | Fits if replacing the existing 1,024-B LUT |
| Full same-scale pair: 256 scales ×256 code pairs, BF16x2 | 262,144 | Does not fit |
| Pair with signs factored: 128 scales ×64 magnitude pairs | 32,768 | Does not fit |

The parent uses 231,168 B dynamic shared memory with a 232,448-B opt-in limit.
Replacing its existing 1-KiB LUT with the 2-KiB scalar table totals 232,192 B,
so **not every LUT is excluded by capacity**. However, that version introduces
eight independently indexed scalar lookups per eight weights, plus sign/index
handling, replacing four register-only packed multiplies. It violates the lead's
no-extra-shared-traffic objective and adds load latency/bank-conflict exposure.
Global/constant placement exchanges these for other memory dependencies; it does
not make lookup free. Register or warp-distributed tables need byte selection or
shuffles, sign handling and exponent logic, with no demonstrated net saving.
The primary [PTX PRMT specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt)
selects four bytes from eight source bytes; it cannot directly select arbitrary
BF16 pairs from the complete magnitude table in one instruction.

Per-block prepared BF16 S/3S arrays would store expanded decoded scale data and
increase weight bytes; they are not the requested NVFP4-preserving optimization.
The original FP4/E4M3 representation and all routed terms must remain intact.
A universal format table is different from storing dequantized model weights,
but its resource/performance case still fails the present screening gate.

## Evidence and stopping criterion

Prior work already rejected SS dequant producers without throughput evidence
(`NEXT_DECODER_STRUCTURE_PIVOT.md`), GS4/NTT1 changes without gain, and a compact
two-CTA architecture despite verified occupancy. KPERM is the distinct recent
scale-issue experiment; its official receipts must supply its verdict, not this
source audit. None of those results proves that all dequant work is free.

Recovered parent NCU samples show approximately 74–75% DRAM throughput,
59.4% issue-active, 1.39–1.40 eligible warps, and low tensor activity. They do not
include source-PC attribution or a math-throttle counter. Do not add stall
percentages or multiply hypothetical instruction savings into a latency forecast.
There is no measured exposed HMUL2 bottleneck and no byte reduction from these
proposals. Even a substantial WGS-only improvement affects only three scored
loads; it cannot be presented as an all-load first-place mechanism.

Reopen only for a concrete exact encoding/selection sequence with independently
verified all-code arithmetic and a plausible **net** reduction in generated
instructions/register lifetime, or actual source-level profiling that identifies
this multiply/selection work as exposed latency. Current evidence supports
preserving the incumbent and moving to another structural lead, with no GPU
experiment or speculative variant allocated here.
