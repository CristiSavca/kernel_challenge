# Attention new-K/V FP8 packing: bounded feasibility

2026-11-04. Read-only proposal; no candidate, GPU call or performance claim.
One small instruction-removal lead is worth a source review. No second
implementation-ready mechanism emerged from the bounded attention review.

Exact standalone parent:
`87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c`
(`muse_attention_decode/submission.py`); public receipt `8f9e5971f95841a0`,
private `c21d391dbf604bb5`. Embedded decoder parent:
`97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.

## Exact proposed delta

At standalone lines2734–2739, `segment`'s `has_new && live` cache-write
prologue currently packs eight FP32 values through eight scalar calls:

```cpp
uint32_t w0 = 0, w1 = 0;
#pragma unroll
for (int e = 0; e < 4; ++e) {
  w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);
  w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);
}
```

The fixed replacement would use the existing helper at line1293:

```cpp
const uint32_t w0 = e4m3x4f(x[0]*sc, x[1]*sc, x[2]*sc, x[3]*sc);
const uint32_t w1 = e4m3x4f(x[4]*sc, x[5]*sc, x[6]*sc, x[7]*sc);
```

`to_e4m3` calls `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)`.
The active `OKC_PACK4=0` helper emits two
`cvt.rn.satfinite.e4m3x2.f32` instructions per four values, passing `(x1,x0)`
then `(x3,x2)` so x0 occupies the lowest byte, and combines the two halves.
Both use round-to-nearest, ties-to-even and finite saturation. Keep each
identical FP32 multiply by `sc`; there is no FP16 intermediate, changed norm,
RoPE, reciprocal, probability or output arithmetic. The source comment about
`OKC_PACK4=1` producing the same SASS concerns two forms of this already-packed
helper, not a measured comparison with the scalar cache-write loop.

Primary semantics: [CUDA 13.0.2 FP8 conversion API](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)
and [PTX cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt).
Exact scalar-header lowering and NaN sign/payload handling still require the
critic's pinned-header review before calling the replacement bit-equivalent
over all FP32 bit patterns. Finite RN/saturation semantics match. The scalar
intrinsic may itself use a packed instruction with a dummy operand, and ptxas
may already combine scalar conversions. Actual emitted instruction savings
are therefore **unknown**, not established by the source call count.

## Scope and bounded operation counts

`has_new = end > seq` (line2369). Only a segment intersecting the new-token
suffix reaches this work. Each such segment packs all `L*4` K/V rows of64
dimensions, even if its global-store mask owns fewer than L new positions.
That is256L scalar conversions, or128L paired conversions after the proposed
change. Its32L active threads represent L warp-instruction instances. If the
old scalar calls each survive separately, the conversion instruction reduction
is4L per segment, distributed across those warps. Shifts/ORs may also combine;
no exact SASS count is claimed. Speculative length validation precedes this
block, so a failed length speculation does not pack twice.

For the scored TP8 shapes (one KV pair), let H count nonempty segments with
`has_new`. These are source-derived counts/bounds, not runtime instrumentation:

| Workload | H | Old scalar calls | New paired conversions | Conditional warp conversion reduction |
| --- | ---: | ---: | ---: | ---: |
| B1, L1, either layer | 1 | 256 | 128 | 4 |
| B16, L1, either layer | 16 | 4,096 | 2,048 | 64 |
| B32, L1, either layer | 32 | 8,192 | 4,096 | 128 |
| B1, L8, sliding | 2 | 4,096 | 2,048 | 64 |
| B1, L8, full | 1 | 2,048 | 1,024 | 32 |
| B16, L8, sliding | 16 | 32,768 | 16,384 | 512 |
| B32, L8, sliding | 32 | 65,536 | 32,768 | 1,024 |
| B16, L8, full | 16–32 | 32,768–65,536 | 16,384–32,768 | 512–1,024 |
| B32, L8, full | 32–64 | 65,536–131,072 | 32,768–65,536 | 1,024–2,048 |

L1 has one new position per request, owned by exactly one segment. B1/L8
sliding uses64-key chunks; new positions relative to the window are2047–2054,
so two segments execute the prologue. B1/L8 full uses320-key chunks and fixed
seq80000, giving one. B16/L8 sliding uses320-key chunks; B32's tuned four
splits use576-key chunks, each with one suffix segment per request. Batched
full L8 has264 flat partitions: scored lengths are74000–86000 and the minimum
virtual partition is thousands of keys, even allowing the600-key boundary
padding. An8-key suffix can cross at most one partition boundary. Actual
batched lengths were not reconstructed, hence the H intervals. For general
inputs a looser bound is B <= H <= B*L per KV pair; none of these counts would
enter candidate dispatch or qualification logic.

All12 standalone cases potentially execute the changed prologue. This is
**not repeated cached-K/V tile decoding**: no long-loop QK/PV load/conversion,
softmax or MMA is removed, and cache/shared/global bytes are unchanged. The
four-instruction conditional Q1 saving is particularly small. No whole-call
or whole-decoder percentage follows from these operation counts.

The decoder contains the same loop at lines2800–2805, but builds attention
with `OKC_QDIRECT=0`, `OKC_PK2=0`, `OKC_W0ENABLE=1` and existing PDL behavior.
A future port must preserve those settings and be tested separately; this
standalone proposal changes neither scheduling nor any launch/barrier/store.

## Review gate and rejected second lead

If authorized, the narrow next step is one isolated standalone replacement,
unique extension name, exact source-restoration check and targeted conversion
ordering/boundary review, then the complete official suite against both exact
parent receipts. No new diagnostic framework, geometry changes or conversion
policy sweep is justified. Root/critic must predeclare a suitably modest
weighted12-case screen; no universal1% requirement or gain is assumed here.
If compiler combining makes this a no-op or full-call timing is unhelpful,
close the fixed attempt. Device correctness remains unproved until the check.

The other inspected lead was Q1 `load_v_frags` (line1026): replace four LDS64
gathers with two `ldmatrix.x4.trans.b16` instructions. It is not ready to test.
The current FP8-byte B-fragment layout deliberately permutes output dimensions
(`16*tig + 8*c + t` in the merge); b16-transpose loads treat adjacent FP8 bytes
as one16-bit carrier and do not directly produce the current lane layout.
Additional cross-lane/byte permutations or a broader MMA/epilogue layout
rewrite would be needed. There is no demonstrated net instruction reduction.
This is a mapping obstacle, not a proof that no useful design exists. Saved
V5 changed QK precision and V11 changed Q8 ownership; neither supplies evidence
for this particular Q1 gather rewrite. Do not implement it in this experiment.
