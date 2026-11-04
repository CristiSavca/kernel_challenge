# Expert routing experiment — 2026-10-24

Private check `1cc96c4a472b470a` tests `candidates/local_routing_v1.py`.
The immutable upload and full status are recorded under `../experiments/`.
Private H200 validation passed correctness and static checks. Measured speedup:
**1.0526377681×**, 95% interval **[1.0368482016, 1.0684273346]** relative to SGLang.
Scored submission on November 3: `8f1f69d433364789` passed at
**1.0457873952×**, 95% interval **[1.0301005843, 1.0614742061]**. It uses
the same frozen source as the passing check.

This candidate retains SGLang's NVFP4 Marlin path, FP32 accumulation, BF16
intermediates, SwiGLU and per-expert normalized weighted sum. It changes only
the sorting bucket count and the shape-based tile heuristic:

1. Map global expert IDs to local IDs exactly as stock does, then align using
   the local count 32. The alignment wrapper adds its own bucket for -1.
   This avoids 224 impossible expert IDs and enables the existing native
   small-bucket alignment path.
2. Estimate routes per local expert using the global expert count 256.
   Non-local routes are not executed by this rank. At 128 tokens, block M
   changes from 48 to 8; at 256/257 tokens it changes from 64 to 16. A hot
   expert still receives as many complete tiles as its actual routes need.

The pinned wrapper/CUDA implementation was inspected to verify the sentinel
contract, and local syntax compilation passed. The official private H200 check
also passed numerical correctness and established the measured gain above.

If the combined change regresses, test a count-only control before changing
the math. Do not pre-dequantize the persistent NVFP4 weight bank.

The equivalent serving change is staged in the separate pinned checkout and
exported to `../serving/patch.diff`. It is an unvalidated draft and cannot be
checked by the official serving service until that track opens November 5.

## November 3: fuse the small-batch finalizer

Private check **`8443a6493f7c4234`** tests `candidates/fused_finalize_v2.py`.
Local SHA-256: `1c630ca67809da86e67d258baa8de0cbcff5639dbc8e563482354247c3ed5ea2`.
Official H200 result: **passed**, 5/5 numerical cases, all static checks,
no flags or rejections. Measured speedup **1.205784×**, 95% interval
**[1.187698, 1.223871]**, score **11,153.32 µs** versus normalized SGLang
**13,448.50 µs**. This exact V2 snapshot is eligible for scoring; the active
V3 file is still unvalidated. No scored submission was made by this analysis.

The v1 profiles identify a specific gap: stock's post-expert norm helper uses
an eager PyTorch fallback below 32 tokens. At one token, Marlin takes about
46 µs of the total 83 µs; the mask, FP32 casts, per-expert norm, weighting and
reduction account for most of the rest. At 8 and 16 tokens the same fallback
also launches a dozen kernels. V2 fuses those operations into one Triton
kernel for every batch size. It reads only local routes, applies the norm
to each expert before weighting, and retains FP32 intermediate arithmetic
and a final BF16 output. GEMMs, weight layout and the v1 tile heuristic stay
the same, isolating the fusion's effect.

## November 3: direct packed NVFP4 GEMV for tiny batches

Private check **`cdec36c2da4b410c`** tests `candidates/packed_gemv_v3.py`.
Local SHA-256: `f01c5293dcc3a4bd8f0e448e0e0ad52550c3b98719e6dd35ef5641a9622693e5`.
The result is pending. `submission.py` currently matches this frozen source.

V3 includes V2 and dispatches batches of at most 8 tokens to two custom
Triton kernels. The first reads each live route's packed gate/up weights,
decodes E2M1 and applies the E4M3/FP32 scales, performs two FP32 reductions
with BF16 weights and inputs, rounds the GEMM outputs to BF16, and fuses
SwiGLU into the BF16 intermediate. The second performs the down GEMV and
writes BF16 expert outputs. The V2 finalizer then normalizes each expert
before the weighted sum. Batches above 8 tokens retain Marlin.

`prepare` retains lossless local packed copies, approximately 864 MiB per
prepared state in addition to Marlin's packed bank. It never stores
dequantized BF16/FP32 experts. Routes and activations are read anew on every
timed call; non-local IDs remain -1 and are skipped on the GPU. Allocations
and dispatch depend only on shapes, so the path is CUDA-graph compatible.

Local validation: Python AST parsing and an exhaustive CPU check of all 16
E2M1 nibble encodings, including signed zero, passed. Static review covered
packed row/scale strides, gate/up split, local ID masking, BF16 boundaries,
and full writes of every local output. CUDA compilation, numerical
correctness and performance remain pending official H200 execution.

Performance risks: direct GEMV cannot exploit tensor-core weight reuse
across tokens as grouped Marlin can, so the 8-token path may regress; the
extra packed weight bank increases GPU memory use; register pressure and
the unscheduled reduction loops may limit bandwidth. If V3 regresses but
V2 passes, retain V2. Revisit a one-token-only dispatch after inspecting
per-shape timing, without publishing an unmeasured variant.

V2 is complete and passing; V3 remains queued at the latest checkpoint.
The initial receipts were `received` at 2026-11-03 23:00 UTC. The helper has frozen the
uploaded bytes and saved receipts under `../experiments/`. An independent
coordinator review found no source-level issue with V3's decoding, scaling,
BF16 boundaries or route masking.
Do not retry queued work. No further checks, scored submissions or commits
were made by this subtask.


## November 3 result analysis: V2

| Tokens | SGLang paired time | V2 paired time | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 76.30 µs | 54.44 µs | 1.402× |
| 8 | 143.62 µs | 113.47 µs | 1.266× |
| 16 | 196.45 µs | 163.51 µs | 1.201× |
| 32 | 206.18 µs | 211.99 µs | 0.973× |
| 128 | 471.68 µs | 430.62 µs | 1.095× |
| 256 | 638.64 µs | 471.85 µs | 1.353× |

The fusion provides the expected small-batch improvement. The prior V1 check
measured 81.95/142.21/189.90 µs at 1/8/16 tokens; V2 is materially faster in
those separate runs. At 32 tokens V1 was also approximately 0.971× SGLang,
so the remaining small regression predates the finalizer. The normalized
aggregate score time is 12.7% lower than V1's private check and 13.3% lower
than the V1 scored run; these comparisons are across separate runs and seeds.
Retain V2 as the validated incumbent while V3 is queued. Do not submit active
`submission.py` until V3's own result establishes correctness and improvement.
# November 3 public baseline and FP8 masked-load repair

Validated fused-finalize V2 (1.2058× private) was submitted unchanged for scoring
as `777df2bfda8b4a57`; score pending.

Packed GEMV V3 check `cdec36c2da4b410c` failed compilation on the small-token
path: integer `other=0` cannot cast directly to the FP8 scale type. Larger-token
Marlin tests passed. Three scale loads now use `other=0.0` in
`candidates/packed_gemv_v5_fp8_fill.py`, also active in `submission.py`.
Fresh private check: `8af27f0313b44d84`, pending. No V3/V5 score was submitted.

`candidates/public_local_align_v4.py` is an independent candidate based on the
compact published GPT-6.1 Sol MoE kernel from Auto-research-agent commit
`018b7a102348ea3b73701e8be52d24f82d8c91ee`. It adds provenance comments only,
uses local Triton route alignment plus Marlin, and preserves its MIT license and
original source under `third_party/`. Our private check `7d7d9292e9054337` is
pending. Published local measurements do not establish our score.

## Goal continuation: compact public candidate validated

Private check `7d7d9292e9054337` passed all five numerical tests and six timed
shapes at 1.3348268492×, CI [1.3148044465, 1.3548492519]. The exact frozen V4
source was submitted as `d674d6a1ada44d16` (pending) and promoted to active
`submission.py`. Fused-finalize V2 scored job `777df2bfda8b4a57` is now done
and passed at 1.2100468954×. Packed-GEMV V5 remains a separate pending check.

The reviewed native CUDA V6 source, with full licenses and diagnostics removed,
is in private check `143f8587286e4384`. Its production CUDA/CPP and dispatch
constants match the imported source. Do not score it until our check passes.
