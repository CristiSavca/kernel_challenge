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
The result is pending. No scored submission has been made for this candidate.

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

Both checks are queued at the coordinator's latest checkpoint. The initial
receipts were `received` at 2026-11-03 23:00 UTC. The helper has frozen the
uploaded bytes and saved receipts under `../experiments/`. An independent
coordinator review found no source-level issue with V3's decoding, scaling,
BF16 boundaries or route masking.
Do not retry queued work. No further checks, scored submissions or commits
were made by this subtask.
