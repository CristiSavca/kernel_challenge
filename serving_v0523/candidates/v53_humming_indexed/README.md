# V53 indexed Humming NVFP4

V53 replaces the target model's expert backend in V41 with the Humming indexed Hopper W4A16 path. Marlin remains available with its ordinary launch flag. The Humming variant keeps one packed expert bank, preserves the draft backend, emits uncombined BF16 expert outputs in original slot order, and uses Muse's existing per-expert RMSNorm, routing score sum, and TP reduction.

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Patch SHA256 `341c37c4d6e8cd7011218597de42870023fe2a33a046c6e2c7249d123107cb48`
- Patch size 22,335 bytes across six Python files
- Humming API source `humming-kernels==0.1.12`, sdist SHA256 `c96c37c95f74379067d2d2d0c5154d12ee9a8cfa29a7c3370a7f3e34f207ca72`, upstream `636ba85648c30ae2c2bfb335c9399593a67ecc1d`. The pinned SGLang package declares this dependency. [Humming release and source provenance](https://pypi.org/project/humming-kernels/0.1.12/)

## Arithmetic and dispatch

The Humming weight schema consumes raw packed E2M1 codes and one E4M3 scale per 16 K elements. Preprocessing reproduces the stock Marlin E4M3 subnormal-to-zero behavior. The adapter explicitly uses BF16 inputs, FP32 MMA accumulation, and a full K reduction per tile. The package's default Stream-K path reduces BF16 partial outputs, so the candidate disables it. After each GEMM, a Triton kernel applies the stock gate-column or down global scale to the BF16 output, with BF16 rounding before SiLU or RMSNorm. The alignment and scale kernel skip padded and nonlocal routes. The stock finalizer visits every local expert in its original top-k order.

Weight preparation calls the package repacker through `HummingMethod.prepare_layer_meta` and `transform_humming_tensors`. Forward uses `HummingMethod.forward_layer` directly twice. The generic Humming MoE forward combines experts before it could apply Muse's per-expert RMSNorm. Gate and down alignments are shared when their tuned M tile matches and selected separately when they differ. Both GEMMs reuse one output allocation, as does V41 Marlin.

## ModelOpt platform gate

The first draft reached the inherited ModelOpt method before the Humming post-load hook. That generic constructor rejects Humming on Hopper. V53 now uses a default-off capability attribute enabled only by `MuseSparkNvFp4FusedMoEMethod`. Admission requires explicit Humming selection on CUDA SM90. The Muse expert constructor separately checks BF16 and uses the existing initialization for the quant config, fallback layout flags, and cache. Other ModelOpt subclasses retain the stock guard; the configured process-wide backend is never changed.

## Local verification and limits

- The patch passed forward application against an isolated index of the exact SGLang base, reverse-application check, original file mode preservation, patch rules, and Python compilation of all six files.
- 192 mocked executions of the actual ModelOpt constructors agreed with stock across backend, device and layout settings except for the Muse Hopper Humming admission. Eight mocked executions of the actual Muse expert constructor covered BF16 Hopper Humming, rejected devices and dtypes, automatic Marlin, explicit Marlin, and Blackwell TRT-LLM. Source review traced the target expert creation to the Muse subclass and separated the ModelOpt dense linear backend selection from the MoE runner.
- A CPU test of the actual Python adapter with mocked tensor and Humming operators passed 40 cases for all four offsets in a synthetic EP layout. It covered empty/all-inactive batches, mixed local and remote routes, separate M tile sizes, and active slot ordering against a direct oracle. It also checked output parameter ownership and gate-column selection. This test does not execute Humming or the Triton kernel.
- An exhaustive CPU model matched stock Marlin's and V44's block-scale decode for all 127 nonnegative finite E4M3 byte values. A separate source-derived CPU model matched Humming and Marlin for all 2,032 signed E2M1 × nonnegative finite E4M3 operands. Humming’s executable repack, exponent handling, device accumulation, and CUDA graph replay still need GPU validation. The exact imported library and offline writable JIT cache need to be verified on the judge image.
- Packing releases the serialized expert weight and scale parameters after preparing each layer. Both banks coexist during that layer's transform. Peak allocator reserve, graph memory, and CUDA compile time are unmeasured. After replacement, that layer requires the packed Humming path for the lifetime of the model load.

## H200 gate

This candidate is a **private-check candidate**. Advance to a scored run only if the exact patch and flags pass the GPU gate. The activation flags are `--flag moe-runner-backend=humming --flag speculative-moe-runner-backend=marlin --flag swa-full-tokens-ratio=0.25`. SGLang otherwise inherits the target expert backend for the draft.

Check the exact judge image's Humming import, successful initialization of all ranks, positive live and all-remote routes across graph replay, first-turn and later-turn agreement, graph memory and overall peak, and end-to-end TTFT. Use the 16-user check first and follow a promising result with a 32-user check. Compare against the best passing dense build, including first-turn TTFT and decode speed. Humming avoids the dense branch's 2.13 ms/layer measured dequantization cost at 16,384 tokens, but its indexed GEMM runs the low-bit unpack for every tile and V53 adds two scale kernels. Humming's throughput in this mode can fall below dense V44 or Marlin, particularly without Stream-K. No speedup is claimed.

V26's 1.6851× private component result covers only M ≤ 256 on the previous baseline. Its V27 integration timed out before a numerical case completed, and the native 384-row bound implies 43 core passes for a full 16K prefill. V53 has no such chunk loop and needs a much smaller adapter, but it has no measured GPU component advantage yet. The previous V26 result remains the stronger decode-specific performance evidence.
