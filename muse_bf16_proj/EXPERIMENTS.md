# Dense projection experiments — 2026-10-24

## Active candidate

`submission.py` is byte-for-byte `candidates/v001_splitk.py`. The snapshot was copied 19 seconds after the check; the source was not intentionally edited after submitting. However, the server hash differs from the local source hash, so identity with the queued server payload is **not cryptographically verified**.

Remote off-board check: `43cb543b57c64b29`.
Server `content_hash` reported by OKC: `f1667fba62e225b06f1fc75e9b0c304a2c34fafb00b935167055fd61465bb841`.
Local source SHA-256: `eed1a2f6b8122617194d0f7163accd8a1e1fabc09791651804dbd824c3e4e4d8`.

A loopback-only mock API probe confirmed the installed OKC CLI sends all 3,978 bytes unchanged in its JSON `code` field; no remote job or quota was used by that probe. Common newline, whitespace, header, comment, docstring, and JSON-serialization transformations do not explain the server hash. Root should treat queued-payload reproducibility as unresolved until the service supplies code evidence or an explicitly authorized new check records a matching digest.
Latest observed status: queued. No numerical correctness or H200 performance claim is established yet. Local Python parsing passed; this Mac has no CUDA runtime.

The implementation keeps original BF16 and FP32 weights and packs BF16 matrices by transposing them without changing values. For one token, it computes a bandwidth-oriented FP32-accumulating GEMV over original weights, split into 1,024-channel chunks, then reduces the partials. For 2–32 tokens it uses BF16 tensor-core GEMMs with eight K partitions. The FP32 router splits its original weight into BF16 high and low residual components and computes both products with FP32 accumulation. Other token counts keep ordinary PyTorch BF16 GEMMs. The sliced small-batch unpooler also retains that fallback in this first candidate.

## Audit

- All branches depend only on fixed shapes/config fields. Input activation values are read on every call; no activation state or outputs are cached.
- `prepare` only retains/repackages weights. No input computation happens there.
- BF16 projection weights remain BF16. The router uses its original FP32 weight for the single-token case and the explicitly permitted BF16 high/low split elsewhere.
- GEMV and split GEMM accumulation and partial reductions use FP32. BF16 projection outputs are rounded only on their final store; router outputs stay FP32.
- Masked stores preserve output shape. Masked GEMM loads cover non-multiple token counts. The first 1,024 unpooler rows use original row stride correctly.
- Each split handles disjoint K indices; all are reduced exactly once. Dimensions 1,024, 4,096, and 8,192 are divisible by the candidate K tiles, with masking retained for bounds.
- All allocations have fixed shape. No synchronization, network/process calls, harness imports, timing changes, or module mutation appear in the submission.

Remaining risks require a real runner: Triton JIT/compiler compatibility in the challenge image, GPU register/occupancy and memory-bandwidth behavior, floating-point comparison against the strict router reference, and speed under CUDA-graph cold-L2 timing. The larger router batches currently use only one K partition, which may underutilize the GPU.

## Next candidate (not submitted)

`candidates/v002_splitk.py` packs the 1,024-row unpooler slice separately and retains eight K partitions for the router at all batch sizes. It is a proposed occupancy/fallback improvement, not a measured improvement. Do not replace the active candidate until the first check's correctness and per-shape profile are available.

## Resume

Run `okc status 43cb543b57c64b29 --json` from this directory. OKC writes the result profile beside the source when it completes. If it passes, inspect the 30 scored shape timings, especially the single-token pooler and 128/256-token router. Compare any follow-up candidate with off-board checks before spending a scored submission. Do not infer wins from gains inside the reported confidence interval.
