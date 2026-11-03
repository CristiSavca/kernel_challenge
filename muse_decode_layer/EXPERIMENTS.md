# Whole decoder experiments — November 3, 2026

Both candidates are private checks, with GPU compilation, correctness and
performance pending. The official PyTorch starter is preserved in the first
Git commit. These candidates start from the official SGLang baseline.

| Candidate | Check | Change |
| --- | --- | --- |
| `candidates/fused_components_v1.py` | `9e55826b550f414b` | Integrates the previously passing fused QKV producer and local-expert alignment/tile changes |
| `candidates/fused_components_finalize_v2.py` | `a28df47ba0584bed` | Adds the stock Triton expert finalizer below 32 tokens, avoiding its eager wrapper fallback |

The active `submission.py` is V2. Exact upload snapshots, local SHA256 values,
and status receipts are under `../experiments/`.

## Math and source audit

- The QKV producer is AST-identical to the standalone attention producer that
  passed private check `702a17ec1ad94e03` and scored run `8727d7d1d2da427c`.
  It preserves normalization and RoPE BF16 boundaries before FP8 conversion,
  supplied cache scales, page-size-agnostic flat cache rows and live slot zero.
- Dense projections, FA3 attention, output norm/gating, routing, residual
  boundaries and stream synchronization remain the production baseline path.
- Expert routing still rotates by the supplied `local_start`, maps the local
  bank to IDs 0–31 and masks other routes with -1. Alignment gets 32 experts;
  its wrapper adds the invalid sentinel bucket. Tiles use expected local
  occupancy and the same block size in alignment and both GEMMs.
- V2 invokes the production `_post_expert_norm_weighted_sum_kernel` directly
  below 32 tokens with the same arguments as its large-batch wrapper. It skips
  invalid slots before reading their uninitialized Marlin rows, applies FP32
  weight-first norm and weighted sum, and stores BF16 output.
- Main/side-stream waits and tensor lifetime handling are unchanged. All new
  allocations and branches depend on fixed dimensions/configuration.

Local Python compilation and an independent source audit found no actionable
bug. The private finalizer symbol's availability in the judging image, numerical
tolerances and performance still require the H200 checks. Do not score either
candidate until its own check passes with a useful measured gain.
