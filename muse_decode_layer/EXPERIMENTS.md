# Whole decoder experiments — November 3, 2026

Both private checks completed on H200 and passed GPU compilation, all 13
numerical cases and all static checks, with no flags or rejections. The official PyTorch starter is preserved in the first
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
bug. Official checks now confirm the private finalizer symbol is available,
the numerical tolerances pass, and V2 has a useful measured gain.

## November 3 official results

| Candidate | Speedup | 95% interval | Normalized score |
| --- | ---: | --- | ---: |
| V1 | 1.017865× | [1.002597, 1.033133] | 22,223.29 µs |
| V2 | **1.081198×** | **[1.064980, 1.097415]** | **20,921.52 µs** |

Normalized SGLang baseline: 22,620.30 µs. V2's aggregate time is 5.86% lower
than V1's across these separate runs. Recommend scoring the exact V2 source,
SHA-256 `010c881e46e411735adef5aff33713d060106c9a4abef3cc5a4986043c221197`,
from `experiments/candidates/muse_decode_layer/`.

V2 improves one-token decode by 1.185× sliding / 1.181× full attention,
16-token decode by 1.168× / 1.097×, and single-request eight-token verify by
1.161× / 1.147×. Verify at batch 32 improves 1.171× / 1.105× from the local
expert tile changes. Remaining regressions: decode batch 32 is 0.899× /
0.956× and verify batch 16 is 0.959× / 0.974×. Larger-batch routing/GEMM
costs still deserve attention; small-batch finalizer fusion is now validated.

No scoring or additional private uploads were performed by this analysis.
# November 3 scored promotion

The exact V2 snapshot `010c881e46e411735adef5aff33713d060106c9a4abef3cc5a4986043c221197`
passed private check `a28df47ba0584bed` at 1.0812× and was submitted for scoring
as `846bf3fd98c54703`. The scored job is pending.
