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

## Goal continuation: public expert alignment integration

V2 scored run `846bf3fd98c54703` passed at 1.0813030544×, CI
[1.0650835086, 1.0975226002]. V2 remains the active submission.

`candidates/public_alignment_v3.py` replaces only `_experts` with alignment and
finalization from the standalone MoE V4 that passed at 1.3348×. Existing expert
ID rotation, stream synchronization and all other decoder functions remain
AST-identical. Private check `41920b5cf4bf423f` is pending.

## Public native decoder reviewed and queued

Native V4 check `f8e991f1a2c2469e` tests
`candidates/public_cuda_v4.py`, SHA-256
`80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`.
Native source strings remain identical to the imported decoder. The adaptation
owns the routing bias, restores it after synthetic tuning even on failure, and
uses complete fallback attention for full batches above the native 64-segment
bound. Full original licenses are embedded. Routing, weight packing and serial
stream ordering were reviewed; numerical UV reassociation, synchronization,
compilation and timing still require the official GPU check.

## Own window attention integrated in V5

`candidates/window_attention_v5.py`, SHA-256
`d360fe102cc3b662cbeaf5474f236bee790f2b64e6c175724c56ae2e9c5abf63`,
is in private check `8e13cd6e80194b4f`. It changes only attention inside passing
decoder V2. Both grouped window kernels are copied verbatim from passing
standalone attention V3; the QKV producer was already identical.

The custom path handles sliding decode and verify with batch <= 2. Larger
verify batches and full attention retain V2's complete FA3 path. Profiles
suggest useful headroom in the selected regimes, but the old norm is absent
from the truncated reports and cross-run component times are not integrated
measurements. No end-to-end gain is claimed yet.

The custom combine includes BF16 attention rounding, norm and sigmoid gate;
its output feeds O projection directly. All code after that point is byte-
identical to V2. Scratch/output belongs to each call and all attention work
uses the current stream. Syntax, copied-function AST checks, 56 dispatch/page
branches and independent CPU window/page/row coverage checks passed.
Whole-layer numerical accuracy and routing still require the official check.

`candidates/public_alignment_v6_licensed.py` embeds the complete upstream MIT
notice in the compact alignment integration. Its executable AST is identical
to queued V3, including docstrings. It is not uploaded yet; wait for V3's
result before choosing the next decoder candidate.

## Compact alignment integration passed

Check `41920b5cf4bf423f` passed all 13 numerical tests and 12 timed shapes at
**1.1581602895×**, CI [1.1407878851, 1.1755326938], with no flags. The previous
V2 private result was 1.0811975012×. Larger verify gains are substantial:
batch-16 sliding improved from 552.55 us in the V2 run to 406.29 us in V3;
paired baseline ratios increased from 0.9592× to 1.2503×. These are separate
runs, so use the official within-run ratios rather than subtracting times
as an isolated measurement of the change.

The complete-notice V6 source has an identical AST and is now in private check
`83d9b48acd01462e`, raw hash
`5eb7205b63185fb60ba2a33b53b697d6cc176cff68eb541f1f5c3ef8696e5112`.
Promote and score the exact fully attributed source after its check passes.
The native decoder and own window-attention integration checks remain live.

## Combined window attention and expert alignment, V7

After V3's expert integration passed, `candidates/window_and_alignment_v7.py`
combined that integration's fully attributed V6 source with the exact
attention-only prefix and window kernels from V5. Private check:
`b255f6b17f1b4758`; raw SHA-256:
`d0eaeccd0f776cdca3cd953d8d7cdd84301816c28850d21fdad7c18eddc3a3f4`.

All non-attention functions/classes, including `_experts`, remain identical
to V6. Code from O projection onward is byte-identical to V6. The attention
prefix is byte-identical to V5 and copied kernels match its AST. Syntax and
independent merge invariants passed; the complete MIT notice is retained.
This combined decoder is not GPU-validated yet. Keep individual control jobs
to distinguish integration failures from performance interactions.

## Native V4 passed and submitted

Private check `f8e991f1a2c2469e` passed all 13 numerical tests and 12 timed
shapes at **2.1541142783×**, CI [2.1218025642, 2.1864259925], with no flags,
rejections or runtime errors. Every timed shape improved; ratios range from
1.671× for sliding decode at batch 32 to 2.449× for sliding single-request
verify. This validates the full integration, including the previously noted
UV rounding and routing concerns, for the official tested cases.

Exact frozen source
`80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`
is now active and was submitted for scoring as `4b216098993e4112` (pending).
Complete source licenses and the bias-restoration/batch-bound fixes are retained.
Prepared native scratch still assumes sequential calls; this kernel result
does not establish safety under overlapping serving graphs.

The compact alignment, own-attention and combined controls remain in flight.
Do not score them merely because they pass; compare them with this much faster
native incumbent first.

## Own attention-only V5 control resolved

Check `8e13cd6e80194b4f` passed at **1.1060129400×**, CI
[1.0894227459, 1.1226031341], with no flags or errors. The isolated attention
integration is numerically valid and improves modestly over V2's 1.0812×
private result. It is substantially slower than native V4's 2.1541×, so it
will not be scored or promoted. Preserve it as measured integration evidence.

## Licensed alignment V6 control resolved

Check `83d9b48acd01462e` passed at **1.1612102333×**, CI
[1.1437920798, 1.1786283868], with no flags or errors. Its executable AST is
identical to the earlier V3 integration, whose check measured 1.1582×; the
complete-notice packaging did not change the computation. Retain V6 as a
fully attributed fallback. It will not be scored because native V4 is much
faster and already submitted. Only the combined V7 control remains pending.

## Combined V7 control resolved

Check `b255f6b17f1b4758` passed at **1.1749782771×**, CI
[1.1573536029, 1.1926029513], with no flags or errors. This is numerically valid,
but the small difference from V6 lies within the reported uncertainty, and it
remains substantially slower than native V4 at 2.1541×. Do not score or promote
this control. All private experiments in this batch are now resolved; the
four native scored reruns remain outstanding.
