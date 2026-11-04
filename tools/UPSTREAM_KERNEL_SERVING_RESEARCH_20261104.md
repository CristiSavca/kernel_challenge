# Kernel and serving research prompted by the SGLang release article

Reviewed November 4, 2026. User supplied
[AICoder's v0.5.22 report](https://aicoder.com/news/news-1792384474854-fl9ix).
The linked [official release](https://github.com/sgl-project/sglang/releases/tag/v0.5.22)
supports its headline H200 result for **DeepSeek-V4.1**, not Muse Spark.
[v0.5.23](https://github.com/sgl-project/sglang/releases/tag/v0.5.23), released
November 2, is newer and includes additional graph, speculative and MoE work.
No upgrade was installed: the challenge starts from pinned SGLang a55ab7d5,
so useful changes need an individually reviewed source port and measurement.

## Decisions from primary sources and the actual checkout

| Source | Mechanism worth examining | Decision for this project |
| --- | --- | --- |
| [SGLang #41251](https://github.com/sgl-project/sglang/pull/41251), merged Oct 3 | Hopper GEMM selection, exact FP8-to-BF16 derived weights, fused activation with preserved rounding, reduced indexing/padding work | The relevant added statements in the activation helper, Marlin launcher, generic fused-Marlin path and FP8 utility files are already present in our pinned checkout. DeepSeek indexer/mHC changes are not Muse operators. No duplicate port or 2.3x prediction. |
| [SGLang #40762](https://github.com/sgl-project/sglang/pull/40762), merged Oct 20 | Enable FlashAttention in EAGLE's draft-extend graph allowlist | Independent exact-diff and caller review found seven added allowlist lines, absent locally but unused by the configured DFlashWorkerV2. DFlash already initializes graphs through its own path. No port: see `SERVING_UPSTREAM_GRAPH_REVIEW_20261104.md`. |
| [SGLang #43363](https://github.com/sgl-project/sglang/pull/43363), merged Oct 20 | Give FlashInfer CUTLASS MoE startup autotuning a stable prefill-sized tuning range | Absent from our pinned helper, but the selected H200 Muse route is Marlin. This change would not accelerate that path. Reconsider only if a justified backend change actually uses CUTLASS; preserve memory/startup costs. |
| [SGLang #44972](https://github.com/sgl-project/sglang/pull/44972), merged Oct 18 | Combine draft write-window gathering with FULL/SWA physical KV location translation | New relative to this checkout. Actual production caller in the PR is DSPARK and part of a multi-PR cache stack. Retain the fusion idea, but no direct DFlash port without its caller/layout/lifetime proof. Its reported stack result is not an isolated measurement. |
| [Muse Helion verify #44668](https://github.com/sgl-project/sglang/pull/44668) | Model-specific fusion across attention, projections, MoE and communication | Useful architecture reference; the PR remains draft and its latest rebase does not supply fresh GPU validation. A wholesale import is not a measured H200 serving entry. |
| Actual pinned Muse finalizer | For small batches, the Marlin output path uses separate masking, norm, score and sum operations | Implementing an isolated two-kernel replacement. This is directly selected in the real H200 call path, preserves existing BF16 boundaries, and can be checked against stock Marlin using the official kernel evaluator. |

The #41251 presence check compared nonblank added statements to local files;
it does not prove merge ancestry or equivalence of every branch in that PR.
Raw GitHub API metadata and patches for #41251/#43363/#44972 are retained in
`experiments/research/20261104-upstream-serving/`. No downloaded source has
been executed, and no model weights or extra GPU resources were requested.

## Papers with useful mechanisms and explicit limits

- [FuTransformer, Oct 16](https://arxiv.org/html/2610.23796v1): fuse producer
  epilogues with KV placement and decode quantized values directly into MMA
  fragments. Its INT4 KV and changed projection precision are outside this
  challenge's required formats; folding normalization into weights can change
  rounding boundaries. Our native MoE already uses direct fragment decode.
  Keep the boundary-fusion mechanism where BF16 intermediates are preserved.
- [Dissecting GPU Utilization on Hopper, Sep 11](https://arxiv.org/abs/2609.12923):
  separates tile utilization, scheduling, occupancy and memory pressure rather
  than treating one busy-SM percentage as efficiency. Useful for choosing a
  concrete diagnostic; it supplies no speedup prediction for our source.
- [AutoMegaKernel, Jun 8](https://arxiv.org/html/2606.09682v1): schedule checks
  for persistent kernels and an automated keep/revert loop. Its checks are not
  a mechanized proof. Its disclosed A100/H100 losses and precision-asymmetric
  wins prevent treating the project as a ready H200 replacement. Reuse schedule
  invariants when useful, not an additional optimization platform before testing.
- [DFlash, revised May 28](https://arxiv.org/abs/2602.06036): explains the
  parallel draft/target-verify structure already required by the contest.
  Target redundant metadata, launches and movement while keeping the released
  drafter and its weights. [DSpine](https://arxiv.org/abs/2609.36173) and
  [DFlare](https://arxiv.org/abs/2606.02091) change learned draft architecture,
  so their reported gains are not an allowed direct replacement here.

InstantTensor is a model-loader optimization and touches a forbidden serving
patch area. Prefill/decode disaggregation requires an applicable serving topology;
it is not a switch that makes this fixed one-node benchmark inherit the reported
multi-worker gain. These are reasons to filter the article's mechanisms before
spending experiments, not reasons to dismiss the underlying engineering.

## Executed consequence

The preceding instruction-footprint experiment v21 passed correctness but
failed its fixed speed gate: MAIN normalized latency increased12.39%/13.42%
versus public/private incumbents. It is closed, with no public replacement.
The next concrete work is the serving small-batch finalizer, with a fixed5%
same-run stock-Marlin improvement gate and explicit unaffected controls.
Its standalone check cannot establish full serving speed or a leaderboard win.
The full objective remains active and unachieved.
