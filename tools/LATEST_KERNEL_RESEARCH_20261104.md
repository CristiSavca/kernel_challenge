# Research decisions — November 4, 2026

This pass covered recent arXiv papers, DeepSeek's official repositories and
deployment documentation, FlashInfer and CUTLASS releases, Hugging Face kernel
sources, and X announcement leads. Direct X fetches returned 403, so technical
claims were traced to available author releases and code. No unavailable post
was treated as verified. Exact commits, licenses and scoped release-absence
findings are in the four detailed notes below.

| Lead | Decision |
|---|---|
| [PipeLink, October 27](https://arxiv.org/abs/2610.37200) | Released Hopper source skips accumulator rescaling when the scale is exactly one. The same unconditional work exists in our Q1 attention source. Independent semantic/control-flow review is required; no whole-framework adoption or inherited speedup. |
| [FlashInfer 0.7.3, November 4](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.3) | Real recent source for Hopper BF16 MoE and ordered communication/load scheduling. Its weights, normalization and rounding boundaries differ from Muse. Retain mechanisms for a measured serving bottleneck. |
| [DeepEP V2.5](https://github.com/deepseek-ai/DeepEP/tree/f6aa38fe06cd4a0489e7730365d4e0114970a2f3) | Deferred epilogues are relevant; its BF16 combine input and new runtime/API prevent a direct replacement for our FP32 per-route normalization and sum. |
| [DeepInfer deployment Q&A, November 3](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md) | Useful scheduling/communication guidance. This is a documentation refresh; the released engine targets a different model and SM100/Ascend. |
| [ZipFusion, October 24](https://arxiv.org/abs/2610.33719) | Direct-register lossless BF16 decompression is relevant in principle, but the required attention cache is already FP8 and our dense path already has a related 12-bit codec. No new byte-saving candidate follows. |
| [PTX in Lean, November 3](https://arxiv.org/abs/2611.04447) | Valuable distinction between modeled ordering, termination and real floating-point behavior. It is not a ready-made correctness/performance filter for our packed arithmetic. |
| [Hugging Face kernel sources](https://github.com/huggingface/kernels-community/tree/f923c618f29b019080f4b02ca2be5bc35fd10c51) | Several apparently compatible BF16 APIs quantize activations internally. Inspect actual compute precision and build architecture before considering reuse. |

Rohan's main, all 13 branches and latest public release remained unchanged from
our earlier audit. No newly released GA/predictor/Lean optimizer was located in
that repository. This is a public availability finding, not a judgment about
private experiments.

Detailed reports:

- [arXiv and released implementations](LATEST_ARXIV_KERNEL_SCOUT_20261104.md)
- [DeepSeek source and deployment material](LATEST_DEEPSEEK_KERNEL_SCOUT_20261104.md)
- [X leads, FlashInfer and Rohan](LATEST_X_KERNEL_SCOUT_20261104.md)
- [Hugging Face and CUTLASS](LATEST_HF_CUTLASS_KERNEL_SCOUT_20261104.md)

## Actual project measurement during the research

Official private job `cffa322caeff4aa2` checked exact source
`fc2fc6b51822e0e91eb6905790476ced13aa4c8599954dac2bb20edbf45fd39f`.
Only the small-token finalizer differs from the official stock-Marlin baseline.
All 5 correctness tests, 6 benchmarks and 6 profiles passed with no flags.
Root and independent critic reproduced every predeclared gate.

| Tokens | Stock component, µs | Candidate component, µs | Latency reduction |
|---:|---:|---:|---:|
|1|77.168|51.977|32.65%|
|8|142.480|112.058|21.35%|
|16|195.808|160.586|17.99%|
|32|211.936|211.975|-0.018%|
|128|456.800|456.843|-0.009%|
|256|617.616|617.906|-0.047%|

These are the official same-run normalized component timings. The affected
geometric speedup is 1.32034496; all-six aggregate is 1.14891952. Profiles confirm
the intended new kernels on affected cases and the original control path.
This is not an improvement over our faster native expert entry or a measured
whole-service result. No scored candidate was replaced. No Modal allocation or
out-of-pocket compute occurred in this iteration; one free official check was
used. Raw receipt, profiles, source and independent review are retained.

Decision: keep the finalizer and proceed to generic GPU, stream and graph
validation before full serving tests. The next research-derived optimization
is conditional on resolving exact-one rescale control-flow/semantic concerns;
it is not a substitute for the now-measured serving work. Any claimed gain must
still survive the official evaluator and preserve the active incumbents.
The critic identified a concrete issue with the naive warp-vote version:
the enclosing Q1 branch is only known to be quad-uniform, so an unconditional
full-mask vote inside it cannot be assumed safe. No such code was introduced.
The [independent rescale review](ATTENTION_EXACT_RESCALE_RESEARCH_CRITIC_20261104.md)
instead recommends a simple scalar conditional, subject to numerical/compiler
validation. It identifies five affected cases and seven controls. A hypothetical
10% reduction in those attention kernels models only about0.79% whole-decoder
latency reduction; that is a scenario, not a prediction. This supports retaining
the finalizer as the higher-priority measured work.

At 17:56 UTC, CristiSavca remained 401st overall/decoder; decoder 2.1812× versus
leader 2.5762×. MoE was 303rd at 2.3737×. The #1 objective remains unachieved.
The source and evidence have advanced; no leaderboard improvement is claimed.
