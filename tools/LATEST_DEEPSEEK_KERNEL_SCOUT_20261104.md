# Latest DeepSeek primary-source kernel scout — 2026-11-04

**No newly released drop-in kernel found for our H200 BF16-activation/NVFP4 Muse path.** There is useful released work on eliminating intermediate layouts, communication epilogues and fused auxiliary operations. The exact precision, hardware and caller boundaries prevent transferring the headline speedups directly. Continue the already reviewed small-serving-finalizer experiment; do not replace its evidence gate with a paper result.

Fresh public GitHub/Hugging Face metadata and selected pinned source were read through November4. Evidence is `LATEST_DEEPSEEK_KERNEL_SCOUT_20261104.json`,459496 bytes, SHA256 `3109ea6e1d7a5dd85371444635be25e53d1df0669227de5b79ece00d9c2a1593`. No checkout, model download, GPU/provider call, candidate change or serving-source edit occurred. Main-head dates below are commit dates, not necessarily release dates; a repository's later `updated_at` does not establish new kernel code.

| Official repository | Exact reviewed main head | Date | License |
|---|---|---|---|
| DeepInfer | `a679bb86ce36e883a98a97733492b686e5b9a08e` | Nov3; documentation change | Apache-2.0 |
| DeepGEMM | `5081f8513af57e61696de4f4f04b0e707ffb503e` | Oct20 | MIT |
| DeepEP | `f6aa38fe06cd4a0489e7730365d4e0114970a2f3` | Oct21 | MIT |
| FlashMLA | `395cb3afbb331fef7b1b21e9340861b9436b42d8` | Oct27 | MIT |
| TileKernels | `c1cf914a82e6901c679445f872051ce44ac2cc63` | Oct14 | MIT |
| DeepSelect | `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` | Sep30 | MIT |
| DeepJIT | `d19237c44ad11b1b22d1939711fbfd65eaf5944f` | Nov3 | MIT |

Repository licenses permit source study/reuse subject to their notices; an unmerged contributor PR is not a validated DeepSeek release. The JSON retains exact URLs, commit metadata, source hashes, license text for the principal donors, and the local pinned caller hashes. No external code was copied into an implementation.

## 1. DeepInfer and V4.1: genuinely recent deployment material, mostly a different problem

The [released DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md) supports DeepSeek-V4.1-Flash on **SM100 or Ascend950**, with offline inference rather than a complete service. Its displayed performance uses4 SM100 GPUs for prefill and64 for decode. The Nov3 commit updates the deployment Q&A; it is not a newly landed H200 kernel.

The [Nov3 deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md) provides concrete engineering detail: expert replication/load balancing, separate CPU/GPU scheduling, and confidence-based speculative verification. It explicitly credits SGLang/NanoFlow for the asynchronous scheduling approach. Pinned SGLang already has `enable_overlap` and `event_loop_overlap`; porting that feature by name is not new work. DeepInfer's expert weights use MXFP4 per32/UE8M0, not Muse's NVFP4 per16/E4M3 plus global scale.

The latest official HF checkpoint observed is [V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), updated Oct17. The [Sep17 V4.1 report](https://arxiv.org/abs/2609.19969) obtains cache savings through CED/CSA2, cross-layer reuse and FP4 KV storage. Those are model/cache changes. DeepInfer's bounded SWA replay is explicitly approximate; it cannot replace Muse's required full-window/full-context attention. DSpark's confidence head cannot replace the supplied DFlash model/algorithm contract.

**Useful experiment:** when end-to-end serving measurements are available, identify actual GPU idle gaps and distinguish CPU scheduling, expert imbalance and collective latency before proposing an overlap change. **Kill:** the feature is already active, the change needs SM100/model changes/new draft weights, or it skips required keys/routes. No DeepInfer benchmark forecasts an eight-H200 Muse gain.

## 2. DeepGEMM: real Hopper source exists, but the attractive fused FP4 paths miss our contract

Main's [MegaMoE API](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/apis/mega_moe.hpp) dispatches its fused FP8×FP4 and BF16×BF16 paths only for architecture major10. Ordinary SM90 BF16/FP8 GEMMs exist; that does not make fused BF16×NVFP4 available. The merged [Sep30 locality release](https://github.com/deepseek-ai/DeepGEMM/pull/462) reports lower power/higher frequency for its heavy MegaMoE workloads and new SM100 epilogues. It is not H200 evidence.

Two relevant **unmerged contributor PRs** require different interpretation:

- [#473, Oct14](https://github.com/deepseek-ai/DeepGEMM/pull/473), head `724a799bd04382732df8ba73611b4a2702879ac6`: register-A WGMMA decoding, small-M buckets and split-M tails for **FP8×MXFP4**. Reported EP8 gains versus its MXFP4 predecessor are1.10–1.44× on H20-3e, using DeepSeek Flash/Pro dimensions, top-k6 and seed0 graph measurements. H100/H200 validation is listed as follow-up. This is not NVFP4, not BF16 activations, and lacks Muse's post-expert norm. [#454](https://github.com/deepseek-ai/DeepGEMM/pull/454), head `d5925deb53a26895efad49395831f0bcf30d13b1`, mentions NVFP4 but still uses FP8 activations and is also unmerged. Neither can be transplanted without changing required precision.
- [#463](https://github.com/deepseek-ai/DeepGEMM/pull/463), current head `ca1291f7cd2dbe483af7d89f033b2a5a1e96ca1f`: BF16 gathered-row GEMM avoids materialized `index_select`; its earlier controlled H200 graph study reports about6% at a large routed shape, but a small graph case regresses2.31%. Current revision has build/correctness evidence, not a repeated performance study. It requires BF16 weights, aligned expert groups and a row map. Our selected Marlin path already supplies aligned route indices directly and stores NVFP4 weights, so its removed gather is not present at this boundary.

**Useful experiment:** only if a profile identifies a genuine materialized BF16 gather feeding an already-BF16 GEMM elsewhere, compare direct indexed loads under identical graph timing, including small batches. **Kill:** no such copy exists, weights must be dequantized persistently, or activation precision changes. Current native MoE already uses register-A decoding; these PRs do not justify reopening its closed SS-A/compact/decode variants.

## 3. DeepEP V2.5: deferred epilogues are interesting, but not a communication drop-in

At the pinned head, [DeepEP V2.5](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/README.md) removes the old V1/NVSHMEM API, uses NCCL Gin/DeepJIT and supports Hopper. Requirements include CUDA>=13.1, Torch>=2.10 and NCCL>=2.32.3. It adds deferred epilogues, cached expanded layouts and bucket collectives. It does **not** claim zero-SM EP communication. No matching eight-H200 Muse measurement is published in the inspected material.

The [actual EP API](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py) exposes BF16 dispatch and **BF16 input to combine**. Deferred dispatch can defer a CPU receive-count wait/copy epilogue; it is not proof of arbitrary fresh-route CUDA-graph compatibility. Its [combine epilogue](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/combine_reduce_epilogue.cuh) lacks Muse's per-route weighted RMSNorm.

Our pinned H200 Marlin path computes local outputs from replicated tokens, applies FP32 route norm/weight/sum, casts the rank result to BF16 and then all-reduces. Feeding per-route normalized outputs into DeepEP combine would add an intermediate BF16 rounding. Retaining raw outputs and fusing the norm inside a custom reduction is a separate substantial kernel design. Pinned SGLang's DeepEP adapter imports old `Buffer, Config`; the Muse Dockerfile pins `sgl-deep-ep==0.1.2.post1`, so current V2.5 is an API/dependency migration, not an environment-variable switch. Installed runtime versions were not queried.

**Useful experiment:** first measure finalizer plus current eight-rank all-reduce and verify which portion is material; a subsequent fused epilogue must preserve the exact existing route and rank rounding boundaries. **Kill:** dependency/patch limits prevent the needed runtime, FP32 route values must be rounded early, dynamic routing needs host synchronization during graphs, or extra dispatch/layout conversion cancels the saved work. Do not replace current communication based only on a library headline.

## 4. FlashMLA: latest release is explicitly incompatible; older Hopper FP8 is sparse

The [current FlashMLA README](https://github.com/deepseek-ai/FlashMLA/blob/395cb3afbb331fef7b1b21e9340861b9436b42d8/README.md) says September30 removed Hopper and earlier-model support and changed FP8/FP4 cache formats. Its recent race fix targets SM100. The linked [last compatible lineage](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/README.md) has Hopper kernels, but FP8-cache decode is sparse-only; dense decode expects BF16/FP16 KV. The documented packed MLA caches use512/576-dimensional latent layouts, not Muse's separate paged FP8 GQA K/V tensors. Published H800 bandwidth/TFLOPS and B200 fused norm/RoPE results apply to those interfaces.

**Portable idea:** fuse genuinely adjacent norm/RoPE/cache/output operations without changing their casts. Our current native attention already combines several of these operations; any new serving fusion must first identify an unfused selected path. **Kill:** adding sparse indices, dropping keys, converting the cache to FP4/BF16, requiring SM100, or quantizing BF16 projection weights. No new attention candidate is supported by this scout.

## 5. TileKernels/DeepSelect: inspect selection algorithms, but the missing fusion is already present

[TileKernels' current routing implementation](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/moe/moe_topk_gate_forward_cuda.py) keeps local scores, uses warp score/index comparisons, retains one selected ID per output lane, and gathers unbiased scores from shared memory after selection. Its actual scoring is `sqrtsoftplus`, with finite-input assertions and its own normalization epsilon. Replacing Muse's sigmoid or tie/NaN rules would be wrong. TileKernels also requires a newer TileLang/CUDA stack; availability in the fixed image is unverified.

Pinned Muse already prioritizes a CUDA JIT router over Triton for eligible shapes. Its `routing.cuh` sorts each lane's local candidates, exchanges winner ID and **unbiased score** using shuffles, then accumulates selected scores. Therefore the shared-score gather is not a newly missing serving operation. Our rejected v20 concerned the separate native decoder's Triton tuple reduction; this scout does not reopen it.

[DeepSelect](https://github.com/deepseek-ai/DeepSelect/tree/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18) reports2–20× against `torch.topk`, primarily BF16 sparse indexer top-k512 or FP32 sampling near128K vocabulary. That is not evidence for256-expert top8 or already-fused sampling. **Useful experiment:** only profile a selected, still-materialized sampling/top-k bottleneck, preserving exact outputs, tie order and model sampling. **Kill:** existing fused backend is already active or the candidate's dtype/score/NaN semantics differ. No routing rewrite recommended now.

## Decision

The fresh research strengthens a concrete principle already being tested: remove full-sized intermediates and launches while retaining all model arithmetic. It does **not** provide an H200-compatible library swap that has been overlooked. The highest-confidence immediate work remains the reviewed serving finalizer. The next research-to-experiment boundary is actual eight-rank finalizer/collective attribution once available; large-scale DeepSeek results are context, not a forecast. Do not install new dependencies, change model precision, replace DFlash, or launch a new search platform from this scout.
