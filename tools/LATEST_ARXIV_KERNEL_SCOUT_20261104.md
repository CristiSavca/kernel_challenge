# Additional primary-paper scout — 2026-11-04

Read-only research. No source candidate, GPU/provider call, model download, or upstream code execution. This note adds six September–November papers to the already reviewed FuTransformer, Dissecting Inference, AutoMegaKernel, DFlash, DSpine, MonoMoE, GAAC, RaMP and Chord/Humming material. Paper results below are evidence in their stated settings, not forecasts for Muse. The six are ranked by usefulness to this project, including explicit reasons not to port.

**Best concrete experiment:** borrow only exact-one output-rescale elimination from PipeLink's released Hopper attention core. Inspect and, if independently approved, test it in incumbent Q1 attention. Do not adopt the PipeLink toolchain wholesale. The serving finalizer v22 has meanwhile passed its component screen according to root; its generic GPU/full-serving validation should retain priority over speculative infrastructure work.

## 1. PipeLink — a small source-level attention idea inside a much larger framework

[Paper 2610.37200](https://arxiv.org/html/2610.37200), October 27, 2026. Separately compiled components declare resource and synchronization contracts; placement and generated transfers allow composition across CUDA/CuTe/TileLang. H200 is directly evaluated. Its headline decode composition is Llama-1B, and its frozen Hopper attention core is BF16/head128, not our FP8-KV/head64 path. Thus its reported composition gains do not transfer directly.

Released [Apache-2.0 code](https://github.com/PKU-DAIR/PipeLink/blob/4d8ebab9f1527589e1cfd892a8a77a3b6211991a/LICENSE), verified HEAD `4d8ebab9f1527589e1cfd892a8a77a3b6211991a`, October 28. The [SM90 code map](https://github.com/PKU-DAIR/PipeLink/blob/4d8ebab9f1527589e1cfd892a8a77a3b6211991a/campaigns/attention/src/sm90/README.md) points to `role_core.cuh` and explicitly describes deferred O rescaling and skipping row multiplies when the online scale equals one. **Only the latter is the proposed experiment.** Deferred rescaling would require a separate numerical/lifetime argument. Whole-framework use adds CUDA13.3 nvJitLink, TileLang and Z3 dependencies; the native incumbent already owns its fused schedule. [Toolchain source](https://github.com/PKU-DAIR/PipeLink/blob/4d8ebab9f1527589e1cfd892a8a77a3b6211991a/README.md).

## 2. mKernel — released H200 communication fusion, but not our current distribution

[Paper 2609.13585](https://arxiv.org/html/2609.13585), September 11. It publishes tiles to communication workers as computation completes, distinguishing local tile readiness from larger network-message readiness. Evaluation uses two nodes of eight H200 GPUs; the main opportunity includes inter-node traffic. This is a real released systems mechanism, not an automatic improvement to a single-H200 kernel.

[MIT repository](https://github.com/uccl-project/mKernel/blob/ef6225fdde493f4119f519c99bfa4ad9234e323b/LICENSE), verified HEAD `ef6225fdde493f4119f519c99bfa4ad9234e323b`, October 17. Its [kernel inventory and dependencies](https://github.com/uccl-project/mKernel/blob/ef6225fdde493f4119f519c99bfa4ad9234e323b/README.md) expose GEMM+AllReduce and full dispatch+FFN+combine, with custom transport and libibverbs/EFA infrastructure. **Not a portable direct serving patch today:** pinned Muse locally computes experts from replicated inputs and then all-reduces the weighted BF16 result; it does not use the paper's token-dispatch topology. A future finalizer/collective fusion must preserve the per-expert norm, routed weighting and BF16 boundary before publishing, plus graph-safe per-request completion. Removing these semantics is not allowed. The existing source boundary is `layers/muse_spark_fp4_moe.py` around finalization and its output all-reduce, not a raw GEMM epilogue.

## 3. Weave — route-aware communication allocation, not permission to change routes

[Paper 2609.21483v3](https://arxiv.org/html/2609.21483v3), September 25; first submission September 18. A persistent BF16 EP4 MoE kernel assigns communication/computation SMs from current routing volumes, pipelines chunks and temporarily gives idle communication SMs compute work. It also avoids repeated transfers of tokens routed to several experts on one rank. Evaluation is four H100s, six MoE models, not Muse NVFP4. No author release or code license was located in the paper's links and bounded exact-title search; this is a scoped finding, not a claim that no code exists.

**Applicability:** preserve the principle that actual routes may determine legal work scheduling, never expert omission or altered router scores. Our standalone already groups routed work, varies NTT and uses phase-cost scheduling. Its single-GPU case has no communication SM partition to eliminate. Serving's replicated-input path also lacks the redundant token dispatch that this paper removes. Replacing it with a new EP transport would be a substantial architecture change requiring eight-rank measurements, not an isolated kernel win. No port proposed.

## 4. ZipFusion — exact KV compression, but the new paper matches our old dense codec

[Paper 2610.33719](https://arxiv.org/html/2610.33719), October 24. It compresses eligible 512-value **BF16** KV blocks to 768 bytes, keeps low bytes, packs sign plus three exponent-offset bits, and reconstructs Tensor Core register operands directly. Unqualified blocks remain raw. Full pages may be compressed after attention stops reading them; partial pages remain appendable. H100 experiments integrate FlashInfer and SGLang. No project code/license link was located in the paper or bounded exact-title search.

**No direct port:** our required cache is already FP8 E4M3, so BF16-to-12bit storage would increase its bytes. Our dense SK2 already performs the same essential 12bit low-byte/nibble direct-register reconstruction. Replacing it with per-block raw fallback reopens the already examined 12/16bit traffic/decoder tradeoff. The paper's capacity-driven larger-batch throughput also does not establish lower latency at serving's fixed concurrency. Incremental cache relocation would require page-reader lifetime and graph/concurrency proofs absent from a kernel-only substitution. This independently supports keeping decode fused; it supplies no fresh lossless byte saving for the active formats.

## 5. Why Speculative Decoding Works Better Than Predicted on Sparse MoE Models

[Paper 2610.38981](https://arxiv.org/html/2610.38981), October 28. Across ten models, route overlap and L2 reuse make verification cost grow more slowly than a simple independently activated-expert-byte model predicts. Actual runtime shares differ from parameter shares. The hardware/model grid includes H100 and B200 and several precisions; none is a Muse measurement. No standalone optimization kernel/code license was located in its linked material.

**Useful serving decision:** measure actual verification route overlap and latency under the required DFlash configuration before choosing a different expert schedule. Do not infer a bottleneck from weight size, multiply isolated layer gains into a serving forecast, change draft weights, or silently change the evaluator's speculative configuration. This is consistent with our current H200 NCU record: real DRAM reads are only about 1.008 times the per-job weight payload, so a new redundant-load-elimination story needs evidence. Existing runtime route grouping already captures some reuse. This paper is a better measurement model, not a source implementation to transplant today.

## 6. Formalizing PTX in Lean — fresh useful limits, no automatic optimization proof

[Paper 2611.04447](https://arxiv.org/html/2611.04447), November 3. Its Lean model includes async operations, acquire/release and Hopper WGMMA examples, but the result is partial correctness of modeled completed executions. It does not prove termination; floating-point values are real-number abstractions, so BF16/FP8 bit reinterpretation and rounding are outside that model. The examples use 454 PTX instructions and roughly 34,000 proof lines. No public artifact/license link was located in the paper.

**Do not advertise this as a ready-made correctness or physical performance pruner.** It cannot currently certify our packed-bit decode, exact conversion, deployed compiler, or multi-GPU progress. Its useful immediate contribution is a sharper review contract: distinguish modeled barrier ordering, actual numerical correctness and successful termination. Our CPU source/event checks must retain those limitations. A new Lean framework is not the next performance experiment.

## One bounded experiment derived from the released source

**Subsequent independent review supersedes the initial warp-vote sketch below.**
Use a simple `if (alpha != 1.f)` around only the output multiplies as the
preferred proposal, introducing no collective. The review identifies five
affected cases and seven controls: Q1 B1 sliding already selects ONE and skips
this work. The initial six-Q1 selection screen below was never accepted or run;
choose and freeze a corrected screen before any candidate submission. See
[the exact-rescale critic](ATTENTION_EXACT_RESCALE_RESEARCH_CRITIC_20261104.md)
for the actual dispatch coverage, numerical requirements and limited modeled
whole-decoder effect. This remains below generic serving-finalizer validation.

Parent: exact attention `87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c`, at `experiments/candidates/muse_attention_decode/87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c.py`.

At lines2903–2921, the ordinary Q1 branch computes the existing `alpha = ex2(m_old - m_new)`, updates the row sum using `l * alpha + ls`, and unconditionally multiplies sixteen FP32 output accumulators per thread by alpha. `OKC_LAZY1=0` is the default at693. The alternate lazy-offset branch is recorded as neutral and changes offset/probability scaling; it is **not** this proposal. Q8 already uses rare offset rescaling, so leave it untouched.

Proposed fixed delta, subject to independent source/semantic review:

- Keep computation of alpha, max, probabilities, row sum and every MMA exactly where it is. In the non-ONE ordinary Q1 path, use a warp-uniform vote on `alpha != 1.f` to execute the original sixteen output multiplies only if any lane needs them. Do not replace the row-sum FMA/add expression or alter the online maximum.
- Work removed: sixteen FP32 multiplies per participating thread on each all-one warp/tile. Cost added: a vote/predicate/branch on every such tile; there are no new loads, stores, shared buffers, barriers or cache layouts. If different quads need rescaling, the whole warp retains all original multiplies. No savings frequency is currently measured.
- Before implementation, confirm the source scope is reached in all six Q1 cases and compile-time ONE paths cannot be accidentally altered. `87a9` public profiles0–5 are Q1,6–11 Q8; public Q1 full-context M16/M32 attention times are approximately86.8/166.1us, but those times are not recoverable multiply time. Kernel compiler behavior and existing predication may erase the apparent source-level saving.
- Numerical gate: for all relevant finite accumulator values, multiply by exactly1 preserves the value; preserve alpha and all actual rescaling when non-one. Explicitly review signed zero, subnormals/FTZ and nonfinite payload behavior against the allowed input domain. If deleting the multiply changes an observable required case, reject rather than relaxing precision. Do not claim a host oracle proves GPU NaN behavior.
- Vote-participation gate from independent reviewer: the enclosing `m_new != -INFINITY` branch is only known to be quad-uniform from `quad_max`. A full-mask warp vote inside it is not automatically legal. Prove warp convergence on the complete supported domain or choose a correctly participating active-mask formulation without altering arithmetic. A nonfinite/masked-head counterexample must cause rejection or a safe unchanged fallback, not an unproven `0xffffffff` vote. Exact vote placement/mask remains unresolved in this research proposal.
- First validation should be lightweight source restoration plus directed alpha0/1/non-one and mixed-quad review. If approved, run one unchanged official16-test/12-benchmark attention check, with the six Q8 cases as controls. Suggested pre-result selection screen: Q1 geometric-mean normalized speed at least1.01 versus both frozen public/private parents, at least4/6 jointly positive Q1 cases, no Q1 latency regression over2%, Q8 normalized-latency GM within2%, and aggregate official score improved versus both. Root must accept the screen before any submission. Failure closes this exact experiment; no threshold, vote-policy or tile sweep follows.

This is an implementable, small instruction-removal hypothesis. It is not a claim that PipeLink's linker or its published speedup applies to us, and it does not justify postponing the now-measured serving finalizer work. No source has been edited for it.

## Scope and release evidence

Primary HTML and bounded GitHub API/raw reads only. Both repository recursive tree responses reported `truncated=false`; this scout inspected metadata/path lists and selected README/license files, without a clone/build. PipeLink's relevant source locations are `campaigns/attention/src/sm90/role_core.cuh`, `role_load.cuh`, and `role_epilogue.cuh`. Its source map names the exact-one optimization; root's follow-up actual-body check below confirms that local mechanism. A complete implementation/numerical review is still required before any candidate.

Additional fresh papers screened but not promoted: [PagedMoE, November2](https://arxiv.org/abs/2611.03173) changes per-token expert precision and targets SSD-resident weights; [MpFA, September27](https://arxiv.org/abs/2609.33135) uses QK4/V8 plus Blackwell facilities; neither preserves the active precision/hardware contract. [Orthrus finite-precision study v2, September27](https://arxiv.org/abs/2609.15504) reinforces validating generated outputs, not inferring bitwise behavior from mathematical speculative-decoding equivalence. Older Unweight/ZipServ/Approaching Shannon Bound were discovered but are not counted as new September–November releases.

Root additionally inspected the actual pinned [role_core.cuh](https://github.com/PKU-DAIR/PipeLink/blob/4d8ebab9f1527589e1cfd892a8a77a3b6211991a/campaigns/attention/src/sm90/role_core.cuh): 14,868 bytes, SHA256 `efb0c5fff861e8cc46075a293875d31213839eec06eb88bc64ce8739e55a03da`. Lines126–132 guard accumulator multiplies with per-row `scales(m) != 1.f`. This confirms the local mechanism in executable source, beyond the README; it is not a complete kernel audit. The donor's per-row guard differs from our proposed warp vote, so it does not settle our convergence question. Root also verified that the incumbent build has no `--use_fast_math` or `-ftz` flag; its `ex2` intrinsic itself uses FTZ. These facts narrow, but do not replace, the directed arithmetic review.

All release-absence findings are limited to inspected primary material and bounded searches. No paper's arXiv redistribution license was treated as a code license. No leaderboard or performance promotion follows this note.
