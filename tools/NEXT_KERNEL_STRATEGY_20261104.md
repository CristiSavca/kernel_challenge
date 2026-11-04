# Next kernel decision, November 4, 2026

**Recommend one isolated exact-one attention output-rescale experiment. No newly supported large rewrite emerged from this bounded review.** This is a small, concrete opportunity to improve rank, not a credible standalone route to #1. Do not reopen the closed MoE queue-depth, shared-A, compact-grid, scale-compression, prefetch, final-scalar, route-reduction or rolling-loop families.

At the saved current scores, decoder2.181183135 versus leader2.5762 requires **15.3333% less normalized latency**. The kernel cutoff reported in the current handoff is November5 08:00UTC. No new leaderboard/provider query was made here. Spending the remaining time on a source-local experiment is more defensible than starting an unsupported architectural rewrite or another general measurement platform.

## Evidence and actual paths

Re-read current `RESEARCH_LOOP.md`, the current NCU/SASS evidence, latest research/scout summaries, the dense structure/dictionary reviews, current decoder dispatch, and the attention task contract plus actual Q1 loop. Exact source pins remain:

| Source | SHA256 |
|---|---|
| `muse_decode_layer/submission.py` | `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2` |
| `muse_moe_w4a16/submission.py` | `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0` |
| `muse_attention_decode/submission.py` | `87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c` |

The current decoder's small-token paths already use SK2 projections, fused pooler/router, UV boundary handling where selected, and UB2 unpooler/boundary handling. `kernel_fn`/`_after_attention` at14871–15066 show the real dispatch. SK2 consumes the 12-bit lossless BF16 representation directly into MMA operands; its correction terms preserve the original weight values. UB2 preloads/decompresses its weights before its MoE dependency wait. There is no newly discovered global decoded-weight intermediate or second full conversion pass to delete. Profile rows around42us of dense work in the approximately72–78us T1 benchmark identify scope, not exclusive latency: PDL can overlap rows and include waits.

Current MAIN MoE diagnostics, `20261104-current-kperm-ncu-result-critic.json`, establish only M32/M256 standalone launches: 77.30/76.99% reported DRAM throughput and read/nominal ratios1.008754/1.007692. The retained complete BIG1 binary already folds the scale broadcasts into HMUL2 and has the expected active MULDEC bit operations. These facts do not prove a hardware ceiling, but they supply no large removable weight pass or redundant broadcast. Math-pipe stall percentages are not removable elapsed time. They also do not establish a bottleneck for the different tiny/MID paths.

The broader alternatives do not yet have a defensible mechanism:

- A local dense exponent dictionary can remove correction participants, but the saved exact CPU representation adds about3% logical weight requests in its distribution proxy and about25% decoder source operations. It does not establish another resident CTA or a shorter critical path. Do not turn this into a deadline-driven rewrite without a new cost argument.
- Precomposing adjacent projections would reduce dimensions in exact real arithmetic, but the actual residual/RMSNorm/gating and BF16 activation boundaries intervene. Distributing a scalar or replacing two rounded projections by prepared weight products changes those boundaries. Existing UV handling already exploits permitted structure; an arbitrary algebraic reassociation is not an exact replacement.
- Expert-owned end-to-end CTAs could retain intermediate values, but current weight bytes dominate and the required hidden-to-output reduction crosses output tiles. Avoiding that exchange requires much less token/expert parallelism or repeated weights/intermediates. No concrete resource/schedule design here establishes net work reduction.

This is a negative scoped finding, not a claim that larger improvements are impossible. The same profile-share model requires roughly43.66% less all-dense time or33.65% less all-MoE time to close the present gap; jointly reducing each by20% models16.07%. No present source proposal substantiates those reductions. These are scenarios from separate captures, not bounds or forecasts.

## Selected experiment: skip only exact-unit O rescaling

In the current attention source at2903–2921, the Q1 non-LAZY path computes `alpha = ex2(m_old - m_new)`. It then updates its row sum and multiplies sixteen FP32 O accumulator values per thread by alpha. For many possible online-maximum histories alpha is exactly positive one; its frequency is not measured and must not be inferred from a particular input distribution.

The precise change is to surround **only** the existing sixteen O products with `if (alpha != 1.f)`. Preserve the existing `l_r[0] = l_r[0] * alpha + ls`, maximum update, exponent calculations, probability FP16 casts, QK/PV MMA order, producer/ring scheduling and every barrier. Use the actual computed alpha, not a tolerance or an inferred maximum comparison. The `ONE` specialization, existing `OKC_LAZY1` branch and all Q8 code remain byte-identical. No new global/shared bytes, tensor allocation, host tuning or runtime instrumentation is required.

This mechanism is released for SM90 in PipeLink's Apache-2.0 [role_core.cuh at4d8ebab9](https://github.com/PKU-DAIR/PipeLink/blob/4d8ebab9f1527589e1cfd892a8a77a3b6211991a/campaigns/attention/src/sm90/role_core.cuh), lines126–132; the pinned primary file was reopened during this review. Its scalar conditional motivates this experiment. The donor's separate deferred-rescale/WGMMA scheduling is outside scope, and no donor speedup transfers to our kernel.

The maximal local source removal is16 FP32 multiply operations per eligible thread/tile. The same loop retains16 QK and16 PV warp MMA calls, nine exponent evaluations per thread, reductions and memory/barrier work. A comparison and conditional execution are added. Alpha is quad-shared, not necessarily warp-uniform; mixed row groups may require divergence or predication. Source counts cannot establish instruction issue savings or elapsed time, and current attention SASS has not been collected. The compiler may already optimize some of this work or may predicate the new branch; direct official timing decides whether it matters.

This is distinct from LAZY1, which defers maximum updates and changes probability magnitude/FP16 rounding, and from the failed Q8 geometry/pipeline rewrites. It also avoids the invalid unproven full-mask vote inside the enclosing `m_new != -INFINITY` branch. Introduce **no warp collective**.

## Correctness and bounded implementation

One author can implement this as a copy of exact87a9 under a new extension namespace, with just the conditional and required provenance changes. A small source-restoration test should establish all other device code and dispatch are unchanged. Directed arithmetic checks cover alpha1/non1/NaN, finite O including signed zero and subnormals, the first live tile (alpha0), empty rows and mixed quad conditions. No new whole-kernel simulator, profiler platform, parameter search or standalone diagnostic GPU allocation is required before the ordinary private check.

IEEE multiplication by positive one preserves finite values, signed zero and infinities when the operation does not flush subnormals. NaN payload/quieting can differ from a copy; alphaNaN must execute the original multiplication. Current flags are `-O3` with no global fast-math/FTZ flag; the explicit FTZ exponent instruction does not establish the O multiplication's lowering. Therefore do not claim universal bitwise equivalence. The attention contract requires finite outputs and all16 official numerical cases, including edges, cache scales and nearby shapes, remain mandatory. Any observed numerical defect closes this exact implementation unless a concrete preserving fix is separately reviewed; no tolerance relaxation or probability/precision change.

The fixed scope is five affected cases: Q1 B1full, B16sliding/full and B32sliding/full. Q1 B1sliding uses ONE and is a control; six Q8 cases are controls. Compare against the exact matching parents: private `c21d391dbf604bb5` and public `8f9e5971f95841a0`. One official private check supplies correctness, selected profiles and paired timing. Do not repeat a failed check to obtain a favorable sample.

Proposed pre-result screen for root to approve: all16 correctness tests and12 benchmark/profile cases pass; affected normalized-latency GM is at most `1/1.01` against **both** matching parents; at least3/5 affected cases improve against both; no affected case has a latency ratio above1.02; seven-control latency GM lies in[0.98,1.02]. This is a consistency screen, not statistical significance. Source validity/resource failure or missing intended activation rejects the attempt. If the speed screen fails, stop; do not convert it into a branch-policy, mask, lazy-offset or threshold sweep. A useful standalone result still requires a separate unchanged-scope decoder integration check before publication.

## Scope ceiling and rank

Using exact97af public profile SHA `35ad7364409d49c2ded613234beea1572adc691c33f3c154a53b3dff313a2414`, normalized benchmark times, layer weights46/16 and the six-load geometric mean, this review recomputed:

| Assumed reduction of the five affected attention paths | Modeled whole-decoder latency reduction |
|---|---:|
|5%|0.3927%|
|10%|0.7880%|
|20%|1.5866%|
|100%, impossible deletion scenario|8.4125%|

The last row is a scope ceiling **within the profile-share model only**, not a physical bound; PDL overlap and separate captures prevent a critical-path interpretation. Removing sixteen multiplies is much narrower than removing an attention path. There is no evidence this experiment yields1% whole-decoder improvement, much less15.33%. Small validated gains can still improve rank; no arbitrary minimum-large-gain requirement is imposed.

Ranking after this review: **(1)** exact-one scalar bypass, because it removes explicit work with minimal state/lifetime risk; **(2)** existing untested QKV escape-row balancing, only if another bounded experiment is warranted. The latter changes no27MiB primary payload, MMA count or activation bytes, adds9KiB row metadata/scatter work, and only attacks an unmeasured CTA escape tail. Granting the donor's unverified0.5us comment to selected rows models only0.13–0.28% decoder reduction. It ranks below the direct attention operation removal. Do not start both as an unbounded campaign.

Serving v22 remains a separate measured component opportunity with generic numerical/concurrency validation outstanding. Its stock-Marlin improvement cannot be credited to the faster current native decoder or used to justify promoting a kernel entry. This strategy note changes no source, score, deployment or GPU allocation.
