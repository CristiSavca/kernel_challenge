# V33 independent terminal review — 2026-11-04

**Close V33 and the V29→V33 consumer repair/completion line. No retry, parameter rescue, shape-only splice or public promotion.** Job `60b61041d08a4880`, exact source `a8a619ea…`, passes the complete 1 setup / 13 numerical test / 12 benchmark / 12 profile contract with clean checks and flags. The independent strict analysis equals root's saved result; raw receipt, profiles, source/plan pins and full evaluator/image identity match. I separately reproduced the official six-load score using 46:16 sliding/full weights.

| Fixed comparison | Public 97af | Private 97af |
|---|---:|---:|
| Whole-score ratio, required ≥1.025 |0.90501883|0.90690179|
| Affected four-load speed GM, required ≥1.04 |0.86448992|0.86660878|
| Control two-load latency GM, required 0.97–1.03 |1.00820076|1.00685143|

Score is **1.97401181553**, CI [1.94440163830,2.00362199276]. All eight affected cases exceed the 3% normalized-latency cap against both parents. Controls pass. This is a decisive performance rejection independent of activation completeness.

Unlike V32, some new compressed kernels actually run in timed profiles. Counts on the eight affected cases are **[1,1,2,2,0,0,1,1]**, versus required [3,3,4,4,4,4,3,3]. No affected case restores the complete incumbent set. Verify 1 also selects a TOKR8 pooler signature where the incumbent uses TOKR16; preserving fusion code did not preserve the selected runtime path. Shortened symbols do not identify hidden template/runtime fields.

All 22 resource rows (18 COMP, 4 router) have local 0, successful APIs and positive block capacity. Registers are lower on corresponding compressed specializations—for example QKV TOKR8/16/32 now 81/79/108 versus V32's115/109/136—without restoring full selection. These are compiled-pointer attributes, not achieved occupancy or causal timing evidence.

All 20 preparation selftests pass, disabled=[], no retained exceptions. The log retains 18 inner choices and all 24 outer choices; only the inner group stops, leaving unpooler/router M8 unobserved. QKV remains slower internally: 20.58 vs 17.82µs at M1 and 21.41 vs 17.86µs at M32. Fused pooler now wins the outer comparison at M1–16 (about 24.22–24.38µs), explaining partial activation; at M32 it loses 26.66 vs Torch 26.30µs. An outer `bfp` label may contain inner Torch and precedes existing PIN_PICK overrides. These observations are not a same-input cross-version causal experiment.

Actual logged sidecar-unit rates are approximately 4.93–6.67%, with total logical weight requests 2.16–2.75% above the 768-byte primary. The full codec remains numerically acceptable on the recorded tests. The line nevertheless failed its central performance proposition: eliminating correction warps and replacing repair storage/work did not improve the retained decoder. The result does not isolate sidecar latency, instruction issue, cache, occupancy or the removed correction schedule as the cause. We should not turn that uncertainty into another nearby variant.

The next justified action is **one bounded counter capture of original 97af's selected dense kernels**, not another completion variant. The retained NCU evidence is standalone MoE; no corresponding dense capture distinguishes original SK2 bandwidth, instruction, shared-memory or barrier pressure. Use untouched `bfp_prepare` on pinned reference-generated dense weights, then `bfp__run_sk2` QKV M16 and `bfp_fused_pooler_router` M16 at their natural `bfp_SK2_PDL=0`. Pin the source/image/compiler, exact template/PK/FUSE/shape, activation, weight generation and escape/pmax counts. This is standalone dense evidence, not UV, PDL or whole-forward timing.

Run ordinary cold graph correctness/observations separately from NCU replay. From the actual metric-availability query, use a fixed declared set covering DRAM/L2 traffic and throughput, issue/eligible warps, shared load/store traffic and bank conflicts, barrier/short/long scoreboard, and math/MIO/LG/no-instruction pressure. Preserve raw counters, actual pass count and cache/replay policy. Missing counters remain unknown; no substitutions, parameter search or retry for favorable data. Aggregate stalls cannot identify escape PCs or become additive recoverable-time percentages.

The experiment is decision-changing only if provenance, exact activation and numerical checks pass and the counters distinguish a broad pressure class. If they do not, call it incomplete or inconclusive and avoid another escape-path rewrite from those counters. For example, high DRAM pressure would weaken an instruction-only argument; shared conflict/scoreboard or issue pressure would motivate a different source-specific hypothesis, without proving its causal share. This diagnostic has zero direct score gain and no defensible recoverable-time estimate. Fresh guard admission and separate worker/lifecycle review remain prerequisites; this note authorizes no allocation. The small previously identified QKV row-balancing donor remains lower-priority, unmeasured scope rather than a claimed solution.

The adjacent JSON records independent ratios, activation, resource/diagnostic coverage and exact evidence hashes. No provider/GPU calls, candidate edits or frozen-evidence changes were made.
