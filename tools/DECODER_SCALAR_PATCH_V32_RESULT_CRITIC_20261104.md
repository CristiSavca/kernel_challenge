# V32 independent terminal review — 2026-11-04

**Close the fixed attempt. No retry, integration or public promotion.** Job `6a938fff9d5d417c`, exact source `ea4470fc…`, passes 1 setup / 13 tests / 12 benchmarks / 12 profiles with clean checks. I verified the raw receipt, frozen plan/source pins, external-profile equality and full evaluator/image identity against both97af parents, independently recomputed the official46:16 six-load score, and reproduced the saved root analysis exactly.

| Fixed comparison | Public97af | Private97af |
|---|---:|---:|
| Whole-score ratio, required≥1.025 |0.87479146|0.87661153|
| Affected four-load speed GM, required≥1.04 |0.82386553|0.82588482|
| Control two-load latency GM, required0.97–1.03 |1.01390919|1.01255222|

Score is **1.90808038695**, CI[1.87945918115,1.93670159275]. All eight affected cases exceed the3% latency cap; controls pass. Every timed profile has zeroCOMP=true instances, versus required counts[3,3,4,4,4,4,3,3] on the affected cases. Thus both activation and performance reject independently.

The narrow compiler prediction **succeeded**. All19 observed pointers have zero local bytes and successful resource APIs with positive block capacity;15 are COMP. Their previous32×NT local allocation disappeared. This is compiled-pointer evidence during preparation, not achieved occupancy or timed activation. Resources did not improve uniformly: O TOK8 moves128→133 registers and capacity3→2 despite local128→0. That counterexample prevents equating zero local storage with universally better admission; it does not establish runtime causality.

All20 preparation selftests pass (16COMP,4router), disabled=[], no retained exceptions. There are18 inner choices,24 outer rows, and only the explicit inner STOP; unpooler/router M8 inner choices remain unknown. Of15 retained COMP choices,13 still lose to Torch. Unpooler M1 wins internally9.02vs10.30µs but its BFP chain loses9.89vsTriton8.32µs. Pooler M16 now wins internally24.13vs25.15µs, but separate/fused outer PR chains cost28.51/27.17µs versus25.50µs. QKV M1 remains22.82vs17.63µs. Outer `bfp` can contain inner Torch; its logged choice is before existingPIN_PICK overrides.

Several V32 tuner rows are lower than V31 (pooler M1 39.74→26.94µs, QKV M1 26.69→22.82µs). These are rounded, different-run/input observations, not paired causal gains. They are consistent with a useful compiler change but do not isolate its effect. The measured failure is that the resulting whole options still do not win selection. No current counter attributes remaining cost to descriptor instructions, raw loads, bandwidth, cache or scheduling. The official score measures the fallback mixture, not an activeCOMP benchmark.

The next distinct direction, if exact reconstruction remains the priority, is a **regular per-unit exponent-window decoder with whole-unit raw fallback and no per-word repair selection**, assessed first against retained97af. This changes work rather than repeating scalarization. Existing CPU distribution proxies put raw units at5.86–7.10%; with4-byte headers this means8.33–9.99% more requested weight bytes than768-byte primary units, before padding. That is a serious objection, not an endorsement: simpler decode could lose if bytes dominate. Require exact arbitrary-BF16 proof, actual-weight histograms, complete traffic accounting and unchanged acceptance gates before any new implementation/check. V32 does not establish repair control as the bottleneck, so no gain forecast is defensible.

For scope only, the pre-result97af profile model maps a hypothetical20% reduction in all SK2 time to1.04974× whole score; deleting all SK2 rows gives1.33696×. These assume profile shares transfer to normalized timing without overlap compensation. They are neither physical upper bounds nor predictions, and this family alone does not justify the user's2× target. Stop the failed fixed source; do not force its tuner selection or weaken gates to manufacture activation.

The adjacent JSON preserves raw resource changes, retained tuning evidence, both independent comparisons and exact pins. No provider/GPU calls or source mutations were made.
