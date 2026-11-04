# Next coordination mechanism: insufficient evidence to implement today

November 4,2026. Read-only assessment of exact decoder97af after the fixed R16
attempt. No candidate edits, GPU allocation or official calls. **NO GO for a new
implementation on the current evidence.** This is a prioritization decision,
not proof that every other architecture is exhausted.

The distinct mechanism assessed is to construct stable MoE route/job metadata
once in timed work, using the existing `_route`/`_route_split` tail, then copy it
into each expert CTA. Current `MOE_CUDA_SRC::route` repeats the same work in every
CTA: local expert filtering, warp histograms, prefix sums, ordered `ent`/`pos`,
job splitting and job arrays. MAIN also builds `jrow`, weighted cost prefixes
and CTA ranges. With132 CTAs this repeats considerable instruction work, but
132 concurrent copies of a short prologue are not132 times its wall latency.
This is separate from the closed queue-depth, quota, small-tile and finalizer
experiments. It changes metadata production, not weight packing or tensor math.

Source anchors in the frozen97af snapshot:

- `MOE_CUDA_SRC::route`, lines5843–5960: local counts and stable route ordering;
  `early_issue` begins first-weight progress before the final entry-map pass.
- `moe_fused`, lines8429–8432: DBG0 precedes the PDL wait; lines8589–8594
  route and mark DBG1; later MAIN scans job rows, gathers inputs and constructs
  weighted work ranges.
- `_after_attention`, line14942 onward: `_route`/`_route_split` produces expert
  IDs, route weights and normalized BF16 expert input before `okc_run`.
- `_route_split` is one CTA per token. A centralized metadata tail would need
  all tokens' IDs, including zero/skewed local routes, before stable ordering.

The least invasive concrete design would keep top-k and pooled-input math in
the existing routing kernel. Every token CTA would publish its IDs, then a last
arriver would build the stable expert-major metadata, without other CTAs spinning.
The expert kernel's existing dependency wait would protect consumption; each CTA
would copy the resulting immutable-per-call metadata into shared memory before
using its ranges. Timed counters would need a repeat-safe generation/reset
protocol. Token and expert entries must retain existing order and job splitting;
otherwise this changes padded work or accumulation groupings as well as routing.
Small builds and MAIN have different job policies and metadata layouts. Moving
input-derived work into prepare is forbidden and is not part of this design.

This is mathematically feasible but introduces a new cross-CTA publication
dependency, one serialized metadata construction tail, global stores and132
metadata copies. It also delays the existing early first-weight issue. Merely
fusing top-k into all expert CTAs would duplicate top-k and4096-column norm work
and is a different, unpromising cost tradeoff; it does not remove the required
operations.

Preserved diagnostic `experiments/modal/20261104T043758Z-24108d21/result.json`
contains actual per-CTA globaltimer stamps for the earlier standalone ef7 source.
DBG0→DBG1 includes initialization, PDL and routing; it is neither isolated route
cost nor a guaranteed recoverable window. Across three saved draws:

| M | Median CTA window across replays, by draw | Largest observed CTA window |
| --- | --- | --- |
|32|2.048 /1.888 /1.888 µs|2.240 µs|
|128|2.528 /2.720 /2.560 µs|3.424 µs|
|256|3.424 /3.552 /3.552 µs|4.160 µs|

These are standalone routes and older binaries, so they do not establish current
decoder latency. They supply no affirmative evidence for a large removable
prologue cost. The separately preserved97af profile has route kernels of3.5–5.2
µs, but centralized metadata retains the top-k/norm computation in those kernels
and adds a tail; those row durations cannot simply be counted as savings.

Using the existing97af per-case profiler shares, normalized benchmark times,
46/16 sliding/full weights and six-load geometric aggregation gives these
deliberately optimistic deletion scenarios:

| Hypothetical removal | Whole-decoder latency reduction |
| --- | ---: |
|3 µs from every MAIN case, T≥32|0.564%|
|5 µs from every MAIN case, T≥32|0.941%|
|3 µs from all12 cases|1.971%|
|5 µs from all12 cases|3.292%|

These are scope checks, not predictions or physical bounds; separate profiles
contain PDL waits/overlap. No small-path route-prologue trace supports transferring
the large-path windows. The added producer/copy costs would consume part of any
saving, and there is no measured case for the scale needed to close the current
leader gap. A publication-safe rewrite plus fresh numerical/resource review is
therefore not justified as tonight's next high-impact implementation.

Other algebraic shortcuts do not supply an alternative: moving unpooler through
the expert weighted mixture changes required intermediate BF16 rounding, while
moving pooler through the preceding boundary encounters normalization and that
same rounding constraint. The parent already exploits its reviewed UV split.
Splitting its residual projection earlier would request the large pooler/router
weight stream twice; no cache or critical-path evidence supports that tradeoff.
No attention precision shortcut, failed geometry rescue or new tuning sweep is
proposed.

Compact evidence and exact input hashes are saved in
`experiments/iterations/20261104-next-coordination-feasibility.json`.
The independent critic agrees with this prioritization rejection and its limits.
