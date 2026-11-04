# WGS prefix-only parallelization: bounded feasibility

2026-11-04. **Feasible as one small, isolated experiment after SC6; confidence in
a useful speed gain is low.** The donor already supports more than32 jobs with
a chunked scan. There is no need for a new scheduler or platform. Saved timelines
show essentially no exposed wait at the join that follows prefix/gather, so do
not transfer the donor's prologue claim into an expected gain. A cheap experiment
can still test instruction/shared-memory contention; small gains are acceptable.
This is source feasibility only: no candidate or GPU run was created.

Parent: `native_wgs_kperm3_v9.py`, SHA
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
Donor: [keell0renz decoder at890809c](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_decode_layer.py),
SHA `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
Preserve its Apache-2.0 attribution if using the scan. Donor lines6152 onward
combine a scan, parallel boundary searches and changed per-unit costs; extract
**only the scan**. Its comment claiming roughly1500 critical-path instructions
is an author assertion, not measured evidence for this parent.

## Exact smallest source change

Current parent `wgs_prefix` at3228 serially writes both job prefixes in lane0.
`wgs_ranges` at3252 then runs the original two `wgs_range` calls, producing four
boundaries. With default `OKC_WGSPR=1`, lane0 of the X-producer warp executes this
after barrier7 while consumers gather X. All producers/consumers subsequently
join barrier2. Cost tables, job order, ranges and split-unit owners determine
exact producer/consumer work; none should change.

Use the donor's32-job chunk structure inside `wgs_prefix`, with the **parent**
increments, not its added unit costs:

```
ia[j] = wgs_cost(0, S.jC[j]) << 9
ib[j] = wgs_cost(1, S.jC[j]) << 8
```

For each `j0=0,32,64`, participating lane `l` handles `j=j0+l`. Lanes beyond
`nj` contribute zero and still execute every full-mask shuffle. Inclusive scans
use offsets1,2,4,8,16 for each phase. Store `carry+inclusive-increment` at valid
`j`; broadcast lane31's inclusive value to advance both carries. Lane0 writes
the two terminal prefix entries and totals. A full-warp `__syncwarp()` makes all
prefix stores visible before lane0's unchanged binary searches.

Keep `wgs_ranges` explicitly noinline as before. All32 lanes call the prefix;
only lane0 executes the original `wgs_range` calls and optional debug block.
Do not parallelize the boundary searches in this experiment. Two call-site
changes are necessary:

- Default X-producer: replace its lane0-only call at2841 with an unconditional
  warp call. Its existing following `__syncwarp()` publishes lane0's four ranges
  before `WS_JOIN`; retain barrier7 and its544 participants exactly.
- Inactive `OKC_WGSPR=0` branch: change `if(tid==0)` at3552 to `if(tid<32)`.
  All32 compute-warp0 lanes participate; the existing subsequent `WS_JOIN`
  publishes ranges to the whole576-thread CTA. Other compute warps do not call
  a full-mask shuffle. Keep this fallback source valid without enabling it.

No gather instruction, job sorting, shared allocation, scratch offset, CTA grid,
TMA stage, MMA, completion count, finalizer, PDL operation or cost coefficient
changes. Existing `wgs_stage_at`, `wgs_range`, `wgs_cta_of` and every caller's
rounding/tie rule remain byte-identical. This preserves the partition of work,
including phase-A trailing-partial reorder and split-fixup ownership.

## Ragged chunks, arithmetic and limits

MAIN FT64 gives `JMAX=32+3072/64+8=88`. Cases1/31/32 use one scan chunk,
33/64 two, and65/88 three. With one job,31 lanes contribute zero; this can cost
more than the parent's single serial iteration. Zero-padding the final chunk
ensures lane31 still carries the exact full chunk sum. No array entry beyond
`nj` is read or written, except the intentional terminal entry at index`nj`
in the89-element prefix arrays.

The per-lane invariant is: before chunk`j0`, both carries equal the sum over
jobs`[0,j0)`; after an inclusive scan, lane`l` holds the sum over valid jobs
`[j0,j0+l]`. Subtraction gives the parent's exclusive prefix. Costs are positive
integers and prefix addition is exact, so all existing searches see identical
values, including exact ties. There is no floating-point reassociation.

The actual cost vectors remain:

```
A = [112,135,140,158,173,185,200,207]
B = [100,104,115,123,132,144,156,174]
```

For positive job counts1..64, conservative88-job maxima are9,326,592 for A and
3,919,872 for B. At the measured H200 grid132, their largest boundary products
are1,231,110,144 and517,423,104, below2^32. Prefixes also fit signed32-bit stores.
The shuffle/add arithmetic may use uint32 semantics, but no wrap occurs in this
domain. This does not authorize arbitrary enlarged grids or changed costs.

For `nj=0`, the prefix itself writes valid zero totals without a shuffle chunk.
The inherited `wgs_range/wgs_stage_at` then accesses job0 and divides by its cost;
**this proposal does not establish global-zero-job support**. Keep the original
supported nonempty-work domain and disclose that inherited limitation. A zero-job
fix would be a separate behavior change, not silently bundled with the scan.

A bounded offline arithmetic check retained in
`experiments/iterations/20261104-wgs-parallel-prefix-feasibility.json` compares
276 count arrays: all`nj=0..88` with three directed patterns plus nine saved
actual-route job lists in parent fat-first/skinny-second order. It checks552
phase prefixes and144,144 boundary values at grid132. Some array-domain stress
patterns exceed realizable route totals; they are conservative arithmetic tests,
not claimed workloads. These are abstract integer checks, not actual compiled
CUDA or a new proof framework. Actual-source conformance is needed only if a
candidate is later selected.

## Cost and available performance evidence

The parent does two cost evaluations and two prefix stores per job, all from one
lane, plus terminal stores and the four serial boundary searches. The proposed
scan retains those logical reads/stores and costs, but distributes each chunk
across a warp. It adds ten shuffle-up instructions and two carry broadcasts per
chunk, predicated additions and one publication synchronization. It removes
the serial inter-job dependence in batches of32. These are source operation
counts, not SASS counts or a32x speed forecast. Prefix storage and resident CTA
count remain unchanged; actual register allocation may still change.

Saved standalone routes have J16–18 at M32, J24–26 at M128 and J26–29 at M256;
all fit one chunk. Decoder histograms are not established by these standalone
inputs. Every CTA redundantly computes the same prefixes; no cross-CTA sharing
or new global metadata is proposed.

An existing exact-ef7 diagnostic has thread0 timestamps `DBG7` immediately
after gather and `DBG6` after the following `WS_JOIN`. Root's original evidence
file is pinned in the new concise JSON. Across9 inputs ×4 replays ×132 CTAs,
every `DBG6-DBG7` is0..64ns, with32ns median for every input. One M32 input had
over5% instrumentation perturbation; all eight remaining inputs still have the
same64ns maximum. This join also waits for the producer warp that computed the
prefix. It provides **no positive evidence of an exposed microsecond prefix
stall**. The broader prologue medians4.16–9.95us include routing, gather and
other work and cannot be assigned to this serial loop.

These coarse, instrumented standalone timestamps are not a direct prefix timer
or a causal upper bound on a replacement. They use older ef7, write-flush timing
and no real upstream PDL workload. Reduced issue/shared contention could help
gather even if the final join is short; altered compiler allocation could hurt.
Current KPERM and decoder evidence does not resolve that question. Do not sum
all CTA gaps, transfer donor timing claims or infer zero possible benefit.

Affected scope is MAIN M32/128/256; small paths stay unchanged. As a scale
illustration only, the saved exact97af profile model maps a hypothetical3%
MAIN-MoE speedup to about0.72% whole-decoder latency reduction. This is neither
a gain requirement nor a prediction. Several small validated changes can count.

**Selection:** retain one narrowly scoped prefix-only test behind SC6 if its
implementation and source check remain inexpensive. No preparatory GPU timing
campaign is justified. Review exact arrays/ranges and collective participation,
then use the existing official comparison machinery if root selects the test.
Do not add parallel searches, new cost fitting, dynamic scheduling or a parameter
sweep to rescue a weak result. If SC6 passes, any later combination must use an
explicit new parent and measured attribution rather than silently mixing changes.
