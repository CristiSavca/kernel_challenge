# Experiment selection critique — November 4, 2026

**Decision:** prioritize one isolated SC6 scale-packing experiment on the validated
KPERM parent `247888…`. Keep architectural research available, but stop treating
a large hypothetical whole-decoder gain as the admission ticket for every next
experiment. No predictor infrastructure or additional GPU work was performed for
this critique. Root owns implementation authorization and official checks.

The current public decoder is still `97af…`, job `7a8c4941741645ee`, score
2.181183. The reported rank is 393. First place remains unachieved. The evidence
does not support a claim that another large rewrite is likely to close the gap.

The principal failure is **selection from plausible resource stories without
enough evidence of an exposed bottleneck**, followed by paying large new costs
to address it. Correctness review has generally worked: failed candidates have
not replaced the public incumbents. CPU proofs establish allowed work and safe
lifetimes under their stated assumptions; they do not establish latency savings.

| Experiment | What the evidence actually established | Selection lesson |
| --- | --- | --- |
| Compact WGS | Two-CTA admission succeeded; the three changed shapes became roughly 34–37% slower than the earlier public parent. The redesign quadrupled stage units and doubled activation requests. | Occupancy capacity cannot stand in for useful work delivered per cycle. Account for work amplification before valuing extra resident CTAs. |
| Small NW4 | Two-CTA admission succeeded; affected normalized latency increased 10.67% versus current public KPERM. | More independent CTAs did not offset the changed per-thread work and pipeline. This is evidence against that fixed tradeoff, not all smaller CTAs. |
| WAIT2 and REG112 | The new variants had explicit compiler serialization warnings and no useful gain. The incumbent's serialization remained unknown. | A problem observed in an experimental variant does not establish the incumbent bottleneck. A repair of a failed branch needs a credible route back above the incumbent. |
| R16 SK2 | Setup stopped at the predefined two-CTA admission gate: fused pooler used 74 registers where the checked register-only ceiling was 72. No timings existed. | A narrow compiler-resource margin is material uncertainty. Static arithmetic should prune impossible choices, not label a barely feasible source fast. |
| Shared-A WGS | Correctness and one-CTA admission passed; affected latency increased 69.02% versus public KPERM and 70.57% versus private KPERM. Controls were within 0.30%. | Removing register operands did not remove global bytes, decode or MMA work. It added 192 MiB of logical shared payload and 49,152 WG barriers per job and reduced lookahead. The check rejects the combined tradeoff, without isolating which cost dominated. |
| KPERM | A localized reduction in repeated scale load/construction work produced consistent positive affected-family evidence and was integrated and scored. Public decoder score rose from 2.150186 to 2.181183, an observed 1.44% change. | Small improvements are usable. The observed score change is not a precise causal estimate, but rejecting it for being smaller than a broad architectural target would have lost our actual progress. |

The recovered pre-KPERM standalone NCU samples strengthen this critique. At M32
and M256 they read approximately 1.009 and 1.007 times the nominal packed-weight
request bytes and reported roughly 75% sustained DRAM throughput. They do not
prove decoder bandwidth saturation, critical-path attribution or a speed limit.
They do make a large same-traffic gain from register scheduling a weaker starting
hypothesis than a concrete reduction in requested bytes or repeated instructions.

Large profile shares identify where a gain would matter, not how to obtain it.
The latest exact `97af` public profiles give these approximate component shares,
weighted by the actual 46/16 layer weights and normalized case times:

| Load | MoE | Dense projections | Attention |
| --- | ---: | ---: | ---: |
| Decode 1 | 34.35% | 52.78% | 8.10% |
| Decode 16 | 43.99% | 33.50% | 17.94% |
| Decode 32 | 39.71% | 29.10% | 25.22% |
| Verify 1 | 46.97% | 41.38% | 8.23% |
| Verify 16 | 53.26% | 28.29% | 13.24% |
| Verify 32 | 54.41% | 24.09% | 15.77% |

These are profile-share calculations, not exclusive wall-time fractions: PDL
waits, overlap, rounded rows and separate profile/timing captures remain. A 1.01×
MAIN-MoE speedup, with every other row fixed, models only 0.2435% less decoder
latency; 1.02× models 0.4828%. Such gains can accumulate, although neither one
would materially close the full gap alone. A portfolio of independently useful
changes is preferable to demanding that each change individually solve the gap.

The saved complement audit also prevents an easy but invalid shortcut: 25 of 27
passing private receipts matched the evaluator and case contract, and none had
any case at least 3% faster than both current public/private incumbents. That
does not establish that every sub-3% change is useless. It does establish that
there is no already-validated strong shape dispatch to assemble from that set.
These receipts are not 25 independent design families; revisions, ancestry,
unchanged controls and adaptive selection make that interpretation invalid.

Use the following selection rules immediately:

1. **Separate correctness, feasibility and value.** First reject a demonstrated
   contract/resource/lifetime defect. Then rank feasible proposals by a concrete
   work reduction or measured bottleneck, the new costs, the fraction of real
   decoder work affected, and implementation/check cost. Do not turn a CPU-safe
   schedule or an occupancy count into a performance label.
2. **Prefer a narrow mechanism when the bottleneck is uncertain.** Record bytes,
   instruction-level work, stages, reductions, synchronization and scratch
   changes before coding. A complicated mechanism can still be worth one
   experiment, but added traffic and barriers need a stronger reason than an
   attractive hypothetical Amdahl calculation.
3. **Allow modest gains and set their gate before results.** A 3–5% whole-decoder
   threshold is a prioritization heuristic, not a universal acceptance rule.
   Use changed-family direction, both matched incumbents and unchanged controls
   to screen a small result. State that overlapping official intervals leave
   uncertainty. Do not adjust a failed experiment's threshold afterward.
4. **Promote evidence through distinct stages.** One useful standalone result
   can justify one isolated decoder integration, which tests a different real
   workload. A useful full decoder can justify one exact-source scored run.
   These are meaningful validation steps, not repeated identical checks until
   a favorable sample appears. Preserve the exact public incumbent throughout.
5. **Use failure to update the next choice.** Do not retry the unchanged failed
   design. An adjacent design can become reasonable after new measurements,
   a demonstrated compiler fact, or a validated prerequisite changes its cost
   structure. Closing SS-A today does not prohibit all shared-operand research
   forever. A relabeled version with the same unsupported assumption is not new
   evidence.
6. **Keep selection machinery small.** Use the existing source hashes, receipts,
   family labels, analytic eligibility checks and a transparent shortlist. A
   trained acquisition model would need architecture-family-held-out validation,
   calibrated uncertainty and comparison with simple ranking on measured data.
   Six benchmark rows from one source, or 139 CPU-lifetime survivors, are not
   independent performance training examples. There is no basis here for a
   claim of predicting billions of fast kernels or a Lean-certified speedup.

The public Rohan release audit found useful measured-score selection and
verification infrastructure, but not the advertised GPU predictor/genetic
dashboard implementation. Importing another coordinator would not supply the
missing measurements. The useful part of that workflow is retaining every
outcome and choosing the next experiment from evidence, which we can do with
the present tools.

**One concrete next candidate: SC6 on KPERM.** The earlier proposal explicitly
deferred SC6 until KPERM was validated. That prerequisite is now satisfied;
there is no saved isolated SC6 run. The pinned Apache-licensed donor contains a
reversible eight-scale-byte to six-byte representation and once-per-stage
reconstruction. Reuse only that mechanism, retaining the incumbent's math,
NTT1 dual accumulators, GS2/wait1, joined producers, finalizer and cost tables.

Qualify every actual local scale byte in prepare against `0x40..0x7e`, in
addition to the existing PRMT predicate. The decision and extension flag must
belong to each prepared state. If any byte fails, preserve the existing
eight-bit WGS representation or general-scale fallback; do not copy the donor's
mutable global encoding selector. Prove multiple prepared states cannot mix
packing with the wrong compiled decoder.

The compressed global piece is 8,960 bytes and stage is 35,840 bytes, versus
9,216 and 36,864. The shared slot stride stays 36,864 with the same three-slot
rings; global offsets, per-WG offsets and joined TMA expected byte counts must
use the correct distinct quantities. No new barriers, stage units, expanded
weights or occupancy benefit are needed. Exact reconstruction introduces extra
bit operations and a second shared load, which can erase the 2.7778% requested
weight-byte reduction. A purely proportional 35/36 MAIN latency scenario maps
to 0.6847% less decoder latency on the current profile model; this is neither a
prediction nor a bound.

Root accepted the modest **predeclared** standalone consistency screen:
affected M32/M128/M256 geometric speedup ratio at least 1.01 against **both**
`d24dd50196cb4158` and `bcb761dfca264db0`; at least two of three affected cases
positive against both; every affected normalized latency ratio at most 1.02;
unchanged-control GM between 0.98 and 1.02; clean five tests and six benchmarks.
It is an operational evidence screen, not a statistical significance test.
Actual source review and CPU proofs precede any official check. A useful result
would justify isolated decoder integration; it would not automatically replace
the public source. SS-A's already-failed gate remains unchanged.

Cluster multicast and fresh WGS cost tables rank behind this candidate now.
The cluster proposal has real activation-request savings but introduces a new
cross-CTA ownership/lifetime protocol without measured L2/TMA attribution.
The retained old timelines lack isolated post-KPERM per-NTT costs, so numerical
replacement tables would be guesses. They are deferred pending better evidence,
not declared impossible.

Evidence: `20261104-wgs-shared-a-result-critic.json`,
`20261104-moe-hardware-profile-recovered.json`,
`20261104-next-dense-attention-roi-gate.json`,
`20261104-saved-shape-complements-critic.json`, the exact public profile
`97af…7a8c4941741645ee.json` (SHA
`35ad7364409d49c2ded613234beea1572adc691c33f3c154a53b3dff313a2414`),
`PUBLIC_MECHANISM_PROPOSALS.md`, `PUBLIC_KERNEL_SCOUT_20261104.md`, and
`ROHAN_REPOSITORY_RELEASE_AUDIT_20261104.md`. The selected donor is commit
`890809cee421500b00126d084e81b338ef9181ab`; its retained decoder source is SHA
`82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
