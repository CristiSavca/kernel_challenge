# Selecting the next experiment from the evidence we have

2026-11-04, 13:08 UTC evidence cutoff. **Use an analytic eligibility filter and a
transparent experiment priority list now. Do not train and advertise a calibrated
kernel-speed predictor from this dataset.** A small regression can be fitted, but
the saved data does not establish that it predicts an unseen architecture better
than retaining the incumbent. This conclusion does not require a large minimum
gain: several small, measured improvements can accumulate.

## What the announced workflow supplies today

The user's November 2 post describes genetic kernel search, a trained statistical
predictor and Lean bounds. The retained repository audit distinguishes these
announced capabilities from the released implementation. A fresh, read-only
[GitHub main-commit query](https://api.github.com/repos/rohanarun/Auto-research-agent/commits/main)
returned `018b7a102348ea3b73701e8be52d24f82d8c91ee`, dated October 30
15:20:24 UTC; the response Date was November 4 13:08:18 UTC. Main has not changed
since the 09:12 audit. This refresh did not re-audit every branch.

Released [ResearchMethods.py](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Resources/ResearchMethods.py)
selects among measured, validated branches and records source/evaluator identity.
[LeanSearch/cpu_search.py](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Resources/LeanSearch/cpu_search.py)
supplies a finite mathematical search adapter, not an H200 performance model.
The earlier audit found no released GPU genetic engine, trained GPU predictor or
Lean GPU resource theorem in its inspected files. That says nothing about private
tools. Preserve MIT notices for workflow reuse and upstream kernel licenses;
there is no reason to install the whole application for this task.

Our existing `search_wgs_queue_cpu.py` already implements a useful part: a pinned
manifest, explicit oracle, negative controls and retained rejection reasons.
Its 320 schedules include 181 unsafe and 139 lifetime-safe outcomes, 96 abstract
behavior classes, and **zero performance labels**. The selected WAIT2 schedule
predated this wrapper. Its later correctness pass and performance regression
demonstrate why schedule safety is not a ranking by speed. Do not describe
enumeration throughput as measured kernel predictions per second, or executable
CPU checks as Lean proofs.

## Actual data inventory and its limits

A read-only scan of all 48 `experiments/runs/*.json` verified every retained source
against its receipt SHA. There are 44 passing receipts, three terminal failures
and one infrastructure failure. The most recent SS-A receipt is included; no
service call was made. Counts describe artifacts, not independent observations.

| Problem | Receipts | Distinct source hashes | Sources with a passing receipt |
| --- | ---: | ---: | ---: |
| Attention | 9 | 6 | 5 |
| BF16 projections | 5 | 3 | 3 |
| Decoder | 15 | 12 | 11 |
| MoE | 19 | 13 | 12 |
| Total | 48 | 34 | 31 |

Attention, projection and MoE each include an older evaluator identity; decoder
receipts share an evaluator but still contain different component lineages and
dispatch. The full evaluator inventory/image must match, not just its code hash.
Source-equivalent diagnostic copies, check/run pairs, infrastructure recovery,
license-only changes and unchanged control shapes add no independent mechanism.

The recent native MoE interventions are KPERM, compact WGS, NW4, WAIT2, REG112 and
SS-A. WAIT2/REG112 belong to one causal branch; compact/NW4 also share the general
residency hypothesis, despite different implementations. KPERM's standalone and
decoder integration are not independent discoveries. This is an explicit small
list of interventions, **not a certified count of independent statistical
families**. Their six benchmark rows share binaries, preparations and ancestry.

Eleven Modal directories additionally retain varying subsets of requests,
manifests, worker/source hashes, device/runtime versions, paired timings,
profiles and resources. GS4 and early-A use five cases under the same paired
graph protocol; split-finalizer and dense-policy experiments use different case
sets and decoder selection. Keep these as their own evidence stratum. Do not
pool local graph timings with official SGLang-relative ratios or count graph
replays and fresh-input correctness cases as new architecture labels.

The saved complementary-shape audit found no eligible shape dispatch beating
both current comparators by its then-fixed threshold. It does not show that
future small gains are impossible. The current data is adaptively selected,
mostly negative around the native incumbent, and has almost no local density
for a novel encoding or execution architecture. A random row split would leak
the same design into training and validation.

## Small evidence record, not a new platform

For each proposed experiment, add one compact row to its existing plan/decision
record. Use existing receipts and analyzers rather than building another runner:

- **Identity:** problem; parent/candidate/diff hashes; mechanism-family and parent
  family; build flags; evaluator/image; affected and unchanged cases declared
  before results; source of any borrowed mechanism and license.
- **Known before measurement:** work/byte counts, tile and ring geometry, scratch
  layout, exact qualification/fallback rules, analytical resource necessities,
  evidence supporting the mechanism, added costs and a falsifying observation.
- **Known only after compilation/run:** actual registers, local/static storage,
  occupancy capacity, compiler warnings, selected module, correctness, timings
  and profile. Never substitute logical live words for registers or feed these
  future measurements into a purported precompile predictor.
- **Outcome:** invalid contract, unsafe model, compile/admission failure,
  correctness failure, infrastructure failure, timed result, or unknown. Failed
  setup has no latency target; missing diagnostics are unknown, not zero.

For local same-job pairs, retain `log(t_parent/t_candidate)` per paired replay
and cluster by run/input preparation. For official comparisons, retain
`log(speedup_candidate/speedup_parent)` against each exact comparator separately;
these are cross-job normalized ratios, not same-input paired timings. Preserve
raw and harness-adjusted times as separate fields. A shared parent introduces
correlation between candidate ratios. Official CI overlap neither proves equality
nor supplies a CI for the difference. Do not manufacture confidence from repeated
case rows.

Keep complete case vectors. Decoder case aggregation must follow the official
46/16 sliding/full weights within each of six loads, then the six-load geometric
mean; standalone MoE has its own six-case weights. Assess affected cases and
unchanged controls separately before aggregation. Profile-share scenarios are
planning aids, not targets or critical-path measurements, particularly with PDL.

## Selection rule usable now

Use this ordered rule on at most three concrete proposals, without a numerical
probability-of-win or an invented UCB score:

1. Exclude a violated format/math/route contract, proved lifetime/address hazard,
   or resource impossibility under stated bounds. Do not exclude merely because
   registers or speed are unknown. Static shared-memory arithmetic is necessary;
   actual compiler/device admission remains a measured gate.
2. Prefer an exact, inspectable reduction in useful work/traffic or a directly
   evidenced exposed dependency. State whether evidence is measured, inferred,
   or only a donor claim. Count offsetting unpacking, metadata, activation reads,
   barriers and tail effects. A byte saving is not a predicted latency saving.
3. Prefer a bounded change whose next measurement resolves one causal question
   at low implementation/check cost. Retain one genuinely distinct structural
   option when it has a plausible mechanism; do not award novelty points to
   renaming a failed schedule. Do not rescue closed families by automatic sweeps.
4. Declare the acceptance comparison before the run. Small gains are eligible;
   the evidence must distinguish them from controls and observed variability.
   A weak one-run result stays provisional rather than being rounded up into a
   win. No universal 3%, 5% or whole-decoder hurdle follows from the #1 ambition.

This ranks **the value of the next experiment**, not unseen architectures by
predicted speed. It cannot establish that its top choice will beat the parent.
Proof scope should follow the delta: an encoding change needs exact packed-byte
and fallback checks plus unchanged-schedule evidence; it does not need a fresh
whole-kernel asynchronous simulator. Reuse existing proofs. Larger synchronization
rewrites require the additional lifecycle work they actually introduce.

## Concrete shortlist after SS-A

| Priority | Proposed experiment | Eligibility and reason | Principal unresolved cost |
| --- | --- | --- | --- |
| 1 | **SC6 only on validated KPERM source `247888`** | Previously deferred until KPERM worked; no saved SC6 candidate/run found. Reversibly compress qualified actual E4M3 scale bytes from 8 to 6 bits, retaining original fallback. Packed weight stage falls 36864→35840 bytes, exactly 2.7778%; current KPERM permits one scale-record decode per stage. | Extra unpack/load instructions and register pressure may erase the saving. TMA global offsets, copy/expect bytes, shared stride and all fallbacks must agree. No occupancy or speed claim. |
| 2 | Warp-parallel WGS job-cost prefix, retaining current cost coefficients | Licensed donor has a concrete serial-prefix replacement. Could remove prologue work without changing GEMM traffic or math. | Current prologue critical-path attribution is weak; donor's small gain is unverified. Review exact serial dependencies first; do not combine with new cost-table fitting. |
| Closed absent new evidence | Deeper queues/register redistribution, compact two-CTA MoE, shared-A staging | Exact attempted implementations produced adverse timings; CPU-safe schedules do not reopen them. | Repeating a parameter neighbor without a new causal distinction consumes a check without resolving a new question. |

SC6's small possible margin is acceptable. The earlier SC6 note evaluated a
natural-K port with repeated decoding and explicitly deferred a KPERM-based
version. That premise has changed: KPERM is now the measured incumbent. Root
has since authorized one CPU implementation of the reviewed SC6 plan; GPU use
still requires the existing root-owned exact-source gate. Its cost is controlled by preserving the existing pipeline and testing one
encoding. No whole-file donor transplant, genetic crossover or new model service
is proposed. The shortlist is a reasoned choice, not the output of trained weights.

## When a learned ranker would become defensible

Do not make collecting a large dataset a prerequisite for the next useful kernel.
If ordinary experiments later supply enough comparable variation, fit only a
small regularized model on premeasurement features and compare it with the
no-change and shape/family-median baselines. Use leave-one-causal-family-out
validation: all descendants, public/private repeats, decoder integrations and
all cases of the held-out mechanism stay out together. Fit scaling and tune
hyperparameters inside the remaining folds. Also retain a chronological,
prospective sequence of predictions frozen before results; retrospective
family splits cannot remove adaptive search bias.

Report group-level prediction error, sign errors, top-choice regret, and
uncertainty/coverage on the held-out families, including losses. A model that
mostly predicts regressions can look accurate without identifying useful wins;
it must improve experiment selection over simple baselines. Calibrated numeric
intervals require independent calibration groups; do not claim them from the
current few families. Out-of-domain architecture/precision/geometry should
abstain or return a broad unknown, never a confident extrapolation. A residual
model over an analytic cost model is a later option, not a validated result here.

Current evidence supports none of these predictive claims yet. The immediate
deliverable is the SC6 plan and one bounded, reviewable experiment if approved,
using the existing source/receipt machinery.

Evidence read: `ROHAN_REPOSITORY_RELEASE_AUDIT_20261104.md`,
`WGS_QUEUE_CPU_SEARCH.md`, its pinned manifest/source/report,
`PUBLIC_MECHANISM_PROPOSALS.md`, `PUBLIC_KERNEL_SCOUT_20261104.md`,
`20261104-saved-shape-complements-critic.json`, actual official receipts and the
four paired Modal result/manifests cited above. No candidate was edited, model
trained, external code executed, GPU allocated or official API called.
