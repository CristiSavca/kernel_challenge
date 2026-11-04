# Research loop toward first place

The user authorized ongoing work on November 3 evening: pursue #1, use a
demanding independent critic, and consider major rewrites or different methods.
The prior goal of four measured, scored improvements is complete; the #1 goal
is active. Do not confuse successful setup of this loop with achieving #1.

## Schedule and ownership

Codex heartbeat `okc-research-and-harsh-critic` runs every 15 minutes in this
chat. Its prompt performs research and experiments, not merely status checks.
Resume existing work before starting another iteration. An in-flight official
receipt must never be submitted again. Only one local writer owns each result
file; only one project Modal GPU allocation may run at a time.

The heartbeat should surface meaningful verified improvements/rank changes,
material failures, rule/deadline changes, completion or required user actions.
Unchanged queue observations do not need a notification. After the final contest
cutoff, preserve evidence and disable the heartbeat. Do not promise #1 or cash.

## Current target and critique

At approximately November 3, 11 pm Eastern, the official decoder leader was
2.5226×; our exact scored incumbent is 2.150186×, job `4b216098993e4112`.
Passing that leader needs approximately 14.76% less normalized latency.
A 2.60× result would roughly separate scores with their minimum ±1.5%
intervals; this is a target, not a prediction. The live top-50 decoder cutoff
was 2.3442×, requiring about 8.28% less latency.

The November 4 04:56 UTC snapshot ranked us 394/1,947 overall. Overall positions 1–549 matched decoder
ordering, and an entrant ranked 21 had only a decoder submission. Official
wording describes the speedup of the best runs together, but a precise combining
formula was not exposed. Do not claim overall always equals decoder. A better
whole decoder is the best-supported practical route; winning every component
board is unnecessary. Refresh the target before allocating experiments.

Independent critic `/root/leaderboard_critic` found these weighted profiler
shares in our scored decoder (profiler sums, with overlap caveats):

| Load | Experts | Attention | Dense/fused projections |
| --- | ---: | ---: | ---: |
| Decode 1 | 30% | 8% | 57% |
| Decode 16 | 46% | 17% | 32% |
| Decode 32 | 40% | 25% | 29% |
| Verify 1 | 49% | 8% | 39% |
| Verify 16 | 55% | 13% | 27% |
| Verify 32 | 54% | 16% | 24% |

Approximate Amdahl estimates: 25% less expert time gives about 2.43× decoder;
35% gives about 2.56×. Combining 25% less expert time, 20% less attention and
10% less dense time gives about 2.62×. None is measured. Deleting all separate
routing/boundary time would yield only about 5.2%; attention alone cannot
credibly provide a robust first-place result. A single fused launch without
internal breakdown justifies profiling, not assuming no optimization remains.

## Iteration contract

1. Read current rules, deadlines, resource budget, pending jobs and profiles.
2. State one falsifiable hypothesis, affected cases, estimated upper bound,
   correctness hazards and explicit kill/pivot criterion.
3. Have an independent critic attack the hypothesis and evidence. Require
   specific defects, counterexamples and tests; harshness is rigor, not tone.
4. Implement in an isolated candidate. Major rewrites are welcome when the
   bottleneck warrants them. Preserve the exact passing incumbent and evaluator.
5. Run a bounded, paired same-GPU comparison against the incumbent on fresh
   inputs and representative route seeds. Diagnostic timing instrumentation
   belongs in a separate local experiment, never a scored candidate.
6. Discard or pivot on a failed hypothesis. For useful gains, run the full
   official private check, then score that exact frozen source and verify the
   scored result and leaderboard identity. Local ratios are not official scores.
7. Record source hashes, costs, correctness coverage, per-shape timings, critic
   verdict and response, decision and a concrete next experiment. Commit resolved
   work; leave a concise handoff for unresolved jobs.

## Initial experiments

- Split the persistent WGS expert final norm/weighted sum into a separate
  ordered kernel above 16 tokens. Test whether eliminating the final grid-spin
  barrier and register-heavy tail helps. Preserve counter generations, alternate
  scratch, every route and reset, and PDL ordering. Lower register use does not
  itself prove occupancy improves when shared memory already limits CTAs.
  Continue if expert time improves at least 10% at 128/256 tokens, without more
  than 3% loss at 32, across three routing seeds. The removable tail must be
  measured before claiming a gain ceiling.
- If dependency/tail waiting is at least 15% of expert time, test readiness-based
  scheduling of down-projection work in place of fixed A/B work ranges. Target
  at least 15% expert improvement; 30–35% is a stretch hypothesis, not evidence.
- Test existing dense choices in complete decoder graphs instead of choosing
  every backend from isolated timings. Stop after no repeatable 3% whole-decoder
  gain. This supplies a possible secondary margin rather than the primary win.

## Compute and deadlines

Modal payment verification is complete. The billing UI showed $30 included
credits; the successful single-H200 smoke used about $0.00277 in metered compute
and terminated its sandbox. Grant evidence is `modal_credit_grant.json`; the
runner's budget guard must be active before recurring benchmarks. It retains
at least $5 headroom, holds an exclusive allocation lock, reserves conservative
cost before allocation, and checks fresh billing. No out-of-pocket compute is
authorized. If the guard blocks, use official GPU checks and offline work.

Live November 3 update: kernel closes November 5 at 2:59 am Eastern; serving
now opens **November 10 at noon Eastern**, not November 5. The serving board
is private until completion, and maintainability affects judging. Verify live
rules again before serving work. The published November 19 close still needs
reconciliation with the earlier November 18 legal terms; target November 18.

Sources: [updates](https://openkernelchallenge.dev/llms.txt),
[live tracks](https://openkernelchallenge.dev/api/v1/problems),
[overall board](https://openkernelchallenge.dev/api/v1/leaderboard),
[decoder board](https://openkernelchallenge.dev/api/v1/leaderboard/muse_decode_layer),
[decoder task](https://openkernelchallenge.dev/api/v1/problems/muse_decode_layer).

## Completed first comparison: reject V8

Candidate `muse_decode_layer/candidates/native_wgs_split_final_v8.py`, SHA-256
`9e8881479fc71c985a0724ef60207cdd4d6df66fc027e8fcfafbe1eba55aa01b`, passed static review and the local comparison described below. The critic
required fresh graph replay, scratch-transition tests and confirmation that
`moe_finalize` appears in the trace; all were included. Its predecessor hash `6a8ec...` differs
only in the corrected provenance comment; executable code is identical.

Modal comparison started 2026-11-04 04:09 UTC:
`experiments/modal/20261104T040903Z-f1613663`. App
`ap-fdcwQHJrE5tWvFM50R8ovQ`; owned local exec session `74081` (if still valid).
It uses the exact passing decoder as incumbent, seven cases ordered M16 → M32
→ M16 → M17 → M128 → M256 → M8, four fresh timed replays per implementation,
and separate CUDA profiles. Sandbox timeout is 900 seconds after allocation;
the first registry-image build is separate. $2.42 was reserved conservatively.
The run completed in 595.05 seconds after allocation. All 84 local checks
passed and the actual `moe_finalize` kernel appears at M17/M32/M128/M256.
The sandbox was terminated; the app is stopped with zero tasks. Do not resume
or restart session 74081. Billing showed $0.90 total metered usage, covered by
credits, and $0 billed. The conservative $2.42 reservation remains consumed.

The independent critic rejected promotion: M32/M128/M256 whole-decoder paired
ratios were 0.987678× / 1.000261× / 0.998414× against our incumbent. Expert
profile time (including the new finalizer) changed 105.844→103.756 µs,
178.449→189.535 µs and 227.224→232.210 µs. This provides no evidence of the
required 10% expert improvement. Four paired timing samples and one profile
per case do not establish precise regressions; dense autotuning differs at
M128. One prepared weight/router seed was used. No official check or scored
submission was spent on V8. Preserve it as a rejected experiment.

Machine-readable decision: `experiments/iterations/20261104-v8-split-final.json`.

The runner guard/incumbent/profile changes passed 18 CPU tests. All four active
scored submissions still match the completed native audit. The candidate has
not been promoted or submitted. A separate diagnostic design is being developed
by `/root/decoder_research`; `OKC_WGSPROF` in the imported source is a dead
declaration, so enabling it would not measure anything. Keep `OKC_WGSFIT` off
when interpreting the existing debug timestamp slots.

## Next gate: measured scheduling and bandwidth headroom

The existing DBG A/B stamps are from warpgroup 0, without a preceding CTA
barrier. Thus DBG3→DBG4 includes both local-warpgroups and grid waiting.
Record phase-A and phase-B completion from all four consumer warpgroups with
`globaltimer`, preserving existing slots. Do not compare independent SM
`clock64` counters as global timestamps. The first proposed per-warpgroup
subtraction formula was rejected by the critic: B split-unit dependencies can
propagate another CTA's earlier waiting beyond the final A completion, so the
formula is not a global upper bound. It may be reported only as a descriptive
phase-window heuristic. Neither a low nor high value proves whether a scheduler
can recover 15%; a causal claim needs dependency and fixup intervals. Diagnose
instrumentation perturbations in both directions, not only slowdowns.

Bandwidth also constrains this decision. Active WGS packed weights occupy
exactly 27 MiB per expert; all 32 require 864 MiB, a 188.744 µs weight-only
streaming floor at H200's 4.8 TB/s peak. Actual active expert counts were not
saved in our official results. Raw scored verify-32 expert profiles are
194.6 µs sliding / 218.0 µs full. Inverting the reference's modeled floor
suggests nominal distinct-expert targets near 21, not necessarily 32; this is
an inference, not an observed histogram. Do not budget a uniform 35–40% MoE
gain as physically established.

The next diagnostic must record actual local counts c_e, U=count(c_e>0),
J=sum(ceil(c_e/64)), and padded job rows outside timing. Unique packed bytes
are 27 MiB*U; TMA requests are 27 MiB*J, not measured HBM traffic because
repeated jobs may reuse L2. Compare only raw expert timing with the conditional
weight floor, not paired-ratio-normalized `median_ns`.

The guarded diagnostic completed as
`experiments/modal/20261104T043758Z-24108d21`: all 72 checks passed across
nine cases, with identical control/probe output hashes. Three fresh standalone
route seeds produced U/J counts of 16/16–18/18 at M32, 23/24–26/26 at M128,
and 23/26–27/27 at M256. These are not decoder router distributions.
Control medians were 138–154, 198–211 and 223–240 microseconds respectively.
The probe/control median difference was +0.09–3.39% in eight cases; the first
M32 case was +12.25% and is unsuitable for precise trace claims. Event/device
span discrepancies suggest outside-kernel jitter, not proven probe cost.

The critic found no positive evidence for a 15% scheduler gain. This is not
proof that a different scheduler cannot help. Pivot to one bounded comparison
of existing dense backends inside a complete decoder graph. In the earlier V8
profile, `rq_deep_tma_pooler` was actually the unpooler despite its generic
name; both traces used the combined pooler/router path. Its 21.719 versus
21.926 microseconds does not itself identify a faster backend.

The diagnostic exited zero after 142.47 seconds, terminated its sandbox
`sb-GfLnCojLeZN5VBzPDXuvvp`, and app `ap-StGphqaF7rdObjH1OKpumg` is stopped
with zero tasks. Billing after both experiments shows $1.06 metered and $0
billed, covered by credits. Both conservative $2.42 reservations remain used.
All local old timings used a 128 MiB write flush, so the conditional HBM floors
must not be presented as verified cold-memory bounds. Future general-runner
comparisons use an explicit read sweep with observable sums, before timing.
Neither protocol measures actual candidate DRAM traffic.

### Attention V5 official check resolved

Private job `998dcd09fb9249a9`, submitted November 4 04:40 UTC, passed all
16 tests and 12 benchmarks, with no flags or rejections. Its exact frozen SHA is
`2eb08ca476d2f86cb43d2b8b1d2d630e3aa7489359acaf41cdf646ec49cd00fd`.
V5 uses native FP8 QK instructions only on full-attention q1, retaining every
key and FP16 PV. Its 2.593476× speedup (95% interval 2.554573–2.642907×)
supplies no useful gain over the parent's 2.642539× private / 2.611970× scored
results. Do not upload it again, score it, or integrate it into decoder.
The active attention remains `87a9bddc...`; active decoder remains `80a4eaa3...`.

A fresh public-source check found both original pinned upstreams unchanged.
Lee's Apache-licensed v56 at `c565e373d34575c5f7df862b05f8be7619078b0f`
has a distinct complete-output-unit atomic queue, busiest expert first, with
one-unit lookahead. That is a possible rewrite only if the measured completion
tail justifies it. Its ring interleaving and dependency prefetch are dormant;
its whole-expert readiness is coarser than our existing per-stage readiness.
Published timings are unverified. If the scheduling gate fails, pivot to
whole-decoder dense-path selection and attention improvements.

Sources: [H200 hardware specification](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html),
[Lee pinned expert source](https://github.com/lee101/openkernels/blob/c565e373d34575c5f7df862b05f8be7619078b0f/muse_moe_w4a16/submission.py),
[Lee license](https://github.com/lee101/openkernels/blob/c565e373d34575c5f7df862b05f8be7619078b0f/LICENSE).

## Attention V5 local decision

Run `experiments/modal/20261104T044835Z-19b98c8c` completed in 136.43 seconds,
passed 60/60 checks and the new read-sweep sum check, and terminated its sandbox.
The independent critic agrees to reject promotion: target B16/B32 q1 paired
ratios were 1.010804/1.001858; the unchanged q8 control was 1.045744, with
individual ratios 0.862–1.228. Small apparent target gains are not established.
Numerical error increased under FP8 rounding despite passing oracle bounds.
The official full check also passed without a useful gain, as recorded above.
No further V5 timing, scored attempt or decoder integration is justified.
Decision: `experiments/iterations/20261104-attention-v5.json`.

## Current experiment and next candidate

GS4 for WGS NTT1 (at most eight rows per job), preserving wider GS2, has static
GO from the independent critic. All 16 k16 MMAs and asynchronous operand
lifetimes are preserved; extra operand registers and arithmetic reassociation
still need GPU validation. Standalone candidate SHA is
`7f24e2857e1b419fb528e8031c7c7cbb1cb6233e9fbae46b7efdcad88c390847`;
decoder counterpart SHA is
`acfbdc2e5aead53ee4a9f8bf52a289f0942dd20a5e793de2fb39593f8cafd3f5`.

Its paired Modal run completed (started 04:56 UTC):
`experiments/modal/20261104T045617Z-6354a177`, sandbox
`sb-imlv0EDEgZloQCHtfD1Vzy`, local session `25524` is terminal; do not resume or duplicate it.
It runs M32 → M128 → M256 → M16 → M32, six fresh timing pairs, profiles,
actual routing/job counts and compiler resource metadata. First timed M32 seed
7302 was all NTT1 in the earlier diagnostic; check the actual saved histogram.
The 900-second allocation uses the common credit guard. Require useful expert
gain with mixed-width controls, no numerical failures or added resource costs
that erase the mechanism. A 15% gain is a test threshold, not a prediction.

Dense V9 has static GO for a later bounded comparison. Its SHA is
`3d5b34620a05faa628c30077728328a28e362b4c9dfc54dbb9ee6d5f6dd4d58c`,
file `muse_decode_layer/candidates/native_dense_chain_select_v9.py`.
It chooses existing dense backends in a synthetic post-attention decoder chain
at prepare time, preserving all runtime code. The selector budgets are soft:
require actual prepare time below 300 seconds, a logged changed dispatch, and
at least 3% held-out whole-decoder gain. Do not mistake a budget skip or no
accepted choice for an exercised optimization. Root owns GPU scheduling.

The GS4 comparison passed 80/80 checks but is rejected for promotion. Paired
ratios at M32/M128/M256/M16/M32-return were 0.981728, 0.967743, 0.952977,
1.002119 and 0.991159. Actual seed 7302 had 16/16 NTT1 jobs; its single pair
was only 156.960 versus 153.632 microseconds, not the required material gain.
Both main binaries report REG96/STACK0/LOCAL0; this does not establish a
particular cause for the slowdown or absence of WGMMA serialization. Runtime
was 379.77 seconds; prepare took 187.12/185.19 seconds including weight clones.
The sandbox terminated and the app is stopped. No official attempt is warranted.

Dense V9 completed at 05:15 UTC in
`experiments/modal/20261104T050629Z-62c33804`; local session `11768` is terminal.
All 144 checks passed, and the sandbox terminated. The four affected sliding
cases had a paired geometric mean of 1.000152720×, below the 1.03 gate.
Independent review confirms all 12 selector coordinates ran: 38 alternatives,
31 timed, seven conservative route/numeric rejections, zero accepted changes.
The best synthetic three-seed geometric mean was about 1.00147. This is a
completed null experiment, not a skipped or unexercised selector. The cold
parent preparation (494.43 seconds) versus warm candidate (11.64 seconds)
does not validate cold prepare compliance. Do not upload or retest V9.
Decision: `experiments/iterations/20261104-dense-chain-v9.json`.
Fresh billing after the run showed $2.60 covered by credits, $0 billed; the
app was stopped with zero tasks. The journal reservation remains consumed.

The now-completed, rejected experiment was whole-unit phase-A
publication, using the exact original incumbents (no GS4/dense/V5 mixing):
`muse_moe_w4a16/candidates/native_wgs_early_a_publish_v8.py`, SHA
`2931be209c1ef902158883b1e1aceeb5a34f5ac72f15bb525c13d58643e7a488`.
Its decoder counterpart SHA is
`102c0d1509d39bb44d3884ea41ea9eef9624bf6465527eeee7d5bcec9ef3eb3a`.
The early-A GPU screen is terminal; see its rejection below. Do not rerun or
start a CI-index sweep.

User asked how submissions appear on the board. Verified `okc whoami` is
CristiSavca, and the actual public browser row is #395 on decoder at 2.15×,
job `4b216098993e4112`. Their browser had shown the projection board's first ten
rows. Switched to decoder, expanded through 410, and left their row visible.
Screenshot: `experiments/leaderboards/cristisavca-visible-20261104.jpg`.
This was a status question, not cancellation of ongoing research.

Fresh 05:22 UTC boards: CristiSavca remains #395 overall and on decoder.
The new decoder leader is gauravjain14 at 2.5401× (job `bd4cc135d74d43c6`),
with screened speedup 2.50984561×; use the published leaderboard score for
standing. Our 2.150186× entry needs approximately 15.35% lower normalized
latency to equal 2.5401×, before allowance for uncertainty. Snapshots saved
in `experiments/leaderboards/20261104T052250Z-overall.json` and
`experiments/leaderboards/20261104T052251Z-decoder.json`.
The previous goal turn made progress: completed dense evidence changed the
next action and verified the user's public leaderboard identity.

Early-A screen completed:
`experiments/modal/20261104T052256Z-4c806134`, sandbox
`sb-eoBGNWklvCQepCwpqWyEM8`, app `ap-8V65gdkp3rQmiGPkRp7EFQ`.
Local session `34776` is terminal; do not resume or duplicate it. The exact
standalone candidate `2931be20...` versus frozen `ef7b210b...` passed 80/80
checks with the read-sweep selfcheck, exited zero, and terminated its sandbox.
Paired ratios M32/M128/M256/M16/M32-return were 1.001020, 0.986994,
0.993350, 1.002000 and 0.998470. There is no 3% gain and no basis for a
second confirmation, decoder integration, or official submission. Independent
critic confirmed rejection in `experiments/iterations/20261104-moe-early-a.json`.
Preparation took 208.97/199.14 seconds; all main/other compiler resource tuples
match. The final M32 incumbent profile was unavailable, so its candidate
profile has no paired profile counterpart. Total worker elapsed was
415.62 seconds. Fresh billing is $3.18 metered, all credited, $0 billed;
the app is stopped with zero tasks. Conservative reservations remain consumed.

Next actionable work (no GPU currently allocated):
- `/root/decoder_research` is implementing a bounded hardware-profile mode
  in `tools/modal_h200.py`, preserving the credit/overlap guard and upload
  whitelist. Exact standalone incumbent, M32/M256, fixed questions about DRAM
  traffic and issue stalls. NCU exists in the image; uid 0 and
  `RmProfilingAdminOnly: 0` were observed read-only, but counters have not yet
  been successfully collected. Require independent review and focused tests
  before root allocates. Profiler durations are never A/B speedup evidence.
- `/root/attention_research_v2` is preparing an isolated structural q8-full
  prototype: one QK/softmax warpgroup and one PV warpgroup, with consumed K
  slot bytes reused for P and explicit per-head ownership. No extra large P
  buffer is planned. Exact layout/barrier/lifetime proof and critic review are
  required; GPU correctness and performance are unproven. Parent runtime
  outside q8-full must remain intact. Root authorized implementation, not
  allocation or promotion.
- `tools/NEXT_DECODER_STRUCTURE_PIVOT.md` records why already-fused small-token
  dense paths and an unmeasured shared-memory dequant rewrite are not the
  immediate next experiment. Fresh rules have no schedule change.

All four active scored sources are unchanged. Current overall/decoder standing
remains the last verified #395 at 2.1502×; no new public score was claimed.

Continuation at 05:34–05:40 UTC revalidated all Modal apps stopped; no GPU
allocation is running. Provider billing revised its aggregate to $3.12
(resource breakdown $3.12322333), still $0 billed; prior $3.18 snapshot is
preserved as observed, and reservations are not refunded. A fresh 05:35
leaderboard snapshot still places CristiSavca #395, leader 2.5401× and top-50
cutoff 2.3493×. The previous turn was progress: early-A completed and was
rejected from independently checked paired evidence.

NCU helper `tools/modal_hardware_metrics.py` is under active author review;
root found an EOF/normal-process-exit race before allocation, which the author
is fixing with a subprocess regression test. Intended profile seeds are 7302
at M32 and 9302 at M256, weight seed 7311; exactly two controlled-cold kernel
captures (possibly multiple replay passes each), no source instrumentation. The eventual runtime must validate metrics,
complete launch records, units and actual pass counts. No hardware counter run
has happened yet. Attention's exact decoder build overrides PK2=0, so its
prototype must preserve per-K32-half rescale ordering instead of macro defaults.

Hardware-counter run started 05:46 UTC:
`experiments/modal/20261104T054636Z-ea14816c`, sandbox
`sb-RanXWKIxQCFs7pCilXezr6`, app `ap-kuUj734njDBhlU0tspu85c`, local session
`83063`. It is running; root owns polling, billing and allocation. Resume the
same session, never a duplicate. Exact runner `105c5ace...`, helper `6e9ac592...`
passed root/author CPU suites and independent critic static review. The new
suite has 51 test executions including inherited tests, not 51 unique checks.
Command: hardware-profile --hardware-issue --timeout 900. One exact ef7
prepare, two profiled sizes M32/M256, seeds 7302/9302, weight 7311. Policy:
controlled-cold kernel replay, no clock changes, at most two matching launches,
no profiler retry, 860-second subprocess budget inside the 900-second sandbox.
The guard reserved $2.42, leaving $4.93677667 in its conservative working
budget. Actual counter availability remains a runtime gate.

Attention prototype `native_q8_role_pipeline_v10.py` and adjacent CUH/CPU
verification files now exist but are under active static review and may change.
Do not freeze or test the initial candidate hash. Its exact-layout and modeled
schedule checks passed; actual compiler resources and two-CTA occupancy still
need a reviewed GPU verification path. Source runtime outside the q8-full path
must remain intact. Root has requested explicit resource verification rather
than assuming it from the CPU budgets.

Hardware run is now terminal; local session `83063` exited zero and sandbox
termination is confirmed. Do not resume or allocate a repeat. NCU 2025.3.1
accepted all 11 requested metrics and captured exactly two kernels, four replay
passes each. The child completed all 20 oracle checks, profiled seeds 7302/9302,
and one 203.35-second prepare. Total diagnostic elapsed was 223.45 seconds.
The original report correctly remains `hardware_unavailable` because our parser
expected long CSV, while NCU emitted wide CSV (280 columns, units row, two data
rows). Raw capture is intact. Author is adding strict wide-format parsing and
saving an offline recovery separately; never rewrite the original result,
manifest, or frozen helper, and no GPU retry is justified.

Independent raw audit: M32 U=J=16, read 456,964,352 B/write 3,375,872 B;
M256 U26/J27, read 769,691,136 B/write 4,264,192 B. These reads are 1.0088×
and 1.0069× the respective nominal repeated-job packed bytes. NCU DRAM
throughput is 74.43/75.06%, issue 59.38/59.48%, eligible warps 1.40/1.39,
tensor activity 11.22/23.71%. Its 128.576/214.304 us durations are controlled
replay diagnostics, never benchmark speedup. NCU-injected ordinary medians
479.632/483.344 us differ conspicuously from normal runs and are unusable as
independent performance evidence. At nominal 4.8 TB/s the same observed traffic
has only about 25.4/24.8% conditional headroom; this neither localizes stalls
nor establishes removable wall time. Prioritize the concrete attention test.

Final attention core source `6eb318df58c3ef5576c4ea456b4b494e1ef4548090fa7707c84e247f960c2222`
has exact-byte critic static GO. Root independently reran layout/schedule/
rescale proofs. The final guard encodes NP128, SETMAXNREG, MERGE3, !QDIRECT.
Author is preparing an isolated attention wrapper with exact decoder native
source/flags and a host-only resource export, avoiding cold whole-decoder
compilation for the first screen. It is not the active standalone87a9 source;
parent/wrapper equivalence and actual occupancy require review before allocation.

Continuation at 05:57–06:00 UTC: user asked why their name was not visible.
Fresh authenticated CLI identifies CristiSavca, overall/decoder rank396,
decoder2.1502x; all four official scored entries remain public. The user's
Chrome homepage showed top-ten attention rows only. Root opened the full
leaderboard, selected decoder, expanded410 rows, and left CristiSavca396
visible. Proof: experiments/leaderboards/20261104-public-entry-verification.json
and cristisavca-visible-20261104-rank396.jpg. No new score was submitted.

NCU recovery is complete and independently approved. Final recovered report
SHA0921485edb8a6494302288956fe1ef3614e7980f5fbf84fd831f27977a4c36b7;
working helper ba26f5f43d15ba38e57f01766e85f532aa8fd659fab6e0acfe0d2cc28d1e2ec9.
Root reran53 parser/lifecycle test executions and independently matched each
recovered launch to raw fixture and all original artifact hashes. Original
failed-parser status remains unchanged; no rerun. Billing after NCU: aggregate
$3.47/resource$3.47495600, credits cover all, $0 billed, app stopped0tasks.

Exact decoder-attention wrappers dc6d0c48/d8be9200 and host resource hook
have independent static GO. Agent attention_research_v2 now owns a narrow
runner integration and tests; root alone owns allocation. Require actual
complete ptxas output, post-warmup exact-specialization resource attributes,
localBytes0 and2CTA occupancy, actual flat/cooperative plan, correctness and
paired graph timings. No GPU currently running.

Goal continuation06:00UTC: previous turn was progress (committed offline NCU
recovery/parser and proved public submissions); no blocker. Fresh Modal CLI
confirms every app stopped0tasks and billing unchanged$3.47/$0billed. No pending
GPU session. Official quotas still decoder35private/19scored, attention38/18.
Upstream head refresh06:02 confirms Auto-research018b7a1 and vjeuxf35a41ee
unchanged. Root and critic reviewing fixed attention-v10-screen integration
owned by attention_research_v2; root requested pre-launch compile checkpoints
and timeout artifact recovery. No allocation authorized until finalstaticreview.
Decoder agent researches one substantive next hypothesis offline, using actual
profiles and75%DRAM evidence, not another blindparameter sweep.

Attention V10 screen started06:08UTC, after exact static GO and commit05f263b.
Live local session67460, sandboxsb-3mm1lWhj1XrvQpUuX74GW9,
appap-XWVq5AL34dUXQKHXc1bv8a; root owns polling and allocation. Evidence
experiments/modal/20261104T060813Z-37b2e369. Resume same session; no duplicate.
Command attention-v10-screen --timeout900. Runner826e134d/helpercda695f4.
Guard reserved$2.42 afterprior$16.94; observedbilling revisedto$3.44,
workingbudgetremaining$2.20. Original$3.47snapshot preserved. Expected128checks
across8cases; exact decoderparentwrapper comparison, not standaloneincumbent.
Compiler logs checkpointbeforeGPUlaunch; actual2CTA/local0/activeflatplan gates.
No numericalorperformance result yet.

Attention V10 terminal at06:10UTC: session67460 exited1; sandboxtermination
confirmed, app stopped0tasks. Bothcompilecapturescomplete(~58.6s each). Parent
exact<8,true,0> haslocalBytes0; candidate haslocalBytes296 and explicitC7512
WGMMAserialization dueinsufficientregisterresources. ptxasstatic296Bstack,
1128Bspillstores/864Bspillloads. Both80initialregs/2CTA/114688shared.
Onebaselineeagercheckpassed; zero candidateoraclechecks andzero timings.
Firsttargetactualplanseligible264x1x1, butresourcegate correctlystopped.
RejectV10ascompiled; no performance/numericalclaim, noofficialupload. Logs
andresultwere normallyretrieved despite secondary recoveryNotFoundafterappstop.
Elapsed120.9478s. Billingafter$3.61/resources$3.61343377, allcredits,$0billed.
Decision experiments/iterations/20261104-attention-v10-resource-rejection.json.

Root authorized NEW V11candidateonly (noGPUyet), ownedbyattention_research_v2:
compile-timeROLE3QK/ROLE4PV throughrun_segments→segment→body so96/112
allocations dominate disjointspecializedcalls ratherthan reconvergingintoROLE2.
PreserveV10bytes,96/112/32 budgets,math/storage/grid,256consumerbarriers and
parentfallbackROLE2. Ringinit must includeROLE3/thread0. This isoneconcrete
allocator-visibility hypothesis, notknowncause; criticreviewrequired beforetest.
Do notraisebudgetsblindly orrelaxgates. Currentguardwillnotfit900seconds;
600seconds reserves$1.78 andmayfitcurrent~$2.03 budgetafterfreshpreflight.
Decoderagent falsified11bitdensecodec aslargeranklever (already12bit) and
is nowreading a newApache-licensed publicsourcelead keell0renz pinned890809c,
reportedfair2.2532—notitsdisclaimedjudge-max#1. Noexecution/integrationyet.

V11 compiled-role revision frozen core7b57f0f8/wrapper6724284b, independently
reviewed; rootfullCPUverifier and85runner testexecutions pass. Runner60cbfa93
adds only fixedrevisionselector/CLI/metadata; sharedgates/timing/checks/lifecycle
unchanged. Commitce4f3e5. Screenstarted06:22UTC with600secondlimit:
experiments/modal/20261104T062254Z-94807c25, session24197(rootonlypoll),
sandboxsb-FiVd9sx1Enyms0uWAdezk2, appap-tSHquo88CT3QHir7H28Tci.
RUNNING, resumesamehandle, neverduplicate. Guardfreshspend3.73252457,prior
reservations19.36,new1.78,workingbudgetremaining0.12747543. Do notreset
reservations/relaxguard; laterGPUwork shoulduseofficialprivatechecks when
localguardblocks. Expected128checks/noresultsatlaunch.

V11terminal: session24197exit1, sandboxterminatedtrue/allappsstopped. Exact
changedtemplate noC7512/serialization in completecompilerlogs; local/stack88B
andstaticspillstores132B/loads192B remain. Originalzero-localgate correctly
rejectsV11. Parentzero, both2CTA/80initialregs/114688dynamicshared. Elapsed
117.411934s; compile55.896/58.050s. Onebaselinecheckonly,0candidatechecks/0timings.
ResultSHAb21649ba1457c72f7dd65c9c42088ff472c0d1732256c17693097a44c3dc4bed.
Freshbilling$3.90/resources$3.90423640,$0billed. Conservativejournalreserved
21.14USDnowexhaustsworkingbudget; donotreset/refund/recycleorrelaxguard.

Independentcritic approved ONESEPARATEofficialprivatecheck of exactuninstrumented
V11wrapper6724284b. Rationale: zero-local wasperformanceheuristic, notcorrectness
invariant; forcedWGMMAserializationnowremoved, occupancyretained, staticspills
alonedonotdecidenetruntime. No localgatechangeorretroactivepass; preservefailure.
Officialjob88f84f79fb11462d submitted06:30UTC, authoritativequeued/checkat06:32.
Rootpollswithtools/experiments.py status88f84f79fb11462d; no duplicateupload.
Compareq8cases separately anddiscloseparentexactdecoderattentionvsactive87a9
standaloneflags. No candidatecorrectness/speedresult, nopromotionyet.

Newlicensedpublicsource scout is archived underthird_party/public_scout/
keell0renz-890809c withApacheLICENSE/NOTICE and1005920verifiedbytes. Fair
2.2532scoreis authorreported,no rawreceipt; doNOTcallit#1code. Sourcehasunsafe
MFINtimeoutfallthrough andmanyconfounds; no wholefilecopy/execution.
MFIN-only tailpotential ismodest~1%aggregate underconditional fixed-completion
accounting, plus64Jnewatomics. SC6only saves2.78%weightbytes butdonorSC6also
usesKPERM3, soisolatedSC6cannotinheritpublishedgain.
KPERM3 isnewactionablelead: proved256elementKbijection,256fragmentterms,
pairedgate/uprowpermutationwithin4096, preserved256-columnreadinessstages;
scale-blockindexindependentofloadgroup permitsreuse. Fewerload/decodeinstructions
not4xlessuniquesharedtraffic. AddedgatherinstructionswithequalcoalescedHBMbytes.
Rootauthorizeddecoder_research toimplementnewstandaloneef7KPERM-onlycandidate
AFTERsavingproofandcriticplanreview. PreserveSC6off/FP32ybuf/FCI3/joinedpipeline/
jobcosts/allterms; provecorrectnaturalpackedweights fornonWGStiny/MIDfallbacks,
activationspermutedinsidetimedkernels only. NoGPU/officialsubmitbyauthor.
Officialprivatequotasremainavailable; #1goalactive andnotachieved.

06:34UTC authoritative refresh: overall/decoder396,2.1502x; leader2.5401x, top50cutoff2.3493x. Privatejob88f84f79fb11462d remainsqueued/check, notmissingorfailed; pollsamejoblater, neverresubmit. AllModalappsconfirmedstopped; noactiveGPU/localprocess. Latestcommittedresearch7e8b474. Thisgoalturnmadeconcreteprogress(tworesourceexperiments, astructuralrevision, separatelyapprovedofficialcheck, newlicensedKPERMlead); goalnotcompleteandnoblockingcondition.
