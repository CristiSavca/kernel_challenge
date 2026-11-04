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

At November 4, 12:17 UTC, the official decoder leader was 2.5510×. Our exact
scored incumbent is 2.181183×, job `7a8c4941741645ee`, source `97af3994…`;
decoder and overall ranks were392. Passing that observed leader needs approximately
14.50% less normalized latency. This remains an unachieved goal, not a forecast.
Standalone MoE is 2.373691×, job `d24dd50196cb4158`, source `247888ac…`, rank292.
Both active submission.py files equal their scored snapshots. The preceding
80a4/ef7 files remain frozen; private experiments never replace these incumbents
without a useful verified result. See the dated live snapshot in
`experiments/leaderboards/20261104T121714Z-r16-terminal.json`.

The November 4 04:56 UTC snapshot ranked us 394/1,947 overall. Overall positions 1–549 matched decoder
ordering, and an entrant ranked 21 had only a decoder submission. Official
wording describes the speedup of the best runs together, but a precise combining
formula was not exposed. Do not claim overall always equals decoder. A better
whole decoder is the best-supported practical route; winning every component
board is unnecessary. Refresh the target before allocating experiments.

Independent critic `/root/leaderboard_critic` found these weighted profiler
shares in the preceding 80a4 scored decoder (profiler sums, with overlap caveats):

| Load | Experts | Attention | Dense/fused projections |
| --- | ---: | ---: | ---: |
| Decode 1 | 30% | 8% | 57% |
| Decode 16 | 46% | 17% | 32% |
| Decode 32 | 40% | 25% | 29% |
| Verify 1 | 49% | 8% | 39% |
| Verify 16 | 55% | 13% | 27% |
| Verify 32 | 54% | 16% | 24% |

Historical Amdahl estimates from that 2.150186× parent: 25% less expert time gives about 2.43× decoder;
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

06:41 UTC follow-up: authenticated CLI again verifies CristiSavca and four public
scored rows. Overall/decoder rank396 (2.1502x), attention278 (2.6120x), MoE341
(2.3561x), projections357 (1.4868x). Exact public job IDs remain unchanged.
Snapshot: experiments/leaderboards/20261104T064143-public-entries.json. User was
again told explicitly that private checks do not publish and official submit does.
Latest V11 status at06:39 remained queued/check; no duplicate submitted.

KPERM3 plan has independent critic GO. decoder_research is implementing
muse_moe_w4a16/candidates/native_wgs_kperm3_v9.py and an actual packed-byte
CPU inverse proof for both stored layouts. Source is in progress, not frozen or
uploaded. Root inspected the focused diff and requested current packing docs
and self-contained upstream NOTICE comments. Critic reactivated for exact-source
review and one private-check recommendation after the author finishes. Root alone
submits. Preserve unchanged encodings0..11, tiny/MID/fallback dispatch, WGS
barriers/GS2/FCI3/NTT1 and math precision. No new Modal allocation. Goal active.

06:48 UTC concrete progress: KPERM-only source247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0
is frozen and committeddca53c2 after root+critic full actual-byte inverse proof
(100663296codes/6291456scale bytes, both stored layouts,14161scale pairs) and
static review. Root submitted ONE official private MoE jobbcb761dfca264db0
at06:47UTC; subsequently confirmed queued/check. Resume that job, neverduplicate.
Local submission session83767 is terminalexit0. CPU environment bounded at
/tmp/okc-kperm-cpu-20261104 (NumPy2.5.4), rootproofsession92666 terminalexit0.
No local GPU allocation. Exact source/snapshot/receipt retained.

Attention V11 official88f84f79fb11462d is DONE/PASS with16tests,12benchmarks,
12profiles, no flags/rejections. Nonetheless reject this exact version: speedup
2.2708996 vs active2.6119701; target B16/B32 fullQ8 normalized151.304/276.706us
vs100.491/183.120us (~50% slower). Profiles confirmattn_kernel<8,true,0>.
Independentcritic agrees no score/no decoderintegration. Differentdecoderflags
andfreshseeds prevent isolated attribution to ownership/spills; resource screens
remainfailed. Rejection andpercaseevidence:
experiments/iterations/20261104-attention-v11-official-rejection.json.
Attentionagent saves precise officialcoverage/flags checklist. Decoderagent now
looks for one distinctdense/routingmechanism in already-preserved licensed source,
no variants/GPU. Activepublicentriesunchanged; #1goal remainsunachieved.

07:16 UTC research reset: latest primary papers through Nov3 reviewed; see
 tools/RESEARCH_RESET_20261104.md and tools/ATTENTION_ARXIV_RESET_20261104.md.
KPERM standalonebcb761dfca264db0 DONE/PASS5tests+6benchmarks/no flags,
2.38983596x versus scored2.35610395x. Aggregate intervals overlap. Critic rejects
immediate score but permits one isolated decoder check based on changed-size
signal; root+critic actualdecoder packing proof passed50,331,648codes/3,145,728
scale bytes. Candidate97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2
was submitted as private jobb2a773e2d7194de8. Local submission session85639
terminalexit0, received status atsubmission. Rootalonepolls samejob; no duplicate.
Currentoverall/decoder398,2.1502x; leader2.5401x. Allpublicsources unchanged.
No Modalallocation; conservativebudget remains exhausted, no budgetreset.
Previousgoalturnwasprogress: latestresearch, completedKPERMresult and exact
newdecoderintegration/test. BroadcompactWGS twoCTA architectureplan is in
independentreview, and smalltoken dense research continues. No#1claim.

07:29UTC: isolateddecoderKPERM b2a773e2d7194de8 DONE/PASS13tests+12benchmarks,
12profiles,no flags; prepare114.52s, score2.17665445x CI2.14400463–2.20930427.
Public80a4score2.15018610 remainsunchanged. Wholegain~1.23%overlapsintervals;
M1unchangedcontrolalsoimproves~2.6%load, changedM32/M128loads~2.65/2.20%,
M256~0.11% againstpublicparent. Criticconductingpercaseaudit; no scoreyet.
Newresearchderiveddensecandidateb0387be327a480c84227985566bf086c70d894394f95ad9b0589144c902e260b
passedroot+critic CPUaddress/order/ringmodels andexactsourceaudit. It retainsall
escape warps/FP32planes/geometry anddecouplestheirinputreadsfromTMA ring.
RootsubmittedONEprivate decoderjobd4c26ebb65e94a84, local60776terminalexit0;
receivedatsubmission. Resumeexactreceipt, no duplicate. Publicsourceunchanged.
CompactWGSauthorisstillimplementing/provingnew320thread128row/K128source,
notyetfrozenoruploaded. Rootcorrectedcritic'sPDLadmissionhazardagainstNVIDIA
primarydocs: launch_dependents already requiresallparentCTAs tosignal/finish.
Initialfindingwithdrawnwithrecord; preserveparenttriggerplacement. Confirmed
64bitpartitionoverflowfixremainsneeded. NoGPU/Modal allocationauthorizedtoagents.

07:37 UTC: compact MoE cf09ba584cc1537802636bb905ac08cda2095f83a163358e1c9bb716490130b8
has final independent GO and was submitted as private job94e63981c631474e.
Submission session9612 is terminal exit0; receipt says received. Root alone
polls this job and dense jobd4c26ebb65e94a84 (latest queued). No duplicate uploads.
Root, author and critic independently passed the complete compact CPU suite:
100663296 codes,6291456 scale bytes,1179648 activation coordinates,64296960
actual extracted C++ stages,16 exact fallback builds. A second reviewer found
no targeted source blocker and quantified changed FP32 norm association.
Final cf09 differs from tested d27 only in corrected provenance comments;
root and critic reconstructed the prior SHA and verified complete AST identity.
Actual GPU occupancy>=2 is enforced before the enlarged-grid spin barrier;
compiler spills/serialization, numerical accuracy and timings remain unverified.
Do not score/integrate until the official result is reviewed. Decoder KPERM
b2a773e2d7194de8 result critique is saved; HOLD score pending broader tests.
No Modal allocation or budget reset. Public entries unchanged; #1 not achieved.

07:57 UTC: both broad research candidates are terminal PASS but performance failures.
Dense d4c26ebb65e94a84:13 tests/12 benchmarks,2.087688x versus2.150186x public;
reject/no score. Four changed SK2 loads regress3.52%; no-SK2 controls regress1.67%.
Backend selections also changed, so do not attribute the whole loss to global reads.
Compact94e63981c631474e:5 tests/6 benchmarks,2.047216x versus2.356104x public;
reject/no score/no integration. Actual host attributes320 threads,98048 dynamic,
occupancy capacity2/2,96 regs and64 local bytes per thread. Affected sizes show
~34.6% higher SGLang-normalized latency; controls remain close. No hardware
counters isolate stage overhead, activation requests, final reads or local memory.
Both exact sources and terminal profiles/rejection audits are retained.

Independent critic resolved the temporary KPERM HOLD after broader failures:
GO for ONE scored publication per already-tested source, with overlapping
intervals and no precise gain claim. Root submitted standalone247888 as
scored jobd24dd50196cb4158 and decoder97af as scored job7a8c4941741645ee.
Both receipts are received; decoder local session91366 is terminal exit0.
Resume these exact jobs, no duplicate upload or timing-seeking retry. Do not
replace incumbent files until scored receipts and leaderboard are reviewed.
Fresh overall rank401 of1957; leadergauravjain14; no #1 claim. No Modal allocation.

FT128 follow-up is rejected before implementation based on weak observed reuse
and extra stage/register resources. A separate dense stage-directory idea is in
plan review: retain exact80a4 parent ring/shared-input corrections and use tiny
immutable per-KST offsets to skip repeated escape-index scans. Existing table
sharing includesWSK/UPW, makingKST invariant whileNESC can differ. Author may
implement only after critic accepts exact compatibility and a credible static
cost rationale; no new GPU/official check authorized to agents. A plan is not a
measured gain. Goal remains active; this turn made concrete research, code,
validation, rejection and public-submission progress, with no blocking condition.

08:25 UTC: both KPERM public submissions are terminal PASS and active files promoted.
Standalone d24dd50196cb4158, source247888, score2.3736910444 (previous2.356104),
all5tests/6benchmarks; live MoE rank288 (was341). Decoder7a8c4941741645ee,
source97af, score2.1811831350 (previous2.150186), all13tests/12benchmarks;
live overall/decoder rank381 (was401), leader2.5401. Exact live snapshot:
experiments/leaderboards/20261104T082347Z-kperm-public-results.json.
Both active submission.py files now equal their exact scored source snapshots;
prior80a4 andef7 remain frozen. No repeat of these completedjobs.
Aggregate intervals still overlap; improvement in officialbest score/rank is
observed, not proof of a precise causal speedup. MoEM128 absolute latency is
slightlyslower despite improved relative ratio due baseline movement.

Directory15a7188b exact-source review and rootCPU suite passed; root submitted
ONEprivate whole-decoder check4b9f33ffa6bc4eed. Local session26671 terminalexit0;
latestobservedqueued. Keep polling samejob, no duplicate. Preserve preparation
SK2_STAGE_DIRECTORY logs to prove actual selection, not just renamedkernel.
This candidate remainsisolatedonold80a4, not new97af. Ifuseful, integrationinto
newincumbent needs separate exactscopevalidation. CPUfitguard now reserves2048B
for staticshared/alignment after rootcaught omittedUV arrays+s_rlast.

User challenged pace and rankslippage. Root explained the priorpublicscore had
stayedunchanged whileothers passed, then verifiednewrank381. Goalstill#1,
roughly14.1%normalizeddecoderlatencycut needed; don'tredefinesuccessassmallwins.
New higher-impact research targets tiny/MID MoE paths affectinghalfthescored
loads. Grid-only2CTA is disproved by savedactual144–149register binaries and
oversizedsharedallocation. Author examinesNW4/NS2 plus right-sized metadata,
correctscaleloadcoverage andadmissiongates; no newsource/GPUapprovedyet. SameK
full weightunit36KiB andthree slots108KiB meansmerelyhalvingNWcannotfit2CTAs.
Need explicitarchitectureandcriticreview, not silentlyweakenresourceguards.
NVFP4 S/3S/LUTaudit isclosed asunsupportedcost tradeoff;4,064exactfiniteproducts
wereverified butno distinct fastermechanism identified. No newModalallocation.
Upstreamrohan018b/vjeuxf35aheads reconfirmedunchanged08:03UTC; free disk35GiB.
Previousgoalturnwasprogress, thisturnhas scoredresults,promotions andnewprivate
check; goalactive, noblockingcondition. Rootalonepolls/submits.

08:43 UTC: dense stage-directory check4b9f33ffa6bc4eed is terminal PASS,
13tests/12benchmarks, score2.149250315, noflags. Reject promotion: -0.044%
versus its exact80a4 public parent and -0.226% private. Preparation logs prove
four of nine immutable tables selected the directory; profiles retain parent
UV/pooler-router fusion. Targeted eight cases show +0.884% public/+0.098% private
geometric-mean ratios, with mixed individual kernel changes. This supplies no
useful whole-decoder gain. Preserve the terminal audit and all profiles; do not
retry, combine with KPERM or score this source. Current97af remains incumbent.

Live refresh08:41 UTC shows overall/decoder rank382, unchanged2.1812; MoE288,
unchanged2.3737. Snapshot20261104T084126Z-public-incumbent-refresh.json. Rank
slipped by one while our score stayed fixed. No #1 claim or completed goal.

Small-MoE NW4/NS2 plan passed independent review for one isolated implementation
from247888. Source is written; actual-source CPU operand checks and independent
ring-lifecycle review are in progress. Root read scale-group accumulators,
route/helper/finalizer/resource guards. Two occupancy queries must succeed at
capacity>=2 before enlarged ordinary or PDL launches. No GPU run yet, no sweep,
no changes to active submissions. Free disk36GiB; official MoEchecks36 and
decoderchecks32 remaining. No new Modal allocation or budget reset.

08:51 UTC: small candidate34ee99c242c4ffb5028ec3511fd8740c5f15461e1c73e9c1343ae6611573e3e3
passed author, root and independent critic actual-source CPU checks:23,592,960
MMA operands,18 old-build preprocessing controls, complete packed coordinates.
Root also reran the independent22-schedule/372,072-event ring model, including
two negative controls. One verifier raw-brace parser bug was repaired before
the successful runs; it was not a CUDA candidate failure. Final critic GO is
limited to one private measurement. Root submitted f46b93032b5548c1, nowqueued;
local submission completed exit0. Resume this exact receipt, no duplicate.
Actual GPU registers/local memory, occupancy, numeric error and speed are pending.
Both active public files remain97af/247888. Decoder integration, if justified,
must preserve its TINY FINPF0 flag; standalone TINY has FINPF1.

Independent next-dense/attention scope analysis found no supported >=5% whole
gain hypothesis: current profile scenarios need18.86% across all small dense
rows or33.76% across all attention. These are scenarios, not physical bounds.
New MAIN MoE feasibility research keeps256-row/K256 units and asks whether
smaller token tiles/right-sized metadata/NS2/separate finalization can fit two
576-thread CTAs without compact V10's stage/activation costs. Existing96regs
would need<=56; this is an unresolved major obstacle, not a proposed speedup.
No implementation or GPU approved for that feasibility question. Goal active.

09:06 UTC: NW4 check f46b93032b5548c1 is terminal PASS but REJECT for performance.
Exact 34ee source scored 2.258751600, below both 247888 parent intervals. All five
tests, six benchmarks and six profiles passed without flags. All three small
extension markers show 160 threads, 101632 dynamic bytes, occupancy 2/2,
168 registers and 32 local bytes. Affected M1/M8/M16 baseline-relative latency
rises 23.55/7.06/2.48% versus public, geometric 10.67%; the private comparison is
geometric 11.16%. MAIN controls remain close.
Root analyzer and independent critic agree: no integration, scoring or retry.
The capacity hypothesis succeeded; higher capacity failed to improve speed.
Local bytes alone do not establish spills or their causal contribution.

The broad MAIN two-CTA feasibility is closed NO GO. FT16 adds 44–85% weight
requests in saved M256 routes; FT32 still adds 11–27%. Dropping S.pos alone does
not make all larger shapes fit. Existing 160 fixup slots cannot handle grid 264.
No register evidence supports 96→56 from splitting the finalizer; no overflow claim
is justified under reviewed bounds. Exact layouts/route counts are retained.

NVIDIA's pinned CUDA 13.0.96 occupancy header supplied a useful correction:
576-thread kernels admit at most 96 reported registers, not the 112 suggested
by aggregate arithmetic. 160 threads at 168 registers permit two CTAs;
200 registers permit only one.
The CPU calculator is a register-capacity model, not an actual GPU query.

One distinct WGS wait2 mechanism is now under official private check:
a0dd88ebc8964b3f, source 97ffccc30a90ce8400cddef800d932889822eaf876fbdedaec18b1013e8f66b6.
It keeps one CTA, 576 threads, all geometry/bytes/math/KPERM, and uses four AF
banks with wait2, preserving NTT1 parity and moving previous-X release to ci1.
Root/author checks passed 16 unchanged non-WGS builds and 1,462,272 native MMA
trace calls; the critic approved one private check only. The abstract lifetime
model passed 13,872 cases and three unsafe controls. The 96-register budget has
no established headroom; compiler spills, serialization and speed remain unknown.
WGS build diagnostics and both specialization resource markers are enabled.
Local submit 41584 completed with exit 0; receipt received. Poll the same job
only. Neither active public file has changed.

User supplied Rohan's November 2 workflow tweet. A fresh public repository audit
still finds main 018b7a from October 30; it inspected 101 workflow text files and
all 13 branch dates. Released code includes measured branch selection, dashboards
and mathematical Lean/QRH tooling. No released CUDA genetic optimizer, trained
GPU performance predictor or Lean GPU physics pruner was found. Do not claim
they do not exist privately. The audit and bounded source cache preserve exact
links and licensing. A small CPU schedule pruner is being added around our
existing lifetime model; it must retain rejection reasons and keep actual
registers and performance UNKNOWN. No learned prediction or formal Lean proof
is claimed. Public incumbents remain 97af/247888; the #1 goal is active and
unachieved. No Modal run, budget journal reset, paid service allocation or
unrelated cleanup occurred.

09:30 UTC: CPU schedule search is complete, independently reviewed and rerun by
root. tools/search_wgs_queue_cpu.py and its pinned manifest enumerate 320 cases:
181 unsafe lifetimes rejected, 139 pass the abstract lifetime checks, including
67 deeper-queue schedules. Both hazards are retained for 60 overlapping failures.
The root run matched the author's complete report SHA 5a183f932df663a466e8267c9f6b74ab548fbdf072e24b2aeba53c955abcacae:
93,408 trace comparisons and 7,472,640 commit groups. Every new configuration
retains UNKNOWN compiled resources and speed; none is automatically GPU-ready.
The previously selected four-bank wait2 candidate is not claimed to be fastest.
This is a bounded CPU filter, not a Lean proof, genetic optimizer or trained
performance predictor. The selected private job a0dd88ebc8964b3f was queued at
09:26 UTC; root remains its sole poller. No duplicate submission.

Additional primary-code review of KernelOPT/ForgeKernel retained 25 pinned files
and licenses (1.30 MB). No compatible faster kernel or validated latency
surrogate was identified. The independent KernelOPT implementation's heuristic
is not the paper's reported result; ForgeKernel's B200 FP4 path has a different
precision/scale contract. Its H100 GEMM wait2 results are mixed and do not predict
our RS MoE result. If the new check exposes register pressure, full-producer-
warpgroup register redistribution is a separate feasibility question, not an
approved implementation. Neither source framework was installed or executed.

09:32 UTC: a0dd88ebc8964b3f ended with official infra_error, no measurements,
and an explicit refund plus instruction permitting the same code again. This
is neither a correctness failure nor a performance result. Root retained that
terminal receipt and submitted one exact-source recovery, f685c2533ebc4214,
from the immutable 97ffccc snapshot. It is received; local session 11378 exited
zero. Only f685 is pending. Do not resubmit either handle while recovery runs.
The ordinary no-repeat rule still applies to measured outcomes; this recovery
addresses the service's explicitly refunded infrastructure failure. Main work
through the completed CPU filter and public-code audits is committed as3037716.

09:50 UTC: recovery f685c2533ebc4214 is terminal PASS, score2.331989071,
5tests/6benchmarks/6profiles, noflags. Reject scoring/integration/retry. Changed
M32/M128/M256 SGLang-relative speedups regress geometrically4.54% versuspublic
247888 and5.41% versusprivate; inverse-ratio latency rises4.76/5.72%. Allthree
raw medians worsen versusboth. Unchangedcontrols improve1.11/0.66%.
Actual WGS builds report C7511 insufficient-register WGMMA serialization for
BOTH specializations,96registers, zero stack/spill/local, oneCTAcapacity. This
is positive compiler-pressure evidence, not proof WAIT2 newly introduced it:
parent successful-build diagnostics were not fully visible. Root analyzer and
independentcritic agree the target failed. No official job is pending now.

Full128-thread producer-warpgroup redistribution is conditionally feasible:
640initialthreads at96registers give61440 CTA registers, exactly balancing
128producers at32 and512consumers at112. Hardware SM headroom is not a free
CTA pool. This remains a compiler/source feasibility question; the author is
preparing an exact plan against rejectedWAIT2 to test whether it can restore
intended overlap. No source or GPU run approved yet. Need actual initial96,
preserved donation/consumer roles/barriers, retaineddiagnostics, no hiddenfallback.

Latest public snapshot09:35UTC: overall/decoder386 atunchanged2.1812,
leader2.5401; MoE289 atunchanged2.3737. Normal CLI access to leaderjob returned
no suchjob and publicpage exposes no profile; no private endpoint pursued.
Known donor source pins remainunchanged. Broader bounded public search inspected
191repositoryrecords and7relevant pinnedrepos; no compatible newMusekernel found.
Only8MIT source/licensefiles362773B retained, withscope/caveats documented.
No framework execution, Modalallocation, creditjournalchange or unrelatedcleanup.
Owned publicinspectionbrowser tabwasclosed. Disk34GiBfree. Goalactive/unachieved.

10:05 UTC: root submitted one private REG112 check,42527fa2c64d4b42, now received.
Exact source31127a37ce35616b555e2ea5a5df673719aa03c422f9edc1a2be75320b3b9f65
is frozen in experiments/candidates/muse_moe_w4a16. Source is based on rejected
WAIT2, with only full128 producer WG, dec32/inc112 disjoint roles,640-thread join,
strict initial96/local0/positive-occupancy guards and a unique namespace. It
attempts to address the observed C7511 mechanism; it has no measured gain yet.
All math, traffic, packing, rings, finalizer and wgs_seg remain unchanged.

Independent exact-source review approved one private check after root verification.
Root reran tools/test_wgs_reg112_cpu.py (ed17310e…) successfully:16 unchanged
non-WGS builds,640 distinct actual dispatch paths repeated6 times,3969 actual
host predicate combinations, and the inherited1,462,272-call native MMA trace.
Repeated job labels do not add routing coverage. CUDA allocations, hint retention,
numerics and speed remain unproved. Local PTYs14313/91304 both ended with exit0.
Only42527fa2c64d4b42 is live; root owns polling. No duplicate submission or Modal
allocation. Retained C7511, dropped hints, visible spills, failed resource gates
or absent useful speed improvement reject this fixed32/112 attempt without a
quota sweep. Compare against247888 public/private and failedWAIT2 separately.

The10:00 UTC board snapshot is20261104T100006Z-register-plan-refresh.json:
overall/decoder387 at unchanged2.1812, leader2.5401; MoE290 at unchanged2.3737.
Public active sources remain97af/247888. Root and critic result audits agree;
parent serialization stays UNKNOWN because its official compiler logs are empty.
The canonical WAIT2 analyzer now preserves direct harness/raw timings separately
from inverse relative speed, and checks the saved compile log against the receipt.
Current turn made concrete source/verification/submission progress. Goal remains
active and unachieved; this is not a blocked turn or a claim of a rank improvement.

10:18 UTC: REG112 check42527fa2c64d4b42 is terminal PASS but REJECT.
Score2.312113768, CI[2.277432062,2.346795475],5 tests/6 benchmarks/6 profiles,
no flags. BOTH BIG entries retain explicit C7511 insufficient-register WGMMA
serialization. Actual640-thread/231168-dynamic admission passes at96 initial
registers, local0/static0, occupancy1/1; ptxas reports zero stack/spill bytes.
This independently fails the mechanism gate. No SASS was retained, so do not
claim setmaxnreg was ignored or quantify its causal contribution.

Root's exact-result audit: changed M32/M128/M256 geometric relative speedup is
0.939252 against public247888 and0.930704 against private247888; controls are
1.010155/1.005705. Direct harness medians rise5.99%/6.57%, raw medians6.48%/6.94%.
Versus rejected WAIT2, affected relative speedup is0.983931. Keep these separate
from inverse-relative-speed latency ratios. No scoring, integration, rescue,
quota sweep or identical retry. No official job remains pending.

The broader critic reassessment found no overlooked broad integration shortcut.
Latest97af profile scenarios imply25% less small-MoE time would cut whole-decoder
latency only5.37%; closing the observed14.13% gap through all MoE alone would
require about31.03%. These are profile-share scenarios, not forecasts or bounds.
Small-MoE WGMMA and lossless dense layout feasibility reviews are still in progress;
no new implementation is approved. Saved standalone M16 histograms allow only
0–10% J8→J16 packed-request savings, and are not decoder route histograms.
M1 standalone REG149 resources do not transfer to decoder's different flags.
No new Modal allocation or budget change. Disk36GiB free at turn start.
Previous turn was progress; this turn adds terminal measurement and closes an
unsuccessful mechanism. Goal remains active and unachieved, with no blocker audit.

10:49 UTC: one diagnostics-only whole-decoder check is received:
c0b4201c5ad345d1, exact source
e444f46f945e03bb42dab8d80ed4869f8a7da6464f845ac09ac3cf8ea19473fd.
It is based on current scored97af. It adds bounded HOST resource reporting for
actual post-prepare SK2 and small-MoE launches; every device function, packing
helper, compiler flag, launch expression and tuning algorithm remains unchanged.
It is NOT eligible for promotion or a performance claim, regardless of score.
Root alone polls; no other official job is live. Do not duplicate the upload.

Reporting stays disabled throughout prepare, including tuning and self-tests,
then arms loaded extensions; lazy modules inherit the armed state. Generation
means last arming epoch, not weight identity. Actual signatures include template,
physical shape, pmax, UV, PDL and dynamic shared size. Cap-only queries are excluded.
SK2 is capped at8192 bytes; six small-MoE modules have320-byte caps plus a128-byte
global reserve, total10240. Dedup state stays bounded after exhaustion. Small-MoE
emits one resource identity per module, then explicit partial coverage for a new
identity; API errors/truncation/missing records remain unknown. Post-prepare
observations are not labeled as benchmark versus correctness calls.

Root full source diff review and frozen CPU rerun passed; report SHA
c255633e96db54e98df4717c2f51783e0a54cabf1b5853d7ab2fc681f86d1c77
matches author's. Proof restores all parent97af bytes and generated SK2 sources,
checks actual logging/lifecycle/errors/bounds and72 MoE native mock cases across
six effective macro builds. Independent final source critic GO is retained.
Local PTYs90463/59171 ended with exit0. No CUDA compilation or numerical proof
is inferred from mocks. Official resource data may expose local storage or
admission limits; it cannot establish achieved occupancy, stall attribution,
dynamic spill traffic or compiler serialization. No automatic variant follows.

The10:48 UTC board now has a new leader, LeshengJin: decoder2.5510 andMoE2.7817.
Our unchanged decoder2.1812 is rank390; MoE2.3737 is292. Required normalized
decoder latency reduction is about14.50% against that snapshot. Source files
remain exactly97af/247888. Snapshot20261104T104834Z-selected-diagnostic-refresh.json.
A bounded public source scout of the new leader handle inspected all20 public
repositories and pinned three README/license pairs. No compatible new Muse/H200
kernel release was found; newest repository push was September23. No private job
access, participant message or branch traversal. Exact GitHub/leaderboard identity
is unverified. See LESHENGJIN_PUBLIC_SOURCE_SCOUT_20261104.md and its JSON evidence.

Independent budget read confirms no Modal allocation fits the existing guard:
nine reservations total21.14 USD and the saved billing evidence leaves negative
working allowance after the5 USD reserve. All nine manifests record termination,
but there is no supported settlement/refund path. Nominal provider credits are
not shown depleted; do not confuse guard allowance with actual credit balance.
No grant/journal/lock change, automatic recycling or new Modal allocation.
Small-WGMMA and narrower dense-codec feasibility are closed without implementation.
Current turn is progress: terminal failure audited, broader limits established,
new information-gathering source independently verified and submitted. Goal stays
active and unachieved; no blocked or completion claim.

## Selected decoder resource diagnostic completed — November4 11:04 UTC

Private check c0b4201c5ad345d1 of exacte444 passed all13 correctness tests,
12 timed cases and12 profiles, noflags/checkrejections. This was diagnostic-only;
its score is not a comparison or promotion signal. Public97af/247888 are unchanged.
No official job remains pending after this receipt.

The retained output has19 SK2 and3 small-MoE records, allg1 and successful APIs.
All observed local allocations are zero. SK2 regs71–116; QKV/pooler capacity1,
output projections and observedT16 BND unpooler capacity2, T32 unpooler capacity1.
CL3 router is block-only; cluster capacity is unmeasured. Small-MoE effective
builds report144/144/148 regs,214784 dynamic shared bytes and capacity1. Fused
pooler has384 static shared bytes; other retained records have0. These are
compiled resources/admission, not achieved occupancy, stalls or spill traffic.
Unexpected local allocation is not supported; close that hypothesis negatively.

No STOP, missing schema, malformed or conflicting records were retained. Saved
log3455 bytes is a prefix of5411-character receipt tail plus24 USDT lines. Thirty
SK2 profile rows each have one compatible record; none establishes exact benchmark
invocation identity. Twelve MoE profile names remain unmapped to flag builds.
Parsercc967 source review and14 fixtures cover conflict quarantine, strict
provenance, official pass eligibility and partial evidence. Root reproduced the
saved analysis exactly and separately audited the raw result. See candidate note
and20261104-decoder-selected-resource-{analysis,root-result}.json.

Independent profile coverage audit (root rerun) finds30 SK2 rows/12 prefixes.
Shared-memory floors already exclude two QKV/pooler CTAs; lower registers alone
cannot change that. Under profile-share scenarios,20% less all-SK2 time models
4.738% less decoder time;5% needs21.089% less SK2 time, or31.0% less QKV+pooler.
These are scenarios, not causal or critical-path evidence. No automatic variant
or repeated diagnostic is approved by this result.

At11:08 a fresh read-only Modal preflight verified the profile, returned zero
application records and provider_spend3.79USD. That is below the older3.90423640
snapshot; preserve the historical high-water figure for conservative proposals,
not a claim that credits increased. The unchanged guard still cannot fit its
minimum0.63 reservation. Grant,21.14 reservation journal and allocation lock are
unchanged. A separate read-only accounting review is checking whether exact
attributed provider charges could remove double counting while retaining every
original reservation. No accounting change or new allocation has occurred.

Current turn remains progress: valid compiled-resource evidence and independent
coverage audit obtained, incorrect local-memory hypothesis eliminated. Goal is
active and unachieved; no completion or blocked declaration.

The11:12 UTC final board refresh ranks our unchanged decoder2.1812 at391
(overall also391); leader remainsLeshengJin2.5510. MoE remainsrank292 at2.3737.
One additional entrant moved ahead; no public source replacement occurred.

### Accounting review and next independent feasibility

MODAL_PROVEN_OVERLAP_PROPOSAL.md preserves every original reservation and only
permits removing double counting proved by exact provider-attributed charges
included in observed usage. Current proved overlap is0: saved evidence lacks
run attribution/inclusion proof. Existing guard remains unchanged; no allocation
or automatic refund/recycling is authorized by this conditional proposal.

A separate bounded source/CPU feasibility review now considers smallerR16 SK2
output-row units across QKV and pooler. It must preserve losslessweights,UV and
routerfusion. The fused-pooler design specifically needs eight eighth-K router
pieces (rather than four quarters),2routerstages,larger of main/router payload
forringallocation, and correct eight-way ordered partial reduction/counter reset.
Actualnewregisters/occupancy/performance stayunknown. Quantify duplicated input
traffic,copy/stage events,reducedlookahead and wholedecoderscope before proposing
implementation. No parameter sweep, sourcecandidate or GPU check is approved.
The5% decoder target is a prioritization heuristic, not a user-imposed minimum;
any smaller gain still needs evidence strong enough to distinguish it from noise.
Authorattention_paper_reset owns the study; result_and_strategy_critic reviews
independently. Existing incumbent and closed failed hypotheses remain preserved.

R16 study checkpoint11:26UTC: critic found shared fused_cnt mixes old4-way and
new8-way schemas, which can incorrectly finalize afterfour arrivals when prior
counter is4mod8. Proposed resolution is uniform eight-way FUSE forTOKR8/16/32,
including autotuner-onlybuilds, with matchingpartbuffer andcounterlogic. It is
not yet an implemented or GPU-tested fix.

PinnedNVIDIA CUDA13.0.96 occupancy calculator CPU check (root independently
reproduced1152 rows) gives two-block register-only ceilings:416threads72regs,
384/352threads80,288threads96,256/224threads128. Compiler/runtime admission
remainsunknown. SourceheaderSHA163b8f296ef2a33c799e2dc8c525b37166da9217bcc6a3c1d26b1c10aca5e22b
is retained locally and excluded fromGit due its redistributionnotice; sourceURL,
probe andresults are versioned in20261104-sk2-r16-register-admission-critic.json.
Temporary native probe directories were automatically removed.

Tradeoff corrections: fusedR16 doubles mainTMArequests andactivationrequests,
unchanged48MiB mainweights/8MiB routerweights andtotalMMAwork. Aggregate twoCTA
weight-ring bytes actually fall33%/40% forTOK8/16; counting resident stages alone
misleads. SeparateT32pooler changesUPS4to8, so producer/TMAeventtotal staysfixed
whileactivationrequestsdouble; emptyarrivals fall. QKVgrid96to144 means only1.09
CTAs/SM onaverage, nottwoeverywhere. These costs must be explicit in the final
plan. Critic accepts a plausible independent-pipeline hypothesis as sufficient
for a bounded experiment after concrete geometry/lifetime/admission gates; no
priorproof of a speedup is demanded. Author's fixed feasibility plan is pending.


## Fixed R16 family submitted — November4 11:58 UTC

The feasibility review and detailed implementation plan were accepted for one
bounded experiment. Exact candidate464115b2c871338f8d0f51768b82f6bd9090139aea13588ec668a9de587e44f8
implements the whole A/B/C family: R16 pooler/QKV, uniform eight-way fused router,
matching scratch/ring layouts, and14 resource cases covering12 compiled pointers.
It preserves all unrelated device sources, compiler options and tuning policy.
Actual preparation admission requires primary two-block capacity, zero local
allocation and clean CUDA APIs; compatibility paths have the documented limits.

Independent final source review f8165595de8a93756b02b85bb1d076bf089f2bbccc8d06409447c2b21f2576f3
found no source blocker, with CUDA compilation/registers/numerics/speed unknown.
Root reran the complete CPU proof at11:57 with exit0 and regenerated the identical
full report f5d75febd9e083563bd42e4edbd73cd93657ffe282e0bc7cc95bda32fbd54c57.
The separate root summary is20261104-sk2-r16-root-cpu.json. The combined report
includes final-source native router tests; the standalone router-native artifact
preserves the earlier094ddc92 review and is not the final-source proof.

Root submitted ONE private check28d3c46e68504ee4 at11:58:46UTC. At12:05 it was
queued with server-confirmed board/mode/team identity. Root alone polls; no duplicate
upload. Before submission decoder had18 scored and31 checks left. This pending
check is not a gain, a passing correctness result or a leaderboard replacement.

The fixed promotion decision needs >=2% whole-decoder speedup over BOTH exact97af
public7a8c4941741645ee and privateb2a773e2d7194de8, positive affected-load gain,
no affected case >3% normalized regression, control-load geometric drift within3%,
clean13 tests/12 timings/12 profiles, and intended activation/resource evidence.
The performance analyzer reproduces46/16 weighted loads, checks source/receipt/
profile/evaluator provenance, and keeps raw/harness/baseline ratios separate.
Its21 fixtures pass independently, including minimal received acknowledgements
without metrics and strict full terminal identity. Resource/activation analysis
is a separate gate; neither script can authorize automatic public promotion.

A failed fixed attempt is closed without quota rescue, nearby sweeps, duplicate
retry or automatic further diagnostic. Actual new compiler resources and the
previously dead generic QKV UV epilogue branch remain experimental risks.
Public97af/247888 are unchanged. The11:56 decoder board is392/1265 at2.1812,
leader2.5510; no new overall rank was fetched. Modal guard and permanent journal
are unchanged, with no new allocation. Current goal remains active/unachieved.


### Fixed R16 resource rejection — November4 12:09 UTC

Private28d3c46e68504ee4 exact464115 completed with passed=false,returncode112,
one failed setup event and no correctness tests/timings/profiles. The first
fusedpooler8 PK1 pointer compiled to74 registers at416threads, zero local bytes,
384static/104160dynamic shared bytes, actualpmax32, and capacity1 vsminimum2.
All six CUDA APIs succeeded. This exceeds the previously established72-register
two-block ceiling. Ordinarypooler16 PK1/PK0 records before it had78 registers,
zero local,90768dynamic,capacity2 vsminimum1. Only3of14 resource cases reached;
remaining11 unknown. No performance estimate follows from this resource failure.

The backend catch reported the resource rejection and the outer preparation
sentinel failed the official setup as designed; no silent alternate backend was
timed. Root preserved the complete terminal receipt and identity-checked raw
audit20261104-sk2-r16-root-result.json. Performance analyzer returns
OFFICIAL_GATE_FAILED with no comparisons. Independent raw/result/parser review
is in progress. Exact fixed R16 attempt is closed without quota rescue, nearby
sweep, duplicate retry or automatic diagnostic. Public sources remain97af/247888.
A separate read-only next-strategy review is considering a materially different
mechanism from actual incumbent profiles; no replacement candidate approved yet.


Independent terminal/result critique3d115aa867cb0f662ae310c3d5e78a01ef50d2ddb63e50e7eac989debf8d1ecb
and parser critique8f3169a04a159ff0d4e1274774f67f78a19c9f227c4052b7e42e96c16a1ac12a
confirm the rejection and unchanged public identities. Root independently ran18
resource-parser fixtures and exactly regenerated reportfb5db779a16b6b3d729da8d668a8eb7343b3c331235429649595427ca33f75f6.
Performance-reader21 fixtures also pass. Dormant successful-activation helpers
remain conservative: UV modes and sharedB32/QKV32symbols can be ambiguous. Their
completeness is not established by this failed job and no extra probe is justified.

A distinct centralized timed MoE route/job metadata scheme was assessed and
deprioritized, not declared impossible. Preserved standalone DBG0-toDBG1 windows
include initialization/PDL/route; they are small and notisolatedremovablelatency.
Source-feasible last-arriver metadata publication adds dependencies/copy/earlyweight
delay. Exact97af profile-share deletion scenarios yield0.564% for3uslessMAIN or
3.292% for5uslessallcases, optimistic scope checks rather than predictions. See
NEXT_COORDINATION_FEASIBILITY_20261104.md anditscompactJSON. Noimplementation.

Next bounded read-only checks: materiallydifferentMoE compute/operand schedule,
and whether savedfailedaggregatecandidates containrepeatableintendedshape wins
that canbelegitimatelycombinedwithoutcherry-pickingunchangedcontrolnoise. Exact
parent247888/ef7compilerserialization remainsUNKNOWN; only WAIT2/REG112 have
positiveC7511evidence. A freshNVIDIAforumC7514loopdependencyissue is adifferent
diagnostic and supplies no causal evidence for ourparent. NoGpu/checkapproved
bythesequestions; actualsourcefeasibility and independentcriticstillrequired.
Primary source: [NVIDIA compiler engineering reply on C7514 loop analysis](https://forums.developer.nvidia.com/t/ptxas-mysterious-warning-for-wgmma-mma-async-instruction-serialization/340610/8).


At12:17UTC fresh official snapshot confirms decoder392/1265,overall392/1962,
score2.1812 vsleader2.5510. Decoder30private/18scored attempts remain; MoE33/17.
No official job is pending. Current exact sources unchanged. Next standalone
MoE feasibility considers SS-WGMMA decodedBF16 A insharedbanks, retaining
originalworktiles/fullweights/jobs/rounding while trading more sharedtraffic
and reducedraw-ringlookahead for fewer live register operands. It is a distinct
source/CPU feasibility, not a retest of quotas or WAIT2. Authorattention_paper_reset
and independentmoe_research_reset are reviewing before any implementation.


Complementary shape-dispatch review2cae13e1fd4cccd491b5884e7b90bc6eedc00f4641a58e8a961cd74af798436a
found no supported combination:27savedpassingprivate runs,25withmatchingGPU/
evaluator andcasefieldsapartfromseed,nonehasanycase>=3%fasterthanbothcurrent
public/privateincumbents. Rootreproduced27sourcehashes,25fullevaluator/GPU/
casecomparisons and allnegative3%winscreens directlyfromreceipts. Unchanged
controlnoise isexcludedregardless. LocalGS4,earlyA,splitfinalizer also
lackrepeatablepositivefamilyevidence. Do notreopenmixeddispatchwithoutnewdata.
Currentgoalturn remainsprogress: actualR16admissionfailure closed, read-only
coordination/combinationalternatives assessed, distinctSSoperand designunderreview.
The#1objective isactive/unachieved, notblocked orcomplete.


## Shared-A WGMMA CPU implementation accepted — November4 12:32 UTC

Root acceptedfixedSS-Aplan c3c30a91d55abc91e37fdb0f37e427111ef1c0f93277e29d74361f4f0dca13b7
(original4df441db plusreviewedstorecostparagraph), afterindependentplanGOand
exactrootreproductionauthor e01d54aa andcorrectedindependent db40647c reports.
A sharedfilenamecollisionwasdetected: thefirstpurportedindependenthelpercopied
theauthor'sscript. Itwasreconstructedunder *_critic.py withitsownoutput before
rootacceptance. Correctindependentscript4742124b; onlythereconstructedreport
isindependentCPUevidence. Source/runningcandidatewereuntouchedbythiscollision.

The fixeddesign keeps576threads,FT64,256outputrows,K256,GS2/wait1,alljobs/math
andKPERM; storesdecodedBF16 A into2K32sharedbanks, addsWGpreoverwriteand
publicationbarriers, andreducesraw/Xrings3to2. Dynamicshared194304B fitsoneCTA
onpaper. Extra96MiBstores+96MiBlogicaltensorAreadsand49152WGbarriers/job are
materialcosts. Documented128B/SM/clockgenericstoremodelat132SM,sampled1980MHz
gives72.22–87.26usserviceforJ24–29; notadditivelatency,notcontinuousclockor
measuredHopperportthroughput. Tensor/TMApathsarenotincorrectlychargedtothat
model. A20%MAINtimecutmodels5.045%decoder; thisisnotaforecast.

Writerattention_paper_reset ownsisolatednative_wgs_shared_a_v15 fromexact247888,
builderandcombinedactualsourceCPUproof. Independentmoe_research_reset owns
itsdistinctlifetime/descriptorproof; result_and_strategy_criticreviewssource.
NoGPUorofficialcalluntilfrozenfullsourcereviewandrootrerunpass. Runtimegate
isvalidlaunch>=1CTA/successfulAPIs;localbytes/C7511areinformation,notarbitrary
rejectionbeforetiming. Fixedusefulgate>=3%affectedMAINgeometricgainoverboth
publicd24dd501/privatebcb761df, noaffected>3%normalizedregression,control
GMwithin3%. Usefulstandalonegainstillrequiresseparatedecoderintegrationcheck.
Noactiveincumbent/Modalguardchange;#1goalactive.


## Shared-A candidate frozen and submitted — November4 12:46 UTC

Exact standalone93c2d0fbfe1d37792389402b335d34f8e00328e84ce25ad8ea732b9726d08a38
isimplementedandfrozen. Deterministicbuilder251690b8 restores247888outside
approvedscope, preserves41Pythonnodes and16nonWGSpreprocessedbuilds. Combined
actual-sourceCPUproof57400109290853f0b52b52bebd2ffcd5814ce068134df90e2ef32bf5e8b244f0
passes16SSwrappers,16384coordinates/exhaustiveBF16pairtransport,1462272matching
parent/candidateoperandtraces, actualfullKPERMpackedbytes, CUDAhostmocks/error/
retry/cached-success and3nativecorruptionnegatives. Nativearithmeticisstubbed;
noGPUTensorCore/numericalperformanceclaim.

Independent02bf25a9 verifier/report68286947 covers48interleavings631632events,
32raggedrange/inverse-ownercases,13unchangedfunctions and9negativecontrols.
ItswaitretirementisWGcollective; actualsourcepre/postbarriersaremandatory.
EmptyCTArangescovered, arbitraryglobalnj0outsidebusiest-rankgeneratorisnot
proved; inheritedroutingunchanged. Finalsourcecritic4f53a826c8437a4e9d3c9ea3f8284d416dc61fa28107e51ebed073416adb1e2c
foundnoblockerandapprovedoneprivatecheckafterrootrerun.

Rootran fullsuiteindependentlywithexpected93c2hash; terminalexit0 andseparate
root-cpu.json isbyte-identicaltoauthor57400109. Theperformanceanalyzer's20
fixturesalsoPASSroot,includingbothactual247888incumbentreceipts. Itnormalizes
onlyabsentvsexplicitemptybenchmarkflagsinoldCLIprofileexports; anyactual
metric/source/profile/evaluator/GPU/caseconflictfails. Noautomaticpromotion.

RootsubmittedONEprivatecheck100bb9bbce8f4539at12:46UTC; acknowledgedqueued
withserverboard/modeconfirmed. Sourcefrozeninexperiments/candidates; receipton
diskbeforefurtherpolls. Rootalonepolls; noduplicateupload. PresubmitMoElimits
were17scored/33checksleft. Active97af/247888unchanged, noModalallocation.
Resourceparser/criticandread-onlydecoderintegrationmappingcontinuewhilepending;
nointegrationcandidateisapprovedwithoutusefulstandaloneresult.

Currentgoalturnisprogress: authorizedrewriteimplementedandindependentlyverified,
newofficialmeasurementqueued. Thegoalremains#1,active/unachieved.
