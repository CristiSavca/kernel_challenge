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

Dense V9 started at 05:06 UTC in
`experiments/modal/20261104T050629Z-62c33804`, sandbox
`sb-CIaIZOy169rvKZqFdQvkGt`, local session `11768`. Resume it; do not allocate
a duplicate. The nine-case combined file includes six held-out timing cases
and three nearby/scale/outlier checks, six fresh replays per case (144 planned
checks), profiles, and bounded prepare-policy metadata. Weight seed is 92011;
input seed base is 91021. The exact candidate remains `3d5b3462...`.
Only one H200 is allocated, at most 900 seconds. Latest billing before this run
was $2.02 metered, all covered by credits, and $0 billed. The credit guard
reserved the next run conservatively before allocation.
