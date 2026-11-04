# Research loop: current handoff

Updated November 04, 2026 at 23:10 UTC with V33 official check accepted.
The #1 goal remains active and unachieved. Prizes belong to the separate serving
track. Component correctness, component speed, public kernel rank and full
serving throughput are distinct outcomes.

## Resume here

**Latest user steering: public #1 and a 100% score-gain stretch target.**
The existing goal remains active. Public decoder is 2.1811831350 at #409,
leader 2.5761628684 in the 21:51 UTC snapshot. Doubling means 4.3623662701;
that is an objective, not a forecast. Official GPUs remain available. Modal
is blocked by the current conservative free-credit ledger; no paid allocation.

**V29 closed; no score or identical retry.** Official `185fee4d182f41dd`
passed 1 setup/13 tests/12 benchmarks/12 profiles with clean flags, but score
1.8985960692 (CI1.8701171281..1.9270750102) regresses against both exact97af
parents: whole ratio .870443/.872254; affected four-load speed GM .818498/.820504.
All eight affected cases fail the latency cap; controls pass. Critically, ZERO
compressed V29 instances appear in those profiles. Dense work falls back to
other kernels; renamed COMP0 router still runs. Four packing logs have normal
raw rates .122..250%, so byte fallback is not the demonstrated cause. Do not
call this a measured repair-kernel slowdown. Selftest exceptions/errors and
tuning choices were silent. Root and independent critic agree the fixed attempt
failed activation and speed, while the exact path's eligibility remains unknown.
Source bbbc37f8 and all raw evidence/critics are preserved. New V31 will expose
those existing decisions without changing math, geometry, thresholds or selection;
it is a single diagnostic-only experiment, never scored or a noise retry.

**V30 closed; no retry, port or promotion.** Official `d0510bc0412a4537`
passed1setup/5tests/6benchmarks/6profiles with clean flags. Score2.0335370900,
CI[2.0030340336,2.0640401463], against public2.3736910444. Whole ratios
.856698/.850911; affected speed GM .730721/.724071 against exact public/private.
M32/128/256 normalized latency public1.41853/1.34930/1.33905; controls stable.
All six phases activate; actual cold kernels64regs/local0/capacity2 and hot
96regs/capacity1 (hotA8B local, hotB0). Route+final only4.4..5.1% of profiletime;
no single causal blame for the loss is established. Source ac03c2ca and all
raw receipt/profile/log evidence, root results and independent critic retained.
Plan timing exception remains explicit: plan/commit followed upload by27seconds,
while source/gates/review were fixed beforehand and terminal result arrived later.

**V31 diagnostic resolved: honest tuner loss; no promotion.**
Official `5f48e0da4dcd4595` passes full correctness, score1.9115472025.
All16 compressed SK2 plus4 router selftests pass, disabled=[] and no exceptions.
Fifteen compressed compiled pointers report localSizeBytes=NT*32 (32/64/96/128B),
exactly the eight-word array size; four router observations report0. Inner timings
reject14/15 retained compressed choices; unpoolerM1 wins internally but loses
at outer selection. All24 outer rows retained, no exceptions; all12 timed profiles
have zero COMP. Inner logging stops after18rows, so unpooler/routerM8 inner results
are unknown. This is prepare-time execution/resource evidence, not timed COMP
performance, nor proof that array storage alone explains the slowdown. Raw logs,
receipt, profile, root and independent critic are preserved in V31 records.

**V32 closed; compiler fix succeeded, speed premise failed.**
Official `6a938fff9d5d417c`, exact ea4470fc16989b9d66cbab40fdae588fb30b2ffe5446f86c14e8b13854813e35,
passed13tests/12benchmarks/12profiles with clean flags. Score1.90808038695,
CI[1.87945918115,1.93670159275]. Whole ratios.874791/.876612 against both97af;
affected four-load GM.823866/.825885; controls pass. All19 observed pointers,
including15COMP, have local_bytes0. Still zero timed COMP activation; all20
prepare selftests pass with no disabled keys. Removing array local allocation
was real but insufficient. No forced selection, identical retry, or gate changes.
Raw receipt/profiles/log, root result and compact closeout are retained underV32.
Independent critic reproduces root result. Next V33 is now frozen and independently reviewed.

**V33 accepted as official check `60b61041d08a4880`; inspect, never duplicate.**
Source/gates/critic were committed `83db72d` before upload. Root owns polling. Source
`a8a619eaa0c531aa115d39fd2db8718d7d52828feac8e6fdea005c4e61a3c090`
at `muse_decode_layer/candidates/decoder_nibble_sidecar_v33/submission.py`.
Fixed matrix-window normal decoder; escaped units use literal primary high-byte
low nibbles plus256B missing-nibble sidecar instead of per-word repairs/full-raw
sidecars. Primary loads no longer depend on descriptor. Geometry/selection/MMA
order/handback remain. Actual retained V32 logical bytes predict+2.15..2.80%
versus768B primary,+1.44..1.92% versusV32; more frequent global requests may lose.
Root/author/critic reproduceCPU98355753:2,572,288 bit comparisons,2,808 mappings,
4 rejected mutations. Separate layout review passes; six analyzer fixtures pass.
Source critic GO4385170f. Fixed plan `20261104-decoder-nibble-sidecar-v33-root-plan.json`
requires whole>=1.025/affectedGM>=1.04 against BOTH97af, all latency caps/controls,
clean13tests/12benchmarks/profiles, exact COMP activation and observedlocal0.
No retries, forced selection or gate relaxation. No public source changes.

The unique-expert traffic audit is closed read-only, no new implementation:
retained job duplicates offer at most a conditional+1.465% decoder scenario,
FT128+1.093%; no actual decoder routing census exists. Evidence in
`experiments/iterations/20261104-moe-unique-expert-traffic-audit.json`.
Root owns all GPU calls. V33 `60b61041d08a4880` is the only job in flight; no Modal allocation.
Serving work remains backlog under the current public #1 priority.


**V28 closed; no score or retry.** Exact decoder source
`d1b1bcf5c241710697bc0d2fb39836b1d7064c262bff7f820b564085e2136d9b`
passed 1 setup/13 tests/12 benchmarks/12 profiles with clean checks and correct
activation markers. Score 2.1146266798, CI [2.0829072796, 2.1463460800], fails
both exact public/private 97af parents. Affected weighted-load latency GM is
1.051510 public / 1.047566 private; every affected case is slower. Unchanged
control average drifts about 1.1–1.2%; separate profile timings do not establish
atomic, traffic or register causality. The source replaced public vector atomics
with private partials/owner sums, retaining public FTZ. This does not negate the
independently passing serving V26 scalar-CAS replacement.

Frozen source/plans/reviews: commit `b0c3eb5`. Terminal records:
`experiments/iterations/20261104-decoder-v28-root-result.json`,
`20261104-decoder-v28-result-critic.json`, `20261104-decoder-v28-closeout.json`.
Retain raw receipt/profile JSON, text and log. Do not change thresholds, rescue
selected shapes or repeat this attempt. `tools/DECODER_AFTER_V28_PRIORITY_20261104.md`
found no evidence-ready high-value public probe; the next useful work is V27
serving integration, not another WGS parameter sweep.

**V27 adapter source-reviewed; fixed device attempt closed incomplete.**
Isolated package: `serving/candidates/native_serving_v27/`.
Native module SHA `7310dd760cee4ae4802b1290d46d3dbb2e871a1cde131b78bb6e69743c23d0c1`;
patch SHA `c0ca4e41e595df59046ccb67d3641053148090b9ed4019dc534eb229522700c6`,
213,160 bytes across four allowed source files. No dirty serving checkout export.
Installer sees actual rank-local raw weights before Marlin repack; qualifies
supported serial execution, prepares before mutation, registers the bank and
releases six raw slots transactionally. It retains actual norm metadata, pre-norm,
original finalizer, BF16 boundaries and outer all-reduce. Model/device/lane owners
share serial workspace; unsupported overlap/partition/offload/compile policies
fall back before mutation. Norm setter uses the native bank device after release.
CPU checks cover real hooks, eight rank installations, policy/metadata declines,
rollback, all chunk/norm cases and source restoration/patch application. Root
and independent critic reproduced them; this is not device or scheduler proof.

Device run `20261104T213729Z-f6cf2030` used the frozen worker/controller from
commit `787f1a7`: oneH200,300s sandbox,280s exec,240s cooperative budget.
Native rank3 installation returned; bank905970816bytes/raw slots released,
independent raw tensors and one workspace were recorded. Last checkpoint is
`stock_prepare` at88.045s, not total runtime. The uncheckpointed span includes
stock clones/preparation, synchronization, input generation and two initial
stock forwards. Exec returned-1 at its deadline. **0/38 correctness cases,
0timings, no serial/stream/graph results.** No numerical failure or pass inferred.
Live-app recovery retained the checkpoint; sandbox terminated, provider app
stopped/zero tasks. Preserve all staged files/licenses and source pins.

Root result/independent critic/closeout are under `experiments/iterations/`
with prefix`20261104-native-serving-v27-device-`. Diagnosis:
`tools/SERVING_NATIVE_MOE_V27_STOCK_PREP_DIAGNOSIS_20261104.md`.
Do not rerun the fixed full suite. Its next distinct diagnostic would trace only
the stock interval with per-stage checkpoints, unchanged caps and no native
rebuild. A separate prospective completion-reader defect is reproduced:
Python3.13 ast.dump omits empty fields by default; show_empty=True reproduces
all three remote AST hashes. Source bytes match. This did not cause the timeout;
a future worker needs stable serialization and runtimePython metadata.
The stock-only probe is deferred behind the user's renewed public-board push.

V26 official private `274c69cbc0954d76`, source
`944768d40e380bfb88f917ed5e69fcc1f9470995d3e6d65cfb6267b756e04bb7`, remains
clean: 1 setup/5 tests/6 benchmarks/6 profiles, same-run stock score1.6851100701,
all four frozen gates pass. M1/8/16/32/128/256 normalized latency:
.869651/.755557/.725845/.632443/.406860/.355883. This is1.3440x V25's normalized
aggregate but only.7099x public native, across different runs/seeds. Keep for
serving integration; never score it as a public improvement. One bank864MiB,
serial workspace123.03MiB; conversion needs two-bank overlap plus packing temps.
T16384 requires43 core/finalizer chunks. Full-model memory, eight-GPU scheduling,
TTFT and throughput remain unmeasured. Integration contract and V26 result/critic
records are retained; all official jobs are terminal as of this update.

## Incumbents, deadlines and money

Preserve public sources:

- MoE: `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
- Decoder: `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
- Attention: `87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c`.

At 21:51 UTC: decoder #409; MoE #312 at 2.3736910444x; decoder #409
at 2.1811831350x. Leaders: 2.7817206619x MoE, 2.5761628684x decoder. No public
score changed in V26/V28. Snapshot: `experiments/leaderboards/20261104T2127-v28-closeout-v27-device.json`.

Live API then: kernel cutoff **November 5 08:00 UTC**; serving opens **November
10 17:00 UTC**, closes **November 20 08:00 UTC**. Recheck live rules/API before
new work. Kernel limits: 40 private/20 scored daily per problem; after V26,
MoE has 25 private checks/17 scores and decoder 28 checks/18 scores left today. Serving: 3 checks/1 score daily,
eight H200s, private board. Overall kernel top 50 earns queue priority, not
prizes. Stop kernel submissions at kernel cutoff; stop experiments and disable
the monitor at final contest cutoff. Live precision/patch/concurrency/TTFT rules
are authoritative over historical notes.

Read `MODAL.md`; use mandatory guarded runners only. At 21:43 UTC provider cost
floor was $4.83274092, billed $0; recent apps stopped and manifests terminal/
terminated. $30 grant, $5 headroom and immutable reservation/correction ledger
leave conservative working allowance -$0.46274092 after the new full$1.14
reservation. No further allocation fits. This is conservative journal accounting,
not a claim the $30 provider allowance has been exhausted. Any completed-cost
reconciliation needs exact provider attribution and independent review, never a
lock/journal reset. Fresh checks remain mandatory. One H200, <=900s, no
allocation retries, guaranteed termination; only one project Modal allocation
at a time. Never reset lock/journal or refund
reservations without reviewed evidence. No out-of-pocket GPU/API charges or
weight downloads are authorized. Use official GPUs/offline work if blocked.

## Supervisor adaptation: diagnose, change, execute, measure

Apply [Evolving Agent Teams](https://arxiv.org/abs/2610.36848) through existing
agent tools. This is a local adaptation, not a reproduction of its results or
installation of the Claude/API harness. Adding reviewers alone is insufficient.
At each substantive iteration root records a compact reflection:

1. Exact objective, incumbent, raw receipts, affected cases and decisive failure
   or unresolved gate. Separate observations from causal hypotheses.
2. Specific diagnosis of team/workflow failure or what changed after success.
   A failed design does not establish a hardware ceiling.
3. Concrete change in roles, coordination or persistent knowledge. Specify
   deliverables and disjoint file ownership. Implementation and independent
   critic remain separate; root alone submits, polls, allocates and commits.
4. Falsifiable experiment and frozen correctness/performance gates, estimated
   upper bound explicitly labeled with assumptions, and kill/pivot criterion.
5. Raw outcome, costs, exact source/evaluator hashes, critic findings/response.
   Relaunching without a changed diagnosis, hypothesis or evidence needs an
   explicit justification; otherwise select a different probe.

Use available slots (four including root), reuse agents, and never launch an
external paid swarm. Choose roles for the actual blocker: measurement fidelity
after a harness mismatch, numerical semantics after a boundary failure, or
integration ownership after a component pass. Research must identify a compatible
code path or reject a lead. Resolve material findings, then execute rather than
prolonging review. Save each reflection under `experiments/iterations/` and link
it here. This change was motivated by V22/V24/V25 failure evidence and V26 success.

Keep this current handoff concise. The prior 1,805-line file is preserved verbatim
as `RESEARCH_LOOP_HISTORY_20261104.md`, SHA256
`74ed09d7e6e2c5cde9c0739540620affdf4d51852e327aad8eb06b2022ef8f4a`.
Its old next steps and budget/deadline snapshots are historical evidence, not
active instructions. Consult the archive for a specific previous hypothesis.

## Validation and research

Read project/problem AGENTS.md and TASK.md. Preserve official baseline/reference/
checks/nvfp4, licenses, exact sources and public incumbents. No seed-specific
logic, caching, tolerance/routing/precision changes or timing shortcuts. Only
weights may be repacked in prepare. Kernel rules and serving BF16 requirements
differ; a kernel pass does not qualify a serving port. Preserve existing edits
in nested `serving/sglang`, pinned at `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`.

Serving export must use allowed source/JSON files under `python/sglang`, never
`python/sglang/srt/model_loader/`, no binary/rename edits, <=409600 patch bytes
and no added line longer than 400 characters. V27 wraps the nine overlong V26 assembly-constraint lines without changing tokens;
its patch passes these static bounds. Recheck live rules before submission.

Freeze source/gates before measuring. Check full numerical coverage, flags,
per-shape/aggregate timings, evaluator/image identity and whole-chain activation.
Score only an exact privately passed source with useful public-incumbent gain;
verify scored outcome and leaderboard identity. No automatic promotion. Serving
additionally needs memory, generic ranks/shapes, stream/graph, prefill and actual
end-to-end validation. Do not infer full-service gain or unobservable private rank.

Research intake:

- `EAT_REVIEW_20261104.md`: team adaptation; no matching H200 Muse decoder source
  found in its advertised repositories/traces. No direct kernel import.
- `PARALLEL_KERNEL_BENCH_REVIEW_20261104.md`: four-H100 distributed benchmark;
  phase-aware communication bounds and slowest-rank validation are useful.
  No direct one-H200 W4A16 speedup; existing SGLang custom AR is already present.
- `TOGETHER_COMPILER_REVIEW_20261104.md`: two validation levels, representative
  routing distributions and concise source-access context. Persistent-engine
  hot reload is a method, not a verified public TKCC code release. Do not launch
  its 24-hour/twice-replica campaign. Official evaluators remain unchanged.
- `LATEST_KERNEL_RESEARCH_20261104.md`, `UPSTREAM_KERNEL_SERVING_RESEARCH_20261104.md`:
  prior X/arXiv/GitHub/Hugging Face/DeepSeek/CUTLASS research and rejected ports.
- `../formal/README.md`: completed source-bound Lean MULDEC uint32 and packing
  bijection proofs, negative controls and optional upload gate. Not full PTX,
  CUDA, floating-point, race or speed proof. Strict binder does not support
  V25/V26 composition; do not add dead helpers or weaken it for a certificate.

## Closed attempts: do not silently reopen

- V25 `033a2d2eca374d82`: official numerical pass, M1 +13.63% stock latency failed
  its fixed gate. V26 is a different reduction design; preserve V25 evidence.
- V22 finalizer: useful official gain; generic 89/90 with H511 cancellation
  failure. Unchanged-helper stage diagnostics closed. No tolerance rescue/port.
- V24 router: 432 exact +8 padding +8 legacy passed, then parent launch-count
  expectation failed before timing/streams/graphs. Closed; no speed claim/retry.
- V23 attention exact rescale: official pass, 2.5911x missed 2.6120x/2.6425x
  parents. No scored submission.
- Older shared-A, R16, SC6, prefetch, wait-depth, register, packing and scalar
  reuse failures remain in the archive and `experiments/iterations/`. New work
  must address the actual failed premise, not repeat a disproven parameter sweep.

Latest completed supervisor record: `experiments/iterations/20261104-v27-v28-supervisor-reflection.json`.
V28closeoutcommit `29d9334`; V27device source/reviewcommit `787f1a7`.
The renewed public-board reset is in progress after the latest user steering.

Heartbeat `okc-research-and-harsh-critic` runs every 15 minutes. Resume live work
first; notify only meaningful verified gain/rank change, material failure,
rule/deadline change, completion or required user action. Commit compact evidence,
keep scratch bounded and close owned processes. Turn-start physical free space
61,319,245,824 bytes; no cleanup was needed.
