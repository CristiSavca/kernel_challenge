# Research loop: current handoff

Updated November 4, 2026 after official V26 job `274c69cbc0954d76` completed.
The #1 goal remains active and unachieved. Prizes belong to the separate serving
track. Component correctness, component speed, public kernel rank and full
serving throughput are distinct outcomes; state which each experiment measures.

## Resume here

V26 source `944768d40e380bfb88f917ed5e69fcc1f9470995d3e6d65cfb6267b756e04bb7`
passed its official private MoE component check: 1 setup, 5 tests, 6 benchmarks,
6 profiles, clean checks/flags. Same-run stock speedup is **1.6851100701x**,
CI [1.6598334191, 1.7103867212]. All four frozen component gates pass. Normalized
latencies at M1/8/16/32/128/256: .869651/.755557/.725845/.632443/.406860/.355883.
Root evidence: `experiments/iterations/20261104-native-serving-v26-root-result.json`.
Source/preflight/review commit: `3189741`. Plan and evidence pins:
`experiments/iterations/20261104-native-serving-v26-root-plan.json`.
Independent terminal-result review agrees: keep for contract validation, no
public promotion. Its pinned evidence is
`experiments/iterations/20261104-native-serving-v26-result-critic.json`;
the compact terminal record is `20261104-native-serving-v26-closeout.json`.

V26 replaces scalar CAS accumulation with private boundary partials and an
ordered non-FTZ FP32 owner sum. It retains reviewed per-partial MMA order,
Marlin scale LUT/BF16 boundaries, local activation FTZ, cooperative launch and
stock finalizer. One weight bank is 864 MiB; one serial workspace is 123.028839
MiB. Both specializations report 576 threads, 231168 dynamic shared bytes,
96 registers, 32 local bytes, occupancy admission 1 and a 132-block cooperative
grid. Local bytes alone do not prove spill cost; profile times are not timings.

Keep V26 for serving validation. Its normalized aggregate is 1.3440x V25's but
only .7099x the faster public native score. These are cross-run comparisons
with different seeds, not direct same-input A/B. Do not score V26 or replace
public sources. No full-model memory, generic rank/empty route, stream/graph,
large prefill, TTFT or official eight-H200 serving pass has been established.
All known official jobs are terminal. No Modal allocation was made for V25/V26.

**Next assignment:** one serving integration contract owner maps actual pinned
SGLang loading/Marlin calls to `prepare_native`, `Workspace`, `expert_outputs`.
Require a concrete one-bank ownership transition, peak-memory account, workspace
lifetime through finalization, stream/graph policy and minimal GPU validation
matrix before writing an adapter. The owner can reject an infeasible contract.
Another kernel parameter sweep does not address these deployment risks. Root
alone owns the next allocation. Check live agents/processes/manifests/receipts
first; never duplicate an accepted job or overlap writers to an artifact.

## Incumbents, deadlines and money

Preserve public sources:

- MoE: `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
- Decoder: `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
- Attention: `87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c`.

At 20:56 UTC: CristiSavca #405 overall; MoE #310 at 2.3736910444x; decoder #405
at 2.1811831350x. Leaders: 2.7817206619x MoE, 2.5761628684x decoder. No public
score changed in V26. Snapshot: `experiments/leaderboards/20261104T2056-native-serving-v26-closeout.json`.

Live API then: kernel cutoff **November 5 08:00 UTC**; serving opens **November
10 17:00 UTC**, closes **November 20 08:00 UTC**. Recheck live rules/API before
new work. Kernel limits: 40 private/20 scored daily per problem; after V26,
MoE has 25 private checks and 17 scores left today. Serving: 3 checks/1 score daily,
eight H200s, private board. Overall kernel top 50 earns queue priority, not
prizes. Stop kernel submissions at kernel cutoff; stop experiments and disable
the monitor at final contest cutoff. Live precision/patch/concurrency/TTFT rules
are authoritative over historical notes.

Read `MODAL.md`; use mandatory guarded runners only. At 20:37 UTC provider cost
floor was $4.25390221, billed $0; recent apps stopped and manifests terminal/
terminated. $30 grant, $5 headroom and immutable reservation/correction ledger
leave conservative working allowance $1.25609779. This dated value cannot bypass
fresh checks. A 600s/$1.78 reservation does not fit; 300s/$1.14 only fits if fresh
guarded accounting permits it and the test can finish. One H200, <=900s, no
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
and no added line longer than 400 characters. V26 has nine overlong lines;
it cannot be exported verbatim. Recheck live rules before the isolated adapter.

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

Heartbeat `okc-research-and-harsh-critic` runs every 15 minutes. Resume live work
first; notify only meaningful verified gain/rank change, material failure,
rule/deadline change, completion or required user action. Commit compact evidence,
keep scratch bounded and close owned processes. Turn-start physical free space
53,796,264 KiB; no cleanup was needed.
