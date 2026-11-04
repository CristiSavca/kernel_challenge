# One deeper WGMMA queue, with unchanged WGS traffic

November 4, 2026. Plan only against exact scored standalone247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0.
The rejected NW4 source is not the base. No candidate or GPU work at this stage.

**Selected realization:** four compile-time AF banks, `ci & 3`, GS2 and
`wg_wait<2>()`, with previous activation-slot release after ci1. Keep the same
576-thread, one-CTA/SM launch, FT64,256-output-row/K256 stages, three weight/x
ring slots, weight packing, KPERM, mathematical instruction order, routing,
costs, epilogues and finalizer. No bank/depth sweep is proposed.

## Mechanism and prior work

The parent commits two k16 MMAs per group and immediately waits until at most
one group remains. The proposed wait leaves up to two groups, allowing decode
and issue of another group before waiting for the oldest outstanding work.
The actual latency benefit is unknown. Weight/activation bytes, TMA stages,
WGMMA operations, commit/wait instruction counts and all job counts stay equal.
Three slots still provide the same weight lookahead; activation-slot reuse is
delayed by one commit group, which could reduce producer lead.

The previous NTT1-only GS4 experiment changed group size and the NTT1 accumulation
partition. It passed80 checks but regressed M32/M128/M256 paired speedup to
0.981728/0.967743/0.952977. Its main binaries reported96 registers/local0. That
does not prove this different queue-depth change succeeds, or that wide-NTT
variants can carry more live operands. No prior wait2/AF3/AF4 WGS implementation
was found in local candidates, archived donor sources or research records.

The recovered M32/M256 counters show8.11/9.58% `stall_wait`, about59.4% issue
active and74–75% DRAM throughput. Those wait counters are fixed-latency execution
dependencies, not an identified WGMMA wait cost; NVIDIA separately describes
warpgroup arrival/wait stalls. No source-PC attribution was captured. These
percentages are neither additive headroom nor recoverable latency. See the
[Nsight profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons).

## Why four banks, and precise source changes

A literal three-bank `ci % 3` resets incorrectly after eight groups per stage:
the next stage overwrites bank0 while the previous stage's group6 may still
read it. Carrying a three-bank phase would require dynamic array indexing or
three specialized stage bodies. We reject both for this bounded experiment.

Four banks divide eight groups exactly. Before decoding group q, the preceding
wait2 leaves only q−2 and q−1 live; the bank's previous owner q−4 has retired.
At the next stage, group0 uses bank0 while old groups6/7 use banks2/3.

The planned changes inside WGS `wgs_seg` are:

1. `af[2][GS][4]` becomes `af[4][GS][4]`; replace only AF index `ci & 1` with
   `ci & 3`. Keep the first write's bank0 reference. Assert GS2 for this version.
2. Change the per-commit `wg_wait<1>()` to `wg_wait<2>()`.
3. Move previous-x release from ci0 to ci1. After ci0, old group7 can still read
   the previous x slot; after ci1's commit/wait2, both old groups6/7 are complete.
4. Preserve every NTT1 `ci & 1` accumulator decision and first-write predicate;
   preserve current `d`/`d_odd` arrays and the final generic addition. The new AF
   bank selector is independent of mathematical accumulation parity.
5. Preserve FCI3, segment-end `wg_wait<0>()`, all epilogues and fixup stores.
   Add only a new extension namespace and WGS-only host resource diagnostics/
   positive occupancy admission check; keep all small/fallback build flags.

PTX specifies that waitN completes all groups older than the newest N, and that
generic access to live register-A/accumulator fragments is invalid. Same-shape
WGMMA accumulator accesses have the required default ordering. The retained
wait0 therefore precedes generic d/d_odd access, split partial stores, epilogues
and final x release. See [PTX WGMMA wait/fence semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group).

## Lifecycle proof and limits

The plan model `tools/audit_wgs_wait2_lifetimes.py` passes13,872 cases,
3,662,208 commit groups and53,856 deferred-flush boundaries. It covers all
NTT1–8, segment lengths0–16, all three initial ring slots and both generations,
partial/full/tail sequences, exact K order and NTT1 accumulator assignments.
It retains exactly two outstanding groups after every wait, the conservative
legal completion case. All three negative controls fail as intended: two
banks/wait2, three banks reset per stage, and the old ci0 x release.
Evidence: `experiments/iterations/20261104-wgs-wait2-lifetimes.json`.

FCI3 flush operates on a preceding segment that already drained with wait0;
it does not consume the current segment's live AF/accumulators. It can now run
while two current groups remain live, so the compiler must preserve their
resources across that code. This is a significant allocation/serialization
risk, not a logical data dependency. Four warpgroups retain their existing
sixteen total x-release arrivals, each after the contributing groups retire.

Three x slots preserve progress: while the producer waits to reuse stage q's
slot for q+3, q+1 was already issued and consumers can reach its ci1 and release
q. Segment-end wait0 also releases the last slot before the next segment/phase.
No new global wait or ownership dependency is introduced. The abstract model
does not emulate every CUDA scheduling interleaving or validate an actual
candidate; final source conformance and official tests remain necessary.

## Resource gate and bounded decision

**There is no established16-register headroom.** The critic checked NVIDIA's
CUDA13.0.96 occupancy implementation:576 threads require the hardware register
check for20 allocated warps, and97 registers round upward to66,560 registers,
exceeding65,536. The actual per-thread register admission cap is96, not the112
suggested by dividing total registers by576 threads. The parent already uses96.

Four declared AF banks contain32 words rather than16. The abstract live AF
peak is24 words (two committed groups plus one being decoded), versus16 in the
parent; neither number predicts physical allocation. NTT8 still needs32 FP32
accumulators plus code, scale, descriptor, pending and address state. The earlier
GS4 resource result involved32 AF words only in NTT1 and cannot prove this fit.
Spills, WGMMA C7512 serialization and outlining remain genuine failure risks.

If approved, implement only this standalone source. Require unchanged non-WGS
preprocessed output/AST scope, the actual-source AF and accumulator-index proof,
and the model rerun against extracted constants. Before launch, check both
compiled BIG specializations have occupancy≥1 at576threads and231,168B dynamic
shared; report register/local/static-shared values. Surface WGS build output
only so actual ptxas spill/serialization evidence can be reviewed in the official
log. Missing/truncated diagnostics do not prove absent serialization.

Root owns one official private check after exact-source review. Numerical tests
and every route must pass. Resource/serialization regressions must be reported;
no hidden fallback, geometry change or automatic rescue. The performance gate
is at least5% lower geometric-mean normalized latency across M32/M128/M256,
no affected case worse by more than3%, and stable tiny/MID controls. This is a
plausibility threshold, not a forecast. Any useful result still requires a
separately reviewed whole-decoder check; it does not establish a path to #1.
