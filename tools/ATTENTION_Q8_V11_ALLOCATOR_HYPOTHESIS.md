# V11: one allocator-visibility test after V10 resource rejection

V10 is rejected. The terminal local screen is
`experiments/modal/20261104T060813Z-37b2e369`; it did not establish candidate
correctness or performance. Only the parent completed an eager oracle check.
The candidate warm call was enqueued and its target dispatch observed, then
the resource/serialization gate stopped before candidate synchronize/check.
Sandbox termination is confirmed. Preserve all original files and evidence.

The exact changed `attn_kernel<8,true,0>` compiled with C7512: WGMMA was
serialized for insufficient register resources. Its compiler report has a
296-byte stack frame,1128 bytes spill stores and864 bytes spill loads. Runtime
attributes report296 local bytes,80 initial registers,114688 dynamic shared
bytes and two resident CTAs. Parent has zero stack/spills/local bytes with the
same initial registers/shared memory/occupancy. Thus occupancy alone does not
validate the proposed register ownership.

## Confirmed source pressure and unresolved compiler cause

The PV worker owns `o[2][32]` (64 FP32 values), `l[2][4]` (8), and
`final_offset[2][2]` (4):76 persistent FP32 scalars before descriptors,
addressing, loop state and rescale temporaries.112 registers would leave36
for the latter; a96-register allocation would leave20. QK owns32 score
scalars and four persistent offsets, with conversion/address/control state.
These source counts are lower bounds, not a ptxas allocation prediction.
The296-byte stack size does not identify which arrays or live ranges spilled.
Neither PTX nor SASS was retained, so the exact allocator cause is unproven.

The concrete control-flow issue is visible in frozen V10: the96/112/104
`setmaxnreg` branches reconverge before one `run_segments<ROLE2>` call. The
QK/PV split occurs much later inside `q8_role_body_v10`, after the common
segment prologue. A compiler may lose the association between the PV body
and its112-register region at that merge. This is a falsifiable hypothesis,
not a claim that ptxas demonstrably imposed96 registers on PV.

NVIDIA specifies a per-CTA register pool; `inc` can block for available
registers, newly acquired registers require initialization, and each warpgroup
must execute the same instruction under a uniform condition. V10's role
conditions satisfy that requirement; no semantic undefined behavior is inferred.
The documentation does not establish how ptxas merges differing budgets.
[NVIDIA PTX setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg).

## Exact V11 change

V11 passes compile-time ROLE3(QK) and ROLE4(PV) through the existing
`run_segments` lambda into `segment` and the templated role body. Each
`setmaxnreg` directly encloses its corresponding specialized segment call.
The body selects QK/PV with `if constexpr`; their actual functions are byte
identical to V10. ProducerROLE1 remains32 registers, QK96, PV112; fallback
ROLE2 retains the original104-register path. There is no per-segment register
release/reacquisition, extra storage, changed arithmetic, changed wait/arrival
count, or new launch specialization.

The initializer formerly guarded by `ROLE==2 && tid==0` also admitsROLE3, so
exactly one consumer thread initializes the original full/empty ring. All256
consumers execute the unchanged prologue and three256-thread body barriers;
the final CTA drain still meets the producer. The common flat finalizer remains
after the role branch. PV accumulator initialization/fences remain downstream
of its112-register acquisition. K/P/V lifetime and per-half rescale proofs are
unchanged. The pool remains `(96+112+32)*128=30720`, equal to384×80 at entry.

Frozen core SHA256:
`7b57f0f8a1936a818ac6a399c047d9a1739e3c865070621a57afdd9e420cc5ed`.
Frozen attention wrapper SHA256:
`6724284b1e8aef03a3af38d1d9d008c1df680b241a37b2d688464b8eeb60633f`.
Parent remains the exact decoder attention wrapper `dc6d0c4884dba218...`.
The frozen V10 core and wrappers remain unchanged. The runner adds a fixed
`attention-v11-screen` source selection and reuses the reviewed V10 diagnostic
protocol. Its retained JSON key `attention_v10` names that protocol;
`candidate_revision: v11` identifies the tested source revision explicitly.

`tools/build_q8_role_pipeline_v11.py` proves exact reverse restoration of seven
listed substitutions to frozen V10. The verifier additionally checks byte-exact
QK/PV math and segment scheduling, exhaustive CTA role mappings, unique ring
initialization, shared finalization,8192 P-byte coordinates,109312 modeled
barrier events,128 rare rescale moves and256 fast halves. Both new role templates
are instantiated by the CPU host syntax check; exact113664-byte shared layout
is preserved. These checks do not compile CUDA/PTX or prove GPU correctness.

## Decisive next gate

One bounded compile/screen is the causal test. Keep the resource gates:
localBytes0, no target C7512/WGMMA serialization, actual two-CTA occupancy,
unchanged shared memory and observed target activation. If spills or target
serialization remain, reject this allocator-visibility revision; do not raise
register budgets, relax gates or infer that it needs parameter tuning.

If those conditions pass, all fresh-input eager/graph correctness checks and
paired timing still must pass. There is no measured speed forecast from the
control-flow edit. A useful target gain is required before whole-decoder or
official integration. Root owns any allocation; this implementation was CPU-only.
