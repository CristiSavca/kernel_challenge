# One bounded attempt to remove observed WAIT2 register serialization

November 4, 2026. **Plan only, awaiting independent review before candidate
implementation.** Base is frozen WAIT2 `native_wgs_wait2_v12.py`, SHA256
`97ffccc30a90ce8400cddef800d932889822eaf876fbdedaec18b1013e8f66b6`.
The scored incumbent remains `native_wgs_kperm3_v9.py`, SHA256`247888acca…`.

## Trigger and falsifiable mechanism

Official private recovery job`f685c2533ebc4214` passed correctness but scored
2.331989× versus the incumbent's2.373691×. Across the three changed MAIN cases,
the geometric mean of incumbent-speedup/candidate-speedup was1.047569, a4.7569%
increase in inverse SGLang-relative speedup. Direct harness median times rose
2.0231%, and raw median times rose3.0092%. These are distinct quantities from
different GPU runs; unchanged small controls had1.1084% higher speedup. The
reported aggregate confidence intervals overlap. Crucially, the retained
compiler log explicitly reports **C7511: insufficient register resources
serialize WGMMA** for both BIG specializations. Both report96 registers, zero
stack/spill bytes, local0/static0 and occupancy1. Zero spills did not establish
the intended overlap. Parent compiler evidence is insufficient to claim WAIT2
introduced the serialization or that it alone caused the regression.
See `experiments/iterations/20261104-wgs-wait2-official-analysis.json` and the
complete `97ffccc3….f685c2533ebc4214.log` under the retained MoE profiles.

The single hypothesis is that full-warpgroup register redistribution can remove
this observed compiler restriction while retaining WAIT2's four AF banks and
wait2 schedule. Select producer32/consumer112,640 launched threads, initial96.
No quota, bank, wait-depth, tile or producer rewrite sweep is authorized by this
plan. Failure to eliminate the diagnostic or useful measured improvement ends
this chosen attempt.

## Exact bounded source changes

1. Change only WGS's additional launched threads64→128. Compute threads stay512;
   tiny/MID thread counts and every define tuple stay unchanged. Add WGS-only
   static assertions for NT512,NTL640 and the chosen fixed quotas.
2. Keep the existing kernel, common initial `griddepcontrol.wait`, and inlined
   WGS bodies. Give the WGS role dispatch an explicit producer `if` and consumer
   `else`. `setmaxnreg.dec.sync.aligned.u32 32` is the first operation inside
   the producer branch. All128 producer threads execute it before any role
   split, wait or return. Warp16 calls the unchanged weight producer; warp17
   calls the unchanged activation producer; warps18/19 execute `WS_JOIN()`
   once then return. The active producers retain their existing join and return.
3. `setmaxnreg.inc.sync.aligned.u32 112` is the first operation in the consumer
   `else`, which encloses the entire unchanged consumer body through routing,
   gather, both phases, PDL trigger, grid barrier, cleanup and finalizer. Use
   volatile inline PTX with a memory clobber. Close this branch immediately
   before kernel exit. No producer path reconverges into live consumer work.
4. Keep `__launch_bounds__(NTL,1)`. Do not add a hard `maxrregcount96` clamp, new
   outlined consumer function or auxiliary launch. Preserve WGS forceinline
   annotations to avoid introducing call-induced serialization.
5. Extend the existing WGS host resource screen, before either ordinary or PDL
   launch: both actual `numRegs ==96`, both `localSizeBytes ==0`, both actual
   occupancy≥1 at640threads and the unchanged231,168 dynamic shared bytes.
   Preserve failure as a hard error; do not choose another quota or fallback.
   Print actual registers/local/static/occupancy and intended32/112 quotas in
   the one-time resource marker. Keep WGS-only verbose build diagnostics.
6. Use one unique extension namespace and accurate provenance comments. All
   Python compute/packing/dispatch/scratch code, build flags other than the
   namespace, and non-WGS preprocessed CUDA remain unchanged.

This structure gives ptxas a consumer region dominated by one112-register
acquisition and producer regions dominated by one32-register decrease. It
avoids the prior attention design's conflicting quotas at a common math merge.
It does **not** establish that ptxas will honor the hints or remove C7511.

## Why this may compile, and why it may still fail

[NVIDIA PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)
defines a CTA-local pool, blocking acquisition, full-warpgroup participation and
absolute quotas. The published [pinned ForgeKernel Hopper implementation](https://github.com/OpenBMB/ForgeKernel/blob/312f20f31b2909cf05dfdde6016fd12f7942ed51/kernels/gemm/gemm_fp16/h100/kernel.py)
uses a disjoint-role dec40/inc232 structure, an initial168 ledger and an explicit
one-block launch bound. Its compiled-resource claim is a concrete precedent,
not a measurement of this kernel.

[CUTLASS issue2904](https://github.com/NVIDIA/cutlass/issues/2904) is a material
limit: NVIDIA contributors state instructions must be compilable under the
initial register budget. Dynamic quotas do not bypass an instruction's birth
budget requirement. Our existing instructions already compile at96: the widest
WGMMA uses32 FP32 accumulator words, four register-A words and a64-bit descriptor.
This distinguishes the issue's fatal large-operand compilation failure from our
observed pipeline serialization, but does not prove the scheduler can use112
for the live pipeline. The larger region budget may help preserve simultaneous
AF/accumulator/deferred-flush state; only actual compiler evidence can decide.

Launch bounds are an upper constraint and optimization convention, not a
guarantee of initial96. The actual attribute guard is mandatory. Initial96 gives
640×96 =61,440 CTA registers, exactly512×112+128×32. Donation and acquisition
are both8,192. Do not borrow from the SM's remaining4,096 or treat padding as
owned quota. The pinned NVIDIA-header CPU proof confirms initial96 admission
and rejects97. It proves total allocation, not the actual assignment of role
warps to hardware register partitions. See
`experiments/iterations/20261104-wgs-setmaxnreg-budget-critic.json`.

Producer32 remains a risk: address/range state and the noinline `wgs_ranges`
integer-division body may spill even though neither producer has large arrays.
Consumer112 may also remain insufficient around FCI3. Do not modify these
bodies or loosen the zero-local screen to rescue this experiment. The attention
96/112/32 results demonstrate that legal conservation and positive occupancy do
not prove zero spills, retained overlap or a performance gain.

## Progress, traffic and source proof

The full producer WG donates before it can wait for routing or consumers.
Consumers may block acquiring registers; the independently ready producer can
supply all four acquisitions. Then the inherited route/range/gather dependencies
run. Barrier2 widens to640:512consumers+32weight+32activation+64idle. Barrier7
stays544, the compute barrier stays512, and each consumer-WG barrier stays128.
No later register handoff is proposed. Idle warps do not enter barrier7 or the
activation producer, and no new barrier is placed after they return.

The plan CPU model `tools/audit_wgs_register_roles.py` covers all120 initial WG
request orders, exact role membership and joins. Negative controls reject a
partial64-thread producer WG, an initial88 pool and missing idle joins. Its
fair-progress assumption is explicit; it is not a CUDA scheduler simulator.

After implementation approval, the actual-source verifier must establish:

- Exact WAIT2 `wgs_seg`, phase, gather, producers, routing, finalizer arithmetic
  and scratch source identity; all AF banks, wait2/ci1 release, NTT1 parity,
  FCI3 and terminal wait0 remain byte-identical. Reuse the native deferred-AF
  reference trace against the candidate once; do not substitute a constants-only
  model for source conformance.
- Non-WGS preprocessing equality for the same sixteen existing control builds;
  Python AST equality apart from CUDA text and extension namespace. Source
  restoration removes only the explicitly listed role/NTL/guard edits.
- Preprocessed WGS role boundaries give every thread exactly one role, one
  matching quota action and the required barrier2 participation, including zero
  jobs and empty local ranges. Extract the actual role dispatch into native CPU
  stubs for all640 thread IDs; record order so no wait precedes producer donation.
- Both host specialization guards dominate both launch paths. Shared layout,
  tile/ring sizes, grid, fixup slots, counters and scratch allocation are unchanged.

Weight/activation bytes, TMA stages, routed jobs, MMA order and numerical
precision stay unchanged. The two extra warps add prologue/dec/join work only;
there are still sixteen compute warps and no larger ring lookahead. The potential
benefit is removal of observed serialization, not additional compute occupancy.

## Exact review and result gates

Require independent exact-source and CPU proof review before root submits one
official private check. Root owns GPU use, receipts and polling. Preserve complete
compiler output: inspect **C7511**, other WGMMA serialization/call warnings such
as C7510/C7512, dropped-setmax hints and spill reports. Absence of warnings in a
truncated log is insufficient. Inspect retained USETMAXREG in PTX/SASS only if
official artifacts expose it; never spawn inspection tools inside a submission.

Initial96, local0 and occupancy1 are necessary admission screens, not proof of
post-increase quota or asynchronous execution. Retained C7511 in either BIG
specialization kills this mechanism regardless of timing. Compilation/resource
failure also kills this chosen allocation. No automatic retry or quota variant.

If the compiler screen succeeds, all official correctness tests must pass and
the changed M32/M128/M256 cases must have geometric mean
`incumbent_speedup / candidate_speedup <=0.95`, with no case ratio above1.03 and
stable unchanged tiny/MID controls. This is the inverse SGLang-relative speedup
screen; separately report direct harness/raw median time ratios and variation
between GPU runs. Compare to failed WAIT2 too to assess this
specific revision. This is a screening criterion, not a prediction. A useful
standalone result still needs separate decoder integration and verification;
it does not establish #1 or a recoverable stall-percentage forecast.
