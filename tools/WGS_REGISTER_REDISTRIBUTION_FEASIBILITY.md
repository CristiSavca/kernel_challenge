# Full producer warpgroup: bounded register-redistribution feasibility

November 4, 2026. Research/source audit only; no candidate, GPU work or submission.
Base is scored `native_wgs_kperm3_v9.py`, SHA256
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
WAIT2 `97ffccc3…` remains a separate pending experiment owned by root.

**Verdict: conditionally feasible, not implementation approval.** One full
producer warpgroup can support a 32/112 producer/consumer split if the compiled
640-thread kernel actually starts with 96 registers per thread and preserves
the redistribution instructions. This is a possible response to measured
compiler pressure in WAIT2, not an independent demonstrated bottleneck or gain.
Wait for that result before requesting a concrete implementation review. Do not
automatically combine mechanisms or sweep quotas.

## Published mechanism and actual initial allocation

PTX maintains a register pool per CTA. `inc` can block until enough registers
are returned; both actions set absolute per-thread quotas. All four warps of a
warpgroup must execute the same action under a uniform condition. Newly acquired
registers require initialization. Quotas are multiples of eight, from 24 to256.
The kernel needs a valid compile-time register convention; the instruction can
otherwise be ineffective. A later redistribution requires explicit warpgroup
synchronization. See [NVIDIA PTX setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg).

The pinned [ForgeKernel Hopper GEMM](https://github.com/OpenBMB/ForgeKernel/blob/312f20f31b2909cf05dfdde6016fd12f7942ed51/kernels/gemm/gemm_fp16/h100/kernel.py)
uses384 threads, producer40/consumer232, `min_blocks_per_mp=1`, and first-action
dec/inc in disjoint role branches after an initial CTA synchronization. Its
source ledger reports initial168 registers:384×168 equals128×40+256×232 =64,512.
This is a concrete published convention, including its stated resource record;
we have not reproduced its measurements. It is dense shared/shared BF16 GEMM,
with128 accumulator words per consumer, not our NVFP4 register/shared MoE.

CUDA launch bounds derive an upper register limit and permit compiler heuristics
below it; they do not guarantee a minimum initial allocation. Retain
`__launch_bounds__(640,1)` as the supported tuning convention, but do not infer
96 actual initial registers from that declaration. See [CUDA launch bounds](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#launch-bounds).
In [CUTLASS issue2904](https://github.com/NVIDIA/cutlass/issues/2904), NVIDIA
contributors also caution that instruction compilation must be possible under
the initial register budget. A larger post-increase quota does not guarantee
ptxas can compile an otherwise inadmissible WGMMA instruction.

## One selected arena, with explicit counterexamples

The critic compiled and called NVIDIA's unchanged CUDA13.0.96 occupancy header
on the CPU. At640 threads, reported96 registers allocates61,440 registers and
is register-admissible; reported97 rounds to66,560 and is inadmissible. This
excludes shared-memory, barrier and actual device/compiler checks. Evidence is
`experiments/iterations/20261104-wgs-setmaxnreg-budget-critic.json` and its pinned
header provenance; the reproducible tool is `tools/critic_wgs_setmaxnreg_budget.py`.

| Quantity | Selected value |
| --- | ---: |
| Consumer threads / producer threads | 512 /128 |
| Exact initial count required for this reasoning | 96 |
| Initial CTA allocation | 61,440 |
| Consumers at112 | 57,344 |
| Producers at32 | 4,096 |
| Producer donation96→32 | 8,192 |
| Consumer acquisition96→112 | 8,192 |

With four register partitions, each has four consumer warps and one producer
warp. The same arithmetic balances per partition:4×32×112+32×32 =15,360.
This budget is not borrowing the unused4,096 registers of the SM. Producer24 /
consumer120 requires64,512: it fits the SM total but exceeds this CTA allocation
by3,072. Producer40 / consumer112 also exceeds the pool, by1,024. Initial88 or80
does not supply the selected32/112 arena. Do not count allocation-granularity
padding as owned registers; require exact96 until another convention is proved.

For comparison, the original576-thread kernel's actual allocated-register field
is55,296 at96; its hardware admission check rounds to20 warps. These are distinct
quantities. Adding64 threads allocates6,144 more initial registers; reducing the
producer role's total from64×96 to128×32 frees another2,048 for consumers. This
explains the8,192 increase without claiming additional compute warps.

## Source shape and progress

Keep512 compute threads,256-output-row/K256 weight stages,FT64,three weight and
three activation slots, the one-CTA-per-SM grid, all packing, math, cost tables,
fixup slots, scratch offsets and finalizer. Only WGS launched threads grow
576→640; tiny/MID builds remain unchanged. Dynamic shared stays231,168 bytes.

The selected role structure is explicit:

1. Preserve the initial programmatic-dependency wait for all640 threads.
2. Threads512–639 execute one full-warpgroup `dec32` before splitting. Warp16
   runs the unchanged weight producer; warp17 the unchanged activation producer.
   Warps18/19 execute the widened barrier2 once, then return.
3. Threads0–511 execute `inc112` in their four complete warpgroups, before the
   entire consumer body. Preserve the producer branch's early return; avoid
   merging a low-budget producer path back into consumer gather/math/finalizer.
4. Barrier2 becomes640 participants:512 consumers after gather,32 weight threads,
   32 activation threads and64 idle producer threads. Barrier7 remains544
   (512 routing consumers plus32 activation producer threads); compute barrier1
   remains512; consumer warpgroup barriers remain128.

There is no new all-CTA barrier before donation. Consumers may wait for donation,
but producer dec has no dependency on consumer routing, gather or ring progress.
After donation the existing dependency chain is unchanged: consumers route and
arrive at barrier7; the activation producer computes ranges; all participants
reach barrier2; producers and consumers run their existing pipelines. Zero jobs
and empty local ranges still require these joins. Idle warps must not enter the
activation producer by falling through the old `else` branch. No later inc/dec
is proposed, so there is no second register handoff to synchronize.

The two extra warps perform launch/prologue, dec and one join only. The selected
shape preserves every weight/activation request, TMA stage, readiness arrival,
WGMMA operation and route job. It adds role-management work and potential inc
waiting; it does not increase the sixteen useful compute warps or ring lookahead.

## Resource uncertainty and stopping conditions

Producer32 is plausible but unproved. Neither producer carries a large value
array, but both retain stage/ring/range state and64-bit addresses. The activation
producer also carries job/base/row state and invokes the noinline `wgs_ranges`,
whose binary search and integer divisions have unknown compiled demand. A count
of source locals is not a register-allocation proof; uniform registers, reuse,
rematerialization, calls and spills matter. Preserve the producer computation
initially; do not hide a needed rewrite inside this feasibility proposal.

Consumers retain the current wide-NTT accumulator32 words. WAIT2 has up to24
logically live AF words, versus16 for the parent;32 are declared. The gather's
source inventory includes32 value words and sixteen words of destination
pointers, while the finalizer caches64 values. Those regions are not all live
simultaneously and cannot be added into a hard lower bound. FCI3 deferred flush
and branch/outlining behavior remain particular WGMMA serialization risks.

Our attention V10/V11 experiments already showed semantically balanced96/112/32
quotas can compile with substantial spills; exposing disjoint roles removed
C7512 but did not eliminate local storage or deliver speed. See
`tools/ATTENTION_Q8_V11_RESOURCE_OUTCOME.md`. A positive occupancy query is not
evidence the dynamic quota was honored or beneficial.

Any later implementation plan must specify exact initial96 and positive actual
occupancy checks for both BIG specializations at640threads/231,168 dynamic bytes,
complete WGS-only build diagnostics, absence of dropped-hint/serialization
warnings, and zero-local/spill screening against the incumbent. Inspect retained
USETMAXREG in PTX/SASS if official artifacts permit; absent artifacts leave that
mechanism unverified. Do not start a subprocess from the scored candidate to
obtain them. A host attribute is the initial kernel allocation, not a measurement
of each consumer's post-increase quota. Compiler failure or a failed resource
gate ends this chosen arena without an automatic quota rescue.

If WAIT2 is already spill-free with no serialization evidence, this audit alone
does not justify another official test. A later favorable compiler screen would
still need official correctness and a useful measured gain, followed separately
by a whole-decoder check. Nothing here establishes recoverable latency or #1.
