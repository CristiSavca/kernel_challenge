# Independent feasibility critique: two shared-A K32 banks

2026-11-04. Read-only review of standalone parent
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
This is a prospective architecture review, not approval of candidate source or
a GPU launch. Reproducer: `tools/audit_wgs_shared_a_feasibility_critic.py`; results:
`experiments/iterations/20261104-wgs-shared-a-independent-feasibility.json`.

Ownership correction: the first shared filename collided with the author's
script. The initially reported `d59b1e46` bytes were the author's ROI model,
not an independent reproduction. The separate critic script was reconstructed
from the independent model's original tool-call text, extended with store-demand
arithmetic, and rerun. The independent JSON now records this correction and its
actual script hash. It writes only the independent JSON; the author's files
remain untouched.

**No hard geometry or operand-layout defect found in the selected plan.** A
bounded implementation is technically defensible after the full plan review.
There is no demonstrated speedup. The trade exchanges live register-A pipeline
state for substantial shared-memory work and synchronization, without reducing
weight bytes, activation bytes, dequantization or MMA operations.

## What differs from the earlier rejected SS idea

`NEXT_DECODER_STRUCTURE_PIVOT.md` considered a dedicated 128-thread decoder
warpgroup, full decoded K128 stages and twice as many stages. This proposal keeps
the existing 512 consumer threads doing the exact decode, the 256-output-row/K256
units, FT64 jobs, GS2, wait1 and original KPERM packing. It stages only two K32
groups per consumer WG. It therefore avoids both that concentrated decode
service bottleneck and the compact rewrite's changed work/stage counts.
It does not decouple dequantization into a separate producer pipeline.

## Layout and math

For register index `r=0..3`, warp-within-WG `wp`, and lane `l`, the parent's
packed BF16 register holds matrix coordinates
`m=16*wp+(l>>2)+8*(r&1)`, `k=2*(l&3)+8*(r>>1)` and `k+1`.
The proposed no-swizzle K-major shared address is
`(m>>3)*256+(k>>3)*128+(m&7)*16+(k&7)*2`.
Descriptor LBO128/SBO256 addresses exactly those 64×16 BF16 elements, with a
2048-byte tile; two tiles per WG and four WGs give 16 KiB per bank. The exhaustive
prospective model finds 16,384 unique BF16 coordinates across both banks and
no scalar 32-bit shared-store bank conflicts within any warp/instruction.
It is not a tensor shared-memory transaction or compiler-instruction proof.

This mapping follows NVIDIA's
[BF16 register-A Figure 151](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-register-fragment-wgmma-64n16)
and [shared layout/descriptor definitions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-smem-layout).
SS BF16 WGMMA supports shared A and B; descriptor values must agree throughout
each warpgroup. Shared writes need an async-proxy fence. Register fences alone
do not publish shared operands. Completion must precede operand reuse and
generic accumulator access.
[PTX WGMMA and synchronization](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)

Store the exact `bmul2l/h` output words, with no conversion or alternative scale
format. Retain both scale-d initialization predicates, NTT1 even/odd accumulator
selection and final FP32 addition, all 16 k16 operations in their original order,
every NTT 1..8 specialization, epilogues and all routes. Equal operands/operator
order are not a GPU bitwise-output proof; numerical checks remain necessary.

## Lifetimes and forward progress

The author accepted two separate 128-thread barriers on WG-private ID 3+wg:
(1) before any store that reuses an A bank, after the preceding wait1;
(2) after each writer's shared stores and `fence.proxy.async.shared::cta`.
Keep the existing `wg_fence`, then issue two MMAs, commit and wait1.
The last outstanding group uses the opposite `ci&1` bank. Eight groups per
stage make bank parity reset safe. Retain segment-end wait0 before NTT1 sum,
fixup, epilogue and last-X release; retain FCI3 deferred flush and its existing
uses of the same WG barrier ID in identical collective order. No producer joins
these WG barriers. A post-store barrier alone is not the required pre-overwrite
barrier.

Joined weight/X rings both become depth 2. Keep both producer arrivals on each
full barrier and all 16 warp releases on empty barriers. Weight-slot release
must remain after every packed code/scale load. The previous X slot remains
live through next-stage group 0, whose wait1 retires it. Stage 1 can already be
issued while the X producer waits to recycle stage 0, so this schedule has no
obvious depth 2 circular wait. An implementation still needs an event model with
independent weight/X completion, all WG releases, skewed stage ranges, reordered
phase A, phase transitions, empty ranges and split-unit deferred flushes.
Global jobs/counters/fixup slots, PDL and 132-CTA admission stay unchanged.

## Exact cost and resource scope

Dynamic shared memory is 22,272 metadata +73,728 packed weights +65,536 X
+32,768 decoded A = **194,304 bytes**, 36,864 fewer than the parent's 231,168.
All offsets are 128-byte aligned and descriptor addresses remain below 256 KiB.
It still targets one 576-thread CTA per SM. Neither this arithmetic nor removing
the 16-word AF declaration establishes a lower physical register count: decode
temporaries, descriptors, accumulator/deferred-flush state, gather and finalizer
remain. Both actual compiled BIG entries need resource/occupancy queries and
valid launch admission, including register and driver/shared reservations.
Local allocation and compiler spill/serialization diagnostics must be reported;
they are not automatic rejection criteria for a valid, correct binary under
this fixed plan. Measured performance still decides whether it is useful.

Per FT64 job, 48 units ×16 stages = 768 joined stages and 1536 TMA operations
(weight plus X), unchanged. Packed weight requests remain 27 MiB. Each of 6144
groups stages 16 KiB A, adding **96 MiB generic shared stores and 96 MiB logical
tensor-A reads** per job. These are payload counts, not DRAM requests or measured
shared transactions. The two barriers add 24,576 WG publication barriers and
24,576 WG reuse barriers per job. A scalar implementation additionally has
25,165,824 32-bit thread-store operations per job. Raw-weight and X lookahead each
falls from three slots to two. Added LSU work/fences/barriers and reduced
lookahead can outweigh any improved register scheduling; this is a measured
experiment question, not a layout impossibility.

Parent C7511 status is **unknown**. Positive C7511 in failed WAIT2/REG112 binaries
does not establish serialization in this exact parent or removable latency.
Retain full compiler diagnostics and actual resources for any eventual check.
Do not turn sampled stall percentages into a speed forecast. Scope is MAIN/WGS
(official M32/128/256), while tiny/MID remain controls; a standalone gain would
still need decoder integration and measured whole-decoder confirmation.

## Additional store-service plausibility check

NVIDIA documents 32 shared-memory banks, each serving 32 bits/clock, giving an
idealized **128 bytes/SM/clock** for generic loads/stores without bank conflicts.
[CUDA Best Practices, shared-memory banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)
The retained selected-resource diagnostic reports 132 SMs. Exact 247888 public
receipt `d24dd50196cb4158` reports 1980 MHz SM clock before and after the run;
these samples do not establish the instantaneous clock during kernels. The
independent JSON pins both artifacts and also evaluates 1500 MHz explicitly as
a sensitivity parameter, not an observed clock.

Under this documented bank model, new A stores alone require ideal service
`96*2**20*J / (132*128*clock_MHz)` microseconds across the device:

| Local jobs J (scenario) | New store payload | At 1980 MHz | At 1500 MHz |
| --- | ---: | ---: | ---: |
|16|1.500 GiB|48.14 us|63.55 us|
|24|2.250 GiB|72.22 us|95.33 us|
|27|2.531 GiB|81.24 us|107.24 us|
|29|2.719 GiB|87.26 us|115.18 us|

The public standalone harness medians are 122.80/177.45/207.30 us for
M32/128/256; raw medians are 126.74/180.50/210.50 us. At 1980 MHz, J24–29 store
service alone corresponds to 40.7–49.2% of M128's harness time and 34.8–42.1%
of M256's; at 1500 MHz the respective ranges are 53.7–64.9% and 46.0–55.6%.
J16 corresponds to 39.2% of M32 at 1980 MHz. These job scenarios are not asserted
to be the routes in those official timing samples. Saved diagnostic routes and
official time measurements are different captures.

This is a substantial service demand, not an automatic extra 72–115 us of latency.
Stores can overlap tensor work, global-memory waits and other WGs. Conversely,
existing packed-weight shared loads, address instructions, proxy fences and
49,152 added WG barriers/job already consume resources, so the model is highly
optimistic about achievable generic-store service. It does not add logical
tensor-A reads or TMA writes to this 128 B/clock calculation: their port/path
interactions have not been established. Nor is this documentation-derived model
a measured Hopper shared-memory bottleneck or a proof that the work cannot fit.

Final review of plan `4df441db`: **GO for one bounded CPU implementation**, with
the stated layout/lifetime/source proofs and a later independent source review.
The added stores plausibly fit within the existing total time budget under ideal
overlap, but consume a large fraction of it; a useful measured improvement is
uncertain. Keep the chosen geometry and both barriers, the parent-source
uncertainty, actual launch-admission gate and the predeclared performance gate.
No GPU, candidate-source or promotion approval is implied by this plan verdict.
