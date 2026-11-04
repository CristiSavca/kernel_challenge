# Structural reassessment after dense V9 and GS4

2026-11-04. Read-only reassessment while root runs the reviewed early-A screen.
Parent is exact decoder SHA
`80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`.
No kernel candidate, GPU job, or official upload was made for this proposal.

**Decision: do not start a generic small-token dense fusion or shared-memory
dequantization rewrite from the current evidence.** If early-A is also null,
the next useful expert step is one bounded production-kernel counter capture,
subject to tooling/provider availability. This is a stop on these speculative
implementations, not a claim that the decoder is optimal or that the #1 goal
is complete.

## Why small-token fusion does not provide the missing large lever

In the actual incumbent, the small-token runtime is already substantially fused:

- `_uv_on`, `_uv_o`, `_uv_pooler` (lines 15228–15291) use the O projection's
  epilogue to produce paired U/V rows and boundary sums; the fused pooler/router
  consumes those rows. T<=8 avoids a separate `_boundary1` launch.
- `_after_attention` (14862–14974) selects that UV path, the fused pooler/router,
  and `UB2` for T<=8; UB2 combines the unpooler slice, final residual boundary,
  and next normalization. The old source also contains a measured-no-gain
  O/boundary fusion for the larger small-token band (`SK2B1=False`, 12735).
- The scored `4b216098993e4112` profiles show seven launches at T1/T8. Dense
  kernels already use lossless 12-bit BF16 packing with escape corrections
  (9192–9198), rather than an untouched 16-bit weight stream.
- Dense V9 selected zero changes after 31 timed alternatives; it did not test
  the existing T<=8 UV/UB2 paths. That leaves those paths open to research but
  does not turn their already-fused boundaries into new opportunities.

At T16, separate boundary1 takes 3.7/4.2 us in sliding/full profiles. Even deleting
both rows entirely saves only 2.29% of that weighted load and approximately
0.386% normalized latency across the six-load geometric mean. This deliberately
optimistic calculation uses profiler sums; it is not measured removable time.
It cannot justify a major rewrite toward a 14.76% aggregate latency gap.

## Distinct expert architecture considered, and why it is not ready

Candidate concept: use a dedicated warpgroup to unpack/scale NVFP4 weights into
shared BF16 tiles, then let the four consumer warpgroups issue shared/shared
WGMMA. This would decouple dequantization scheduling from the current register-A
operand lifecycle. It is different from changing commit-group size, readiness
order, or early-A publication.

The incumbent's `wgs_seg` (8089–8225) already overlaps decoded operand preparation
with one outstanding MMA group. The alternative preserves every packed weight
byte and every unpack/scaling operation, adds shared stores and tensor shared
reads, and concentrates dequant work into 128 threads instead of 512. Its
producer therefore needs a demonstrated service-time model, not an assumption
that moving work makes it disappear.

Resource arithmetic for the existing 256-output-row job geometry:

| Buffer | K=256 | K=128 |
| --- | ---: | ---: |
| Packed codes+scales per stage | 36 KiB | 18 KiB |
| Decoded BF16 weights per stage | 128 KiB | 64 KiB |
| Maximum activation tile per stage, 64 tokens | 32 KiB | 16 KiB |

K256 cannot double-buffer decoded weights within Hopper's shared-memory budget.
K128 with two packed, two decoded, and two activation buffers needs 196 KiB,
plus metadata. A host ABI reproduction of the incumbent's `Smem` gives 22,240
bytes, rounded to 22,272: about 217.75 KiB dynamic shared memory in total, before
any new barriers/bookkeeping and the existing 1 KiB static allocation. This is
plausible for one CTA, not two; CUDA layout/launch limits remain to be verified.
K128 also doubles stage and copy/barrier frequency. Lower consumer register use
does not buy occupancy while shared memory still limits the kernel to one CTA
per SM.

GS4's unchanged REG96/STACK0/LOCAL0 reports and negative timing do not prove
dequantization free or diagnose a spill. They do weaken a simple fewer-commits
story. The independent critic rejects this SS-WGMMA rewrite without evidence
that exposed consumer issue/decode dependency is the limiter and that a producer
can feed all four consumer warpgroups.

Scope matters: this rewrite initially affects WGS, hence the M32/M128/M256
loads, not all six loads. Using their approximate expert shares 40/55/54%, a
20% expert latency cut would yield about 5.10% aggregate latency reduction
(2.266× decoder); 40% would yield 10.51% (2.403×). These are conditional Amdahl
estimates, not forecasts. The saved standalone routes have U=16–27 and J=16–29;
neither a full 32-expert read nor a maximum-CTA wait ratio is a valid savings
assumption.

## One bounded measurement that could change this decision

Root has now verified `/usr/local/cuda/bin/ncu` and `/usr/local/bin/nsys` exist
in the current Modal image using a read-only path lookup. No profiling ran
during the paired experiment. Versions and hardware-counter permissions remain
unknown. Only if root chooses to proceed:

1. Preflight `ncu --version`, available sections/metrics, and counter access.
   If unavailable or denied, record that result and stop; do not change driver
   permissions, allocate repeated probes, or substitute unverified estimates.
2. Profile the exact incumbent WGS kernel for one representative M32 and M256
   input, with actual returned decoder routes if a whole-decoder capture fits
   the existing 900-second allocation cap. Save all local expert counts, U, J,
   weight/input seeds, extension flags, and hashes. A standalone-only capture
   must retain that limitation; it cannot establish the decoder's bottleneck.
3. Request only `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, and
   `MemoryWorkloadAnalysis`, plus source correlation only if needed to identify
   unpack versus barrier/atomic instructions. Exact metric names must come from
   the installed tool. Questions are fixed: actual DRAM bytes/utilization;
   eligible/issued warp behavior; integer/decode versus tensor activity; shared
   traffic/bank conflicts; and whether observed stalls belong to operand
   preparation or release/activation dependencies.
4. Keep profiler execution outside benchmark decisions. Kernel replay can
   serialize launches, change cache conditions, and control clocks. A serialized
   kernel capture cannot measure original PDL overlap. Preserve an ordinary
   unprofiled graph timing separately; do not report a profiler duration as an
   A/B speedup or divide summed stalls into an end-to-end savings claim.

Predefined interpretation: near-saturated DRAM with expected actual byte traffic
rejects a same-byte SS rewrite as the route to a large expert reduction. An
already saturated decode/ALU pipeline also rejects merely relocating the same
ALU work. Only evidence of exposed operand-lifecycle dependencies with useful
memory/issue headroom keeps this concept alive, and it still requires a concrete
producer-throughput design before implementation. Inconclusive counters do not
authorize a broad rewrite or additional counter fishing.

NVIDIA documents that stall percentages are meaningful when schedulers cannot
issue, and explains replay, serialization, cache, and clock effects in its
[Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
Hardware-counter availability is provider-dependent; see the official
[counter-permission guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute).

Evidence: `experiments/iterations/20261104-dense-chain-v9.json`,
`experiments/iterations/20261104-moe-ntt1-gs4.json`,
`experiments/modal/20261104T043758Z-24108d21/result.json`, and the exact scored
decoder profiles beside frozen `80a4…`. The immediate next performance result
remains root's in-flight early-A screen; this note requests no allocation.
