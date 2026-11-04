# Small-token WGMMA replacement: one geometry, no implementation

November 4, 2026. **NO GO for implementation with current evidence.** A compact
WGMMA replacement is technically plausible, but this audited geometry has no
supported advantage large enough to justify the rewrite. No candidate, parameter
sweep, GPU run or submission was created. The separate REG112 result does not
measure this design.

## Prior work and exact baseline

The baseline is scored standalone247888 and decoder97af. Their M1/M8/M16 paths
use `mma.sync.m16n8k16`, eight compute warps split across K, a helper warp,
16-output-row/full-K4096 weight units, three36KiB weight slots, and jobs of at
most eight tokens. MID splits experts above eight tokens into more jobs. Source
comments calling the unused WGS macro a tiny build are historical; active small
dispatch uses NW8/WS, whereas MAIN WGS starts above16 tokens.

A search of local candidates, research/iteration records and the preserved
donor ledger found no measured candidate matching the geometry below. Closest
negative evidence is the donor's WGS2/dual-lane WGMMA experiment, described as
8–20% slower from extra synchronization, without enough preserved implementation
detail to equate it to this design. Its published [lessons](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/LESSONS.md)
also warn that small-MoE ramp, handoff and tail costs can dominate. These are
donor observations, not measurements of our current binaries.

Our compact WGS rewrite affected only large loads and changed stage/activation
counts; rejected NW4 kept warp MMA and doubled each thread's activation cache.
Neither directly tests this proposal. Their failure also supplies no affirmative
evidence for a new small WGMMA rewrite.

Saved baseline tiny/MID resources are144/148/149 registers, local0,288threads
and214,784 dynamic shared bytes. These are source-equivalent standalone small
paths; decoder M1 has different MULDEC/FINPF flags, so its register count is not
established. There is no retained small-MoE NCU attribution showing a removable
tensor-issue, partial-reduction or helper bottleneck. The M32/M256 counters must
not be transferred to these shapes.

## The one geometry assessed

Use288threads: two128-thread consumers and one32-thread helper. Both consumers
operate on the same64 weight rows and eight padded token columns, each owning
K2048. A36KiB stage holds64×512 weights for each WG; four stages cover full K.
Activations move from the current64-register/thread cache into one full64KiB
shared cache per CTA/job:8×4096 BF16. Retain three weight slots, SKT8 job splitting,
all routes and one CTA/SM. No separate global activation repack or split-K
fixup grid is proposed. Register-A dequantization must preserve exact NVFP4
E4M3 scales and BF16 operands; the token padding remains eight, as in the parent.
The [PTX BF16 WGMMA shapes](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)
start at m64n8k16, explaining the larger minimum output tile with weights on M.

| Property | Current small path | Audited WGMMA geometry |
| --- | ---: | ---: |
| Useful compute/helper warps | 8 /1 | 8 /1 |
| Output rows per complete unit | 16 | 64 |
| K ownership | 8 warp partitions ×512 | 2 WG partitions ×2048 |
| Weight copies per complete unit | 1 | 4 |
| Copy payload | 36KiB | 36KiB |
| A/B complete units per job | 512 /256 | 128 /64 |
| Total weight copies / packed bytes per job | 768 /27MiB | 768 /27MiB |
| Shared partial-store/read payload per padded job | 6MiB | 1.5MiB |
| Added logical shared activation-operand reads | 0 | 12MiB/job |
| Dynamic shared allocation | 214,784B, includes unused fat capacity | 212,736B |

The proposed allocation conservatively retains current36,608B rounded metadata,
including a four-slot partial ring of the same byte capacity, then adds110,592B
weight slots and65,536B activation cache. It fits one CTA, not two. Removing the
64 cached activation registers could ease compilation, but baseline has no
spills and shared memory still prevents another resident CTA. AF buffering,
descriptors and WGMMA epilogues replace some of the removed live state; no actual
register prediction or occupancy claim follows from counting source arrays.

## Benefits and costs, accounted together

Two K partials rather than eight reduce the logical partial-store/read payload
by4.5MiB per padded job. The mathematical reduction needs589,824 fewer FP32 adds
per full job. A redesigned helper could aggregate64 output rows and reduce
phase-B norm atomics256→64 per real routed row, but this requires an epilogue
change and is not an automatic property of WGMMA. Parent phase-A credits are
already aggregated per CTA/job at the end of its A work; they do not fall fourfold
merely because the output units get larger.

Against those savings, WGMMA's B operand adds12MiB of logical shared payload per
padded job:192units×2WGs×128k16 instructions×16K×8tokens×2bytes. That is a net
7.5MiB increase before activation-cache fill stores. These are mathematical
payload/request counts, not measured shared transactions or latency. Global
activation loads may remain close because each CTA still caches a touched job;
the saved analysis counts touches separately. It does not claim extra HBM bytes.
The shared cache needs a valid WGMMA layout, padding writes, publication and a
drain before job reuse; none is free or implemented here.

Weight bytes, weight copies, padded tensor work and useful compute warps do not
decrease. WGMMA offers asynchronous issue, but requires AF lifetimes and
commit/fence/wait management. The incumbent already has four independent
accumulator chains per warp. There is no evidence that replacing its issue
mechanism removes enough exposed latency to offset the new shared traffic.
K grouping changes, so FP32 mathematical terms/precision can be retained but
bitwise accumulation order cannot be promised.

Larger complete units also reduce output parallelism and coarsen the tail. With
one local job, A has128 active CTAs and B only64; the latter's busiest CTA handles
four weight stages versus the parent's two. At three jobs, B's maximum stages
rise6→8; at five,10→12. These are general valid histograms, not probability
estimates. In the six saved standalone M16 routes, J8 is11/10/11/8/10/10; the
J11 cases have B's maximum22→24 stages. Every weight byte is still requested,
but a complete64-row tile cannot be divided among more CTAs without introducing
a different design and global partials. No rescue geometry is proposed.

## Whole-decoder relevance and decision

Exact97af public profiles show small MoE sliding/full durations of29.2/22.5µs
at M1,51.8/44.9µs at M8 and73.8/65.8µs at M16. The critic's profile-share model
gives the following whole-decoder scenarios:20% less small-MoE time yields4.27%
aggregate reduction;25% yields5.37%;50% yields11.08%. Achieving5% requires about
23.32% less time in these small paths. Closing the observed14.13% leader gap
from this subset alone would require about62.70%, which is unsupported. These
are scenarios, not bounds or forecasts: PDL and separate profiler captures limit
attribution.

The audited design has possible resource and reduction benefits, but no evidence
for the required scale of latency gain, and concrete traffic/tail costs. Close
it without implementation. A later decision would need specific evidence of a
large removable small-path dependency/helper cost, not another quota/geometry
sweep or an assumption that WGMMA is intrinsically faster.

Reproducer: `python3 tools/analyze_small_wgmma_feasibility.py`.
Evidence: `experiments/iterations/20261104-small-wgmma-feasibility.json`.
It preserves source/route hashes, exact arithmetic and general/saved route
examples. The saved M16 routes are standalone evidence, not decoder routes or
an input-specialized dispatch rule.
