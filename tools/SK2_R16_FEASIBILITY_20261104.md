# SK2 R16 family: bounded feasibility, November 4

**Gate: plausible resource hypothesis for a separate implementation-plan review;
no implementation or GPU approval.** Shared-memory feasibility and exact router
coordinate coverage pass. Actual new registers, admission and latency remain
unknown. There is no measured parent bottleneck: the diagnostic's 19 SK2 records
all have zero reported local bytes. Increasing CTA count alone is insufficient
evidence of a gain, especially because several layouts reduce total weight bytes
held by their rings and increase activation requests.

Evidence is exact scored decoder `97af3994…`, diagnostic `e444f46f…` / private
`c0b4201c5ad345d1`, and the 97af public profile `7a8c4941741645ee`.
The diagnostic records are first post-prepare observations, not benchmark phase
or weight identities. CPU arithmetic: `tools/audit_sk2_r16_feasibility.py` and
`experiments/iterations/20261104-sk2-r16-feasibility.json`.

## One fixed family

All rows use R16/NT2/NWR1 and unchanged lossless 12-bit tile units. NS is fixed;
one additional ring stage fails the two-CTA shared budget for each listed layout.
The dynamic estimates use the diagnostic's table bounds: pooler R16 is a subset
of old R32, hence pmax≤56; QKV R16 can overlap two old R24 blocks, hence pmax≤88
from old pmax44. These bounds apply to those recorded weights, not all future
weights. An actual implementation must account for its prepared pmax.

| Use / physical TOKR | WSK×UPW | NS | consumers + producer + escapes | threads | dynamic bytes at bound | allocated bytes with reserves | maximum registers for 2 CTAs |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| A fused pooler 8 | 8×1 | 4 | 8+1+4 warps | 416 | 104352 | 106496 | 72 |
| A fused pooler 16 | 8×1 | 3 | 8+1+3 warps | 384 | 103248 | 105472 | 80 |
| A fused pooler 32, tuner compatibility | 8×1 | 2 | 8+1+2 warps | 352 | 104320 | 106496 | 80 |
| B separate pooler 32 | 4×2 | 2 | 4+1+2 warps | 224 | 96128 | 98304 | 128 |
| C QKV 8 | 4×2 | 5 | 4+1+4 warps | 288 | 108720 | 110848 | 96 |
| C QKV 16 | 4×2 | 3 | 4+1+3 warps | 256 | 91216 | 93440 | 128 |
| C QKV 32 | 4×2 | 2 | 4+1+2 warps | 224 | 96384 | 98560 | 128 |

Allocated planning bytes include a conservative 1024-byte allowance for new
static shared storage, a 1024-byte driver reserve and 256-byte rounding; they
remain below half of 233472 bytes. This is not an actual function query. Parent
fused static shared is 384 bytes; its actual registers are 88 at TOKR8 and 87 at
TOKR16. NT4→NT2 halves the persistent pooler accumulators from 32 to16 FP32 words
at MT1/NACC2, making lower allocation plausible, but does not prove the required
72/80 register limits. Router/finalizer temporaries may set the peak instead.

The register thresholds come from the critic's native CPU execution of NVIDIA's
unmodified CUDA13.0.96 occupancy header, including 256-register warp allocation
and SM subpartition limits. In particular, 352 threads require ≤80, not the raw
aggregate estimate88; 288 threads require ≤96, not112. See
`experiments/iterations/20261104-sk2-r16-register-admission-critic.json`.
The actual compiled specialization must still be queried for admission.
Query each reachable PK specialization separately: ragged M24 and full M32, for
example, compile different kernels. A tuner-caught launch exception is not
resource evidence. Any future gate must retain bounded attributes/admission
before throwing and account for unobserved specializations explicitly.

## Router and state lifetime: uniform eight-way policy

Use **eight-way reduction for every FUSE specialization**, TOKR8/16/32, including
preparation/tuner alternatives that are absent from the public profile. This
closes the shared-counter hazard without per-call resets or separate state sets.
Keeping even one old four-way FUSE user is invalid: a counter at4 modulo8 makes
the fourth new CTA incorrectly declare itself last.

The exact proposed expressions are `NSTR=2`, `rt=cta>>3`, `rq=cta&7`,
`gsr=rq*2+(st-nst)`, `pp=part+(rt*8+rq)*32*8`,
`pr=part+rt*8*32*8`, and last-arrival `(old&7)==7`.
256 CTAs cover32 router row tiles ×8 K pieces; each piece covers2 stages of512 K.
The CPU audit enumerates all32×16×8 tile/stage/unit coordinates exactly once,
matching the original four-way domain. Router packing stays unchanged:
NW=8, WBYR=16384, and stream stride `32*WBYR`.

The pooler WBY shrinks24576→12288 bytes, below router WBYR. Therefore
`SST=align1024(XBY+max(WBY,WBYR))` must replace the old formula in **device layout,
host launch allocation and Python geometry/fit accounting together**. Main-stage
expect/copy sizes remain XBYr+WBY; router stages remain XBYr+WBYR. The unused
main-stage padding is never decoded. Remove the old WBYR≤WBY restriction only
after every allocation agrees, while retaining COMP/CL1/UPW1/UPS8 and NWR1 guards.

Both allocation sites (`bfp_fused_pooler_router` and `_uv_pooler`) must allocate
`part[32][8][32][8]` FP32, 256KiB instead of128KiB. The32 counters remain shared
between all uniform-eight-way calls and advance by8 per complete invocation.
Each CTA publishes its unique partial with the original store/barrier/release
atomic sequence; only the last CTA sums slots0…7 in fixed order. Other CTAs do
not spin. Retain the original acquire fence, cbar wait before ring reuse, consumer
named barriers, all escape-warp stage handbacks, and PDL waits/trigger placement.
The coordinate/counter audit is not a device memory-order proof.

For UV mode2, M is physical2T. Sum all eight u partials and all eight v partials
before `rs*(u+ra*v)`, preserving the existing output rounding points. Keep paired
rows2t/2t+1, output logical T, the eight boundary-statistics slots, and only cta0's
raout write. Every new CTA still derives the same ra/rs; this doubles those
redundant reads/calculations. O/UV generation, unpooler/BND geometry and BND's
actual occupancy admission stay unchanged.

Eight-way router summation changes FP32 association; reduced NESC changes escape
correction grouping, and B changes K-warp grouping2→4. Every BF16 weight, router
hi/lo term and correction remains present with FP32 partials. Bit-equivalence to
the parent is **not** claimed. A future implementation needs full numerical tests,
including near-tie routing, ragged physical M, UV and repeated graph invocations.

## Requests, work and lookahead

Pooler grids128→256; QKV96→144. At132 SMs the latter offers only1.09 CTAs/SM on
average, not two everywhere. Main tensor operations and logical weight payload
are unchanged. More CTAs distribute that work; they do not remove it.

* A: main weight-copy/TMA pairs2048→4096; each weight payload24→12KiB. Router
  pairs remain512. Main activation requests double; router requests are unchanged.
  Total stage pairs2560→4608. Empty-barrier arrivals30720→55296 at TOKR8 and
  →50688 at TOKR16. The producer is one32-thread warp with lane0 issuing copies;
  escape lanes are128 at NS4,96 at NS3,64 at NS2.
* A aggregate two-CTA main-weight ring capacity decreases: TOKR8
  `1*6*24KiB→2*4*12KiB`,144→96KiB; TOKR16 `1*5*24→2*3*12`,120→72KiB. Aggregate
  stage slots grow, but saying that prefetch capacity uniformly grows is false.
* A router partial stores double64→128KiB per MT1 launch, and counters128→256
  atomics. Final partial reads double, by4096 bytes per physical token. Main+
  router activation requests grow from2.5→4.5MiB per physical token. These are
  logical requests, not measured DRAM traffic; no cache-hit assumption is made.
* B doubles K/stage256→512, reducing stages/CTA32→16. Main copy pairs remain4096;
  activation requests double and empty arrivals32768→24576. K reduction has four
  partials instead of two. Two-CTA weight ring capacity falls60→48KiB. Per-CTA
  pooler reduction storage grows10→20KiB and still fits the ring.
* C keeps K/stage512 and16 stages/CTA; main copy pairs1536→2304 and activation
  requests rise50%. Launch-wide weight lookahead falls to5/6,3/5 and1/2 of the
  parent at TOKR8/16/32, respectively. Escape scans/grouping change with NESC;
  they remain the original shared-input mechanism, not the rejected global path.

## Research priority and next gate

On the earlier97af profile, a hypothetical20% reduction of QKV+pooler rows maps
to about3.20% decoder reduction; roughly31.0% row reduction maps to5%. These are
conditional profile-share calculations, not bounds or forecasts: PDL overlap and
wait time prevent direct attribution. Five percent is a prioritization threshold,
not a contest requirement. Unchanged weight bytes, extra activation requests and
smaller weight rings provide no basis to predict31% row improvement.

The substantive hypothesis is independent admitted pipelines/warps for the same
work, supported by the explicit one-CTA→potential-two-CTA resource transition and
smaller live pooler accumulator state. It is sufficiently distinct for a bounded
source-plan review, but **not** a reason to implement the entire family now or
promise a speedup. A future plan must choose one isolated causal scope, prove
uniform router state/packing/ring lifetimes, retain exact baseline controls, and
require actual resource admission before performance interpretation. If new
registers miss the72/80 critical A thresholds, do not force quotas or begin a
sweep; reconsider the mechanism. Any performance failure rejects that tested
scope rather than justifying a nearby-parameter retry.

The proposed admission gate distinguishes performance targets from compatibility:
require actual2 CTA/SM admission for A8/A16/B32/C8/C16/C32. Uniform-eight-way
FUSE32 must be correct, report its actual resources and have at least1 admitted
CTA with no local allocation. Capacity1 is permitted only when the terminal
timed profile does not select it; selecting FUSE32 at1 rejects the planned
mechanism offline. No candidate code detects benchmark phases or inspects the
harness. A private PASS alone never establishes this admission gate.

Primary hardware references:
[Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/),
[NVIDIA occupancy/register presentation](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/CUDA-Programming-and-Optimization.pdf),
and the pinned NVIDIA CUDA header URL in the critic's register-admission record.
