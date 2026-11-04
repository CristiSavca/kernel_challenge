# Fixed shared-A WGMMA experiment — November 4, 2026

Concrete feasibility and implementation plan, awaiting root acceptance and the
independent critic. No candidate, GPU allocation or official check is authorized
by this file alone. Parent is exact scored standalone MoE
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
WAIT2/REG112 and failed R16 are not parents. Proposed isolated candidate:
`muse_moe_w4a16/candidates/native_wgs_shared_a_v15.py`.

## One mechanism and its actual uncertainty

Replace WGS register-A operands with two small shared-memory BF16 A banks. The
same512 consumer threads decode the same values, store them in each warpgroup's
private bank, and issue shared/shared WGMMA. This frees decoded operands from
register lifetimes spanning asynchronous WGMMA operations. It can change compiler
scheduling and register pressure without an extra dequantization warpgroup or
changes to jobs, K partition, weight bytes or tensor work.

Exact247888/ef7 serialization is **unknown**: their full compilation logs/SASS
were not retained. C7511 is positively observed only in failed WAIT2 and REG112;
it identifies compiler register-resource pressure, not its precise live range.
This plan does not claim a parent compiler bug or established serialization
bottleneck. The experiment tests the operand-storage tradeoff. New serialization
diagnostics are evidence to interpret, not an automatic ban on measuring a valid
correct binary. A faster result would not by itself prove asynchronous overlap.

This differs from the previously rejected dedicated128-thread dequantizer and
full decoded K128 stages: all512 consumers retain dequantization, K256 remains,
and only two K32 windows are materialized. It also differs from WAIT2's deeper
queue: GS2, wait1, accumulator chains and FCI3 remain the incumbent values.

## Fixed scope and exact resource layout

Keep256 output rows/CTA work unit, K256/stage,16 stages/unit, FT64, four consumer
warpgroups, two producer warps and576 launched threads. Keep jobs, ranges,
stream-K fixups, phase-A readiness, gather, KPERM packing, all scales, nonlinear
math, finalizer, PDL, precision, compiler arithmetic flags and every non-WGS
build unchanged. No register quota, launch-bound, FT/tile/NTT dispatch sweep.

Raw weight and X rings become two slots each. Append two decoded-A banks:
`2 banks × 4 WGs × 2 k16 steps × 64 rows × 16 K × 2 bytes =32768 bytes`.
Each WG owns4096 bytes per bank; each k16 tile is2048 bytes. The source-derived
allocation with existing rounded metadata is:

| Region | Bytes |
| --- | ---: |
| Original rounded `Smem` metadata |22272|
| Two raw weight slots, each36864 |73728|
| Two maximum FT64 X slots, each32768 |65536|
| Two shared A banks, each16384 |32768|
| Total dynamic shared |194304|

Parent dynamic shared is231168. New geometry still targets one CTA/SM; shared
storage alone excludes two. Actual compilation/attributes/occupancy remain
unknown. Retain metadata shape and all named barrier IDs. Set raw/X depth2
consistently in both producers, consumer modulo/parity, workspace sizing and
barrier initialization. Define indices from depths: BW_F0, BW_E=NS,
BX_F=2NS, BX_E=2NS+NX. Existing joined full barriers still require two producer
arrivals and both byte counts; every raw/X empty barrier keeps all16 consumer
warp arrivals. Unused legacy X-full slots do not become a completion mechanism.
The A base is `ring + NS*UNIT_BYTES + NX*WGS_XS`.

## Exact A layout and arithmetic

Use a K-major, no-swizzle core layout for each m64k16 tile:

```
offset(m,k) = (m>>3)*256 + (k>>3)*128 + (m&7)*16 + (k&7)*2
A_tile = A_base + (ci&1)*16384 + wg*4096 + jj*2048
A_desc = wg_desc(smem_u32(A_tile), 128, 256)
```

For register index r0..3, original RS fragment row and starting K are:
`m=16*wp+(lane>>2)+8*(r&1)`, `k=2*(lane&3)+8*(r>>1)`.
Store each unchanged `bmul2l/bmul2h` uint32 result at that address: its low/high
BF16 halves occupy k/k+1. No float conversion, recomputation of scale, extra
rounding or change to KPERM. Eight stores/thread/group materialize GS2.
Each scalar store instruction addresses32 distinct banks within a warp under
this layout; that is an address property, not measured transaction throughput.

Replace only WGS's WGSMMA1..8 wrappers with SS forms taking A/B descriptors.
Keep all N=8,16,...,64 dynamic job variants, both PB phases and both BIG kernel
specializations. The SS BF16 instructions use scales1,1 and transpose0,0.
Preserve `mma0` scale-D0 positions, NTT1 even/odd accumulator assignment,
all k16 issue order, final d+d_odd, FP32 fixups, BF16 SwiGLU store and epilogues.
The instruction family/operands/terms remain the same; PTX does not promise
bitwise internal reduction order between forms, so GPU numerical tests remain
mandatory.

[NVIDIA PTX's WGMMA layout and instruction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)
supports BF16 A/B descriptors, aligned shared tiles and K-major core strides.
Its generic-to-async memory rule requires a proxy fence for ordinary shared
stores, and WGMMA completion must be observed before reusing operand storage.
The candidate keeps its existing register fence as well.

## Fixed publication and reuse schedule

For every original commit group ci0..7 in every stage:

1. Execute `bar.sync 3+wg,128` **before any bank overwrite**. Together with
   every thread's preceding wait1, it makes retirement of the previous use of
   bank ci&1 collective. First use is also barriered for a uniform schedule.
2. Decode the two original k16 steps and scatter their unchanged BF16 pairs
   into that WG's private shared A region. Other WGs have disjoint regions.
3. Every consumer thread executes `fence.proxy.async.shared::cta`, then
   `bar.sync 3+wg,128`. This is separate from the pre-overwrite barrier.
4. Keep `wg_fence`; issue the same two WGMMA operations with SS operands;
   keep commit and wait1 exactly where the parent placed them.
5. Keep previous-X release at ci0, deferred `wgs_flush` at FCI3, raw weight
   release after every stage code/scale has been loaded, and terminal wait0.

After group g wait1, all groups through g−1 have retired. Before writing g+1's
bank (also used by g−1), the next pre-overwrite WG barrier covers every lane's
retirement. A stage has eight groups, so restarting ci at0 preserves bank
parity across stages. Segment boundaries drain wait0 before epilogues/fixups.
The WG-private barrier IDs3..6 already serve `wgs_flush`; reuse is sequential
and all128 WG threads participate in each instance. Producer warps never join
them. Preserve zero-job/empty-range returns, existing WS_JOIN/grid barriers,
counter generations and all phase-B per-stage dependencies. No new global
spinning protocol or input-derived preparation is introduced.

## Costs and scope, without a speed forecast

Per complete job there remain48 units×16 stages=768 raw weight copies and768 X
copies,27MiB packed weights,24576 collective WG commit groups and49152 WGMMA
instructions. The new A path adds96MiB ordinary shared stores and96MiB logical
tensor A reads/job. It adds24576 pre-overwrite plus24576 publication WG barriers;
these are collective invocation counts, not wall-clock cycles. Decode/tensor
arithmetic is not removed. Raw/X lookahead falls from three to two slots.

The saved standalone route examples have16–29 jobs, hence add3–5.4375GiB of
logical shared A store/read payload per call and786432–1425408 collective WG
barrier invocations. These are not measured traffic or transferred decoder
route distributions. No shared-bandwidth or cache floor is assumed.

The independent store-service sensitivity check makes that cost concrete:
at an assumed128B/SM/cycle across132 SMs and the public receipt's1980MHz endpoint
clock samples, new stores alone need ideal service of3.009µs/job, or72.22–87.26µs
for24–29 jobs. At an explicit1500MHz sensitivity point that is95.33–115.18µs.
Those scenarios consume a large fraction of saved MAIN latency. They are not
an additive latency prediction: stores can overlap other work, the clock was
not measured throughout the kernel, and those jobs are not the official run's
observed routes. Tensor reads/TMA are not added to that port model. See the
independent feasibility note for the primary bank documentation and pinned
receipt. This substantial store demand is an accepted experimental risk.

On exact97af decoder profiles, affected MAIN cases are T32/128/256 (three of six
loads). A conditional20% reduction in those MoE rows corresponds to5.045% whole
decoder latency reduction;25% gives6.350%; about19.827% would give5%. These
scenarios use actual46/16 weights and normalized benchmark times, with separate
profiler shares. PDL overlap means they are neither savings bounds nor promises.
The scope is substantial enough to test; whether the added traffic outweighs
register-lifetime benefits is the experiment's question.

## Bounded implementation and CPU gates

After root/critic acceptance, one writer makes the isolated candidate. Required
changed regions are WGS constants/barrier indexing, WGS wrapper operands,
`wgs_seg` storage/publication, `smem_need`, a distinct extension cache name and
bounded host resource reporting. Python packing/route policies and non-WGS
preprocessed output must match247888 exactly.

CPU verification must bind actual candidate source and generated NTT wrappers:
all lane/register/half locations and descriptors; actual KPERM packing and scale
decode operands against parent; identical per-NTT/PB MMA call trace including
NTT1 parity/mma0; full depth2 raw/X producer/consumer event schedules; A bank
retirement and explicit pre/post barriers; starts/wraps, partial segments,
zero jobs, skewed routes, split-unit fixups, FCI3 and finalizer ring reuse.
Use negative controls for one bank, wrong descriptor/scatter, missing reuse or
publication barrier and premature X release. Models are not GPU memory proofs.
Actual emitted source, immutable parent restoration and the whole focused diff
receive independent review and a root rerun before any check.

Feasibility reproducer: `python3 tools/audit_wgs_shared_a_feasibility.py`.
Evidence: `experiments/iterations/20261104-wgs-shared-a-feasibility.json` and
the separate `WGS_SHARED_A_INDEPENDENT_FEASIBILITY.md` / independent JSON.
The present feasibility checks are layout/lifetime arithmetic, not a candidate
source proof or CUDA compilation.

## Runtime and measured decision

Use exact compiled BIGfalse/BIGtrue function pointers and actual194304-byte
allocation to query/record attributes and occupancy once per built WGS module.
Require successful resource APIs and valid launch admission (at least one CTA),
not a speculative register target. Report local bytes and compiler serialization
diagnostics honestly; neither alone automatically prevents timing a correct,
valid kernel. Add no register cap, quota redistribution or dynamic algorithm
fallback. Keep host queries outside graph replay; no harness/phase inspection.
WGS-only informational ptxas verbosity may be enabled with bounded host markers;
truncated diagnostics remain incomplete evidence of absence.

Root may authorize one private standalone MoE check after source gates. Require
all5 correctness tests,6 benchmarks/6 profiles, clean flags/checks and intended
MAIN activation; small cases are unchanged controls. Compare against exact247888
public `d24dd50196cb4158` and private `bcb761dfca264db0`, preserving matching
evaluator and case metadata. Proposed useful-result gate: affected M32/128/256
geometric speedup at least3% over both, no affected case over3% normalized latency
regression, and control geometric drift within3%. Root may fix these thresholds
before implementation. Confidence intervals/seeds/nodes limit causal precision.

Only a useful result proceeds to a separate exact97af decoder integration and
whole-decoder check. Correctness/compile failure or no useful performance ends
this fixed attempt; no bank/depth/quota/NTT rescue sweep. Serialization warnings
cannot substantiate a parent-bug story, and an isolated source/compiler success
cannot substitute for measured decoder benefit.
