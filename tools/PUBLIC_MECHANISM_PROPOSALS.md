# Isolated public mechanisms: offline proposals, 2026-11-04

Status: design review only. No candidate was written and no GPU job was launched.
The intended parents are frozen decoder `80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`
and standalone MoE `ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a`.
The reviewed public source is keell0renz commit
`890809cee421500b00126d084e81b338ef9181ab`, decoder SHA256
`82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
Its Apache-2.0 LICENSE and attribution NOTICE are retained with the pinned source.

Priority for further review is KPERM3, because it tests repeated scale-load/decode
issue while preserving weight traffic and pipeline structure. Natural-K SC6 and
MFIN are smaller possible margins with concrete offsetting costs. None has a
supported large gain forecast; NCU's ~75% DRAM activity does not establish
consumer scale decoding as the exposed bottleneck.

## KPERM3-only: preserve the matrix terms, reuse each scale record

The public source's `_kperm_src`, `_kperm_codes`, `_kperm_rows`, `_wgs_copy`
(7136–7184), gather (5794–5799), and scale loads (5987–6035) implement a coupled
weight/activation permutation. This is separate from its MFIN, SC6, BF16 ybuf,
barrier, finalizer and scheduler changes.

For a position `p` within one 256-column stage, define:

```
p = 16 * (4*lg + j) + c
t = (c >> 1) & 3
i = (c & 1) | ((c >> 3) << 1)
pi(p) = 16 * (4*t + j) + 4*lg + i
```

Here `lg,j,t,i` each range over 0..3. This just permutes the four two-bit
coordinates, so all 256 indices occur exactly once. For a lane's fragment
column `c=8*q+2*t+e`, the original source index becomes
`16*(4*t+j)+4*lg+2*q+e`. Its original FP4 scale block is therefore `4*t+j`,
independent of `lg`. A lane needs four scale pairs for the entire stage, rather
than a different set of four pairs in each of four load groups.

The complete paired transform is necessary:

1. Prepare permutes each W13 and W2 row's FP4 code nibbles along K within each
   256-column stage. Codes remain exact four-bit values; scales remain exact
   original E4M3 bytes. Each scale record is reordered to
   `[warp,g,t,j,gate-or-up-half]`, with no compression.
2. Runtime phase-A gather puts `x[stage_base + pi(p)]` into the existing xA
   layout at logical position `p`. For a destination eight-element chunk
   `G=tid&31`, `B=8*((G>>1)&3)+2*(G>>3)+(G&1)`, its four source BF16 pairs
   begin at `2*B + 64*t`, `t=0..3`. The 32 lanes' B values are a permutation
   of 0..31: each scalar load accesses one coalesced 128-byte quarter of the
   same 512-byte warp input region. The store remains the parent's uint4 store.
3. Prepare also permutes W13's gate/up output rows together: logical hidden
   channel `P` is physical channel `256*(P//256)+pi(P%256)`. Its ordinary
   SwiGLU epilogue consequently writes the matching permuted intermediate
   directly into the parent's xB layout. W2's K permutation consumes it.
   W2 output rows and post-expert norm weights stay in their original order.
4. Consumer scale loads use one uint2 record per lane per stage. Construct the
   four BF16 scale pairs once and reuse them through the four load groups.
   Keep the parent's `dq4`, HMUL2 operations, GS2 commits/waits, NTT1 dual
   accumulators and combine, FP32 ybuf, final norm/reduction, and output cast.

Independent CPU enumeration checked the entire 256-index permutation, all
fragment coordinates, all eight-element gather chunks, all 4096 hidden-channel
indices, and the per-lane scale-block property. The permutation stays inside
each 256-column stage, so phase B still waits for the same two phase-A units
and eight warpgroup releases. It introduces no new cross-stage dependency.
This proof establishes index coverage and scale matching, not CUDA correctness.

The two GEMMs retain every product, operand precision and accumulation precision.
The K reduction order changes, including which terms enter the parent's two
NTT1 accumulators. W13 rounding can therefore change subsequent BF16 SwiGLU
values. This is ordinary numerical reassociation, but only fresh-input oracle
checks can establish acceptable error. Do not claim bit-identical outputs.

### Costs and falsifiable benefit

The natural-K parent loads scale records at ef7 lines 3046–3053, from the
layout at 4003–4018. KPERM changes four issued uint2 scale loads per lane per
stage to one, and permits four rather than sixteen scale-pair constructions.
That offers removal of three shared-load instructions and twelve PRMT/IMAD
pairs per stage before compiler scheduling. Actual SASS must confirm reuse;
source-level common expressions do not establish the generated instruction count.

This is **not** four times less unique shared-memory scale traffic: the parent
broadcasts each natural-K scale record to four lanes, while the permuted layout
uses distinct records per lane. Unique bytes remain 1024 per warpgroup piece.
Global weight bytes remain 9216 per piece and 36864 per stage. No HBM traffic
reduction or occupancy gain is claimed.

The gather changes one 128-bit global load instruction into four 32-bit loads,
with extra address arithmetic but the same coalesced bytes. The batched uint4
gather storage and all activation/TMA allocations stay unchanged. Four decoded
scale pairs live longer than two packed words; whether that raises peak register
pressure depends on the parent's simultaneous current/next records and compiler
scheduling. The compiled 96-register kernel is already occupancy-limited, so
C7510/C7512 serialization, stack/local usage or a slower gather can erase the
benefit. No new shared memory or more resident CTAs are expected.

Preparation would replace only the WGS copies at weight entries 12/13. Existing
skinny/MID/fallback encodings and runtime dispatch stay intact. The permutation
does not require an extra expanded model-weight copy after prepare. It does add
temporary nibble indexing and output-row indexing during existing packing;
peak temporary storage and total prepare time must be recorded. Main extension
build count should stay unchanged; give the candidate a unique build name.

Affected official loads are decode 32, verify 16 and verify 32 when WGS is
selected. M <= 16 remains unchanged. Nearby main-WGS shapes and chunked calls
must also work. A useful causal screen would hold the parent's pipeline, cost
tables, all math, PDL and dense choices fixed, then compare whole-layer or
standalone timings plus actual compilation resources. This document does not
authorize that screen or infer a 10–15% gain from fewer instructions.

## SC6-only versus KPERM3

SC6 is easier to isolate for synchronization than MFIN: it needs no new completion
protocol, scratch generation or finalizer. Qualify actual local-bank scale bytes
in prepare against 0x40..0x7E, in addition to the parent's PRMT predicate. Store
the qualification with the prepared state and extension cache key; do not select
encoding from input shapes, seeds or a mutable global shared across prepared
states. If any byte fails, retain the parent's full eight-bit WGS encoding and
its existing general-scale fallback. Tiny/MID/fallback copies remain unchanged.

Eight six-bit values can occupy six bytes, so each WGS piece falls 9216→8960
bytes and each global stage 36864→35840 bytes: exactly 2.7778% of weight bytes.
The joined weight/X barrier must expect 35840 weight bytes plus the unchanged
activation bytes. Keep 36864-byte shared ring strides, current ring depths and
barrier counts, and use compressed global strides/piece offsets consistently.
This saves bytes but no shared allocation. Reconstruct the exact original
scale-pair bits before the unchanged multiplication; exhaust all qualified
scale codes and reject/fallback on out-of-range, signed, subnormal and NaN codes.

The public SC6 decoder relies on KPERM's once-per-stage scale record. A clean
natural-K SC6 port needs four record decodes per lane per stage, with extra
mask/bit assembly and two shared load instructions instead of each old uint2
load. Direct pair construction can limit live state, but this remains a different
cost tradeoff from the public combined code. Its reported ~2% result cannot be
attributed to our natural-K version. The additional flag need only select one
main encoding per prepared bank; do not build a second main extension or keep
both full WGS weight copies merely for production fallback.

Thus natural-K SC6 is less risky for lifecycle than MFIN, but has a small traffic
ceiling and repeated unpack overhead. KPERM3 first is the more substantive
issue-reduction hypothesis; SC6 could later build on a validated KPERM layout,
with any combined attribution stated explicitly.

## MFIN-only integration, retained as a secondary proposal

This is a same-kernel completion change, unlike rejected V8's separate finalizer
launch and mandatory whole-main dependency. Do not copy the public BF16 ybuf,
B4 finalizer, SC6, KPERM, FCI4, separate producer barriers, cost model or prepare
timing helpers. Enable only the existing WGS/NW16/FT64/WGSFINSS configuration;
leave all other builds' old barriers, counters and generation untouched.

Use a dedicated 4096-int region appended after the parent's integer scratch
(`8+1024+EMAX` words), initialized once with scratch. Suggested offsets are CE=0
(two banks of 32 expert counters), EXIT=64, expected fixed G at 65, XR=128
(two banks of 96 row counters), DA=384 (two banks of `JMAX*16` stage counters).
With parent JMAX=88 the live extent is 3200 ints. Static assertions must prove
no overlap and the host must check the allocated extent. No existing ybuf,
fixup, sum-of-squares or tiny/MID offsets change.

Use a separate shared `mfin_parity` rather than repurposing the parent's legacy
`S.gen`: legacy generation and ss alternation remain owned by old non-MFIN
launches. Four extra shared bytes may fit the parent's current rounded padding;
prove the actual compiled `sizeof(Smem)` and dynamic allocation before claiming
unchanged resources. Access parity without keeping an extra pointer live through
the entire accumulation loop, as the parent already does for `ss_cur`.

The ownership and ordering contract is:

1. Preserve `griddepcontrol.wait` before any scratch access (ef7 3303; decoder
   8384). Exactly one consumer thread per CTA increments EXIT with wrap limit
   `2*G-1`; an old value >= G selects parity 1, otherwise parity 0. Scratch's
   fixed G is prepared from the device SM count. All G CTAs must be resident,
   launched each time, and use that same G. One scratch state has one serialized
   stream/graph owner; concurrent users require distinct scratch, as in the parent.
2. After routing, consumers zero only the opposite parity's CE/XR/DA entries,
   distributed without duplicate ownership. The prior kernel has fully completed
   before this point. Initial zeros make the first active bank valid; each next
   active bank was cleared by the previous MFIN launch. Tiny/MID interludes do
   not access this region or advance EXIT. Existing legacy ss cleanup can remain;
   WGSFINSS does not read ss during finalization. Do not redirect legacy doneA.
3. Redirect WGS gather's XR release and the X producer's acquire to the active
   XR bank (ef7 2894/2825; decoder 7975/7906). Redirect the existing whole/split
   phase-A releases and producer stage acquire to active DA (2953/3003/2828;
   decoder 8034/8084/7909). Preserve their counts and the joined pipeline exactly.
4. After a whole phase-B unit's unchanged epilogue, set pending kind 4. Its
   existing deferred-flush boundary remains FCI3. Flush kind 4 performs the
   128-consumer-warpgroup barrier, then one release add to CE for `S.jE[j]`,
   and clears the pending record. The final `wgs_phase` flush at ef7 3175 /
   decoder 8256 must publish a last whole unit even without a following segment.
5. Split-B contributors publish no CE count. Only the existing designated or
   atomic last finisher publishes once, after it reads partials, writes the
   unchanged FP32 epilogue, clears its fixup counter, and synchronizes its 128
   threads. Preserve all fixup zeroing. Each job publishes exactly
   `16 output units * 4 warpgroups = 64`; count only complete output units.
6. Preserve the 512-consumer CTA barrier after local B before finalizer scratch
   aliases the ring. Each consumer has acquired every assigned joined TMA stage
   and drained its MMAs. Producer schedules cover precisely those consumed
   stages, so no future producer issue remains; loop-exit instructions may still
   run, but no unconsumed ring write may remain. Remove only the global completion
   wait and its now-unneeded WGS flag-reset block. Keep the downstream PDL trigger;
   consumers of this kernel still wait for its complete grid.
7. Before each token's existing FP32-ybuf finalizer, its first eight lanes acquire
   CE for its local routes. Let `tot=S.cnt[e]`: job count is 1 for
   `0<tot<=SK_T`, otherwise `ceil(tot/FT_T)`; zero/nonlocal routes require no
   wait. Target is 64 times that exact job count. A subsequent 512-thread barrier
   makes readiness common before any ybuf loads. Leave all finalizer arithmetic,
   cached float4 values, summation order and output stores unchanged.

Fail loudly: reject unsupported grid/configuration/scratch extent on the host;
trap on impossible over-count or invalid expert/count state. A not-yet-ready
counter must keep waiting with acquire semantics. Never copy the public
4,000,000-poll fall-through. A missing release must fail the bounded diagnostic
through a real trap/watchdog/outer timeout, never produce an apparently complete
output. A debug watchdog may trap; it must not turn elapsed polls into readiness.

Required CPU review covers whole/split unit enumeration, counter targets, region
non-overlap and parity transitions including M16→M32→M16, WGS shapes, zero-local
routes and chunked calls. CUDA checks must exercise shared scratch across repeated
fresh graphs, both launch templates, the unchanged fallback, and all output/ID
oracles. New compiler outlining/serialization and global atomic pressure are
first-class risks, even though arithmetic instructions remain unchanged.

### Conditional tail accounting and ROI

`experiments/iterations/20261104-public-mechanisms-offline.json` records every
saved diagnostic replay and the official benchmark/profile provenance. Define
`Bmax` as the maximum of all four-per-CTA WG B-return stamps, and `F` as the
maximum final consumer timestamp. Whole-B stores can still be pending at Bmax:
it is not an all-outputs-visible marker. If every B-return time were unchanged,
the new kernel could not finish before Bmax, so deleting all `F-Bmax` is an
optimistic fixed-completion thought experiment, not a general MFIN bound.

| M | Unperturbed median F-Bmax | Largest unperturbed F-Bmax | Official weighted layer median | Official weighted expert profile |
|---|---:|---:|---:|---:|
| 32 | 5.344 us | 6.304 us | 197.304 us | 81.203 us |
| 128 | 4.912 us | 5.856 us | 279.871 us | 150.335 us |
| 256 | 7.904 us | 8.704 us | 369.631 us | 200.639 us |

M32 seed 7301 is excluded from these summaries because probe perturbation exceeded
5%. These are standalone routes and instrumented write-flush measurements, not
decoder timelines. If their entire absolute tail transferred to the official
three affected loads and disappeared without PDL overlap or added work, six-load
aggregate latency would fall only 1.11% (1.28% using each size's largest replay).
Transferring the tail as a fraction of expert profile time instead gives
0.79–0.93%. Neither transfer is a forecast or certified decoder ceiling.

MFIN adds 64*J CE atomics: 1024 at J16 and 1728 at J27, versus 132 global
completion atomics removed. It also adds full-B publication barriers, parity
metadata work and finalizer traffic while other CTAs are still computing B.
Those costs can outweigh the small tail. Current evidence does not justify
implementing MFIN as a major bet or requesting another diagnostic to refine this
accounting. Retain the proposal; prioritize the independent KPERM issue mechanism
only if its static and resource argument survives critic review.
