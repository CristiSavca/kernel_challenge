# Small-token MoE: two NW4/NS2 CTAs, with unchanged output units

2026-11-04. Proposal and CPU proof, not an implemented or measured candidate.
Base: current standalone KPERM SHA
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
Current decoder is `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
Their small paths inherit ef7/80a4. Root owns every GPU check and submission.

## Reject the grid-only version; retain one substantive alternative

The host does launch only nSM CTAs. However, existing tiny/MID host allocation
reserves an unused fat workspace: actual NW8 metadata is36,584B, rounded36,608;
the host requests214,784B dynamic. Even a tiny-only three-slot allocation still
needs147,200B. Separately, the saved exact ef7 binaries use144/148/149 registers
at288threads (TINY2/MID/TINY; all stack/local0). Thus neither original shared
storage nor registers permits two resident CTAs. These measurements are in
`experiments/modal/20261104T052256Z-4c806134/result.json`, baseline extensions.
The measured TINY uses MULDEC0/FINPF1; decoder TINY uses MULDEC1/FINPF0, so its
precise register count is not established by that capture.

NW4 with three36KiB slots also fails: partial sums alone occupy8KiB, exceeding
the roughly5KiB left per CTA after weight rings and static shared allocation.
Do not test either impossible geometry or silently relax the residency guard.

**One retained design:** only TINY/TINY2/MID builds use NW4, NS2,160threads and
two CTAs/SM. Keep16-row/full-K4096 output units, WSNR4, NACC4, FAST1=2, all
job splitting, original packed weights, epilogues, and math. Remove unused fat
workspace allocation only for TINY+WS. WGS/KPERM and non-TINY fallbacks remain
unchanged. This is not the rejected compact WGS geometry.

| Resource/work | Existing small path | Proposed small path |
| --- | --- | --- |
| Consumers / helper per CTA | 8 /1 warps | 4 /1 warps |
| CTAs per SM | 1 | 2, subject to actual gate |
| Compute/helper warps per SM | 8 /1 | 8 /2 |
| Weight units/job, A+B | 512+256 | 512+256 |
| Weight TMA payload / total copies | 36KiB /768 per job | unchanged |
| Packed weights/job | 27MiB | unchanged |
| Slots per CTA / aggregate per SM | 3 /3 | 2 /4 |
| Native metadata / rounded | 36,584 /36,608B | 27,880 /27,904B |
| Allocated dynamic shared | 214,784B | 101,632B |
| Two proposed CTAs +1KiB static each | — | 205,312B <233,472B |
| Cached activation registers/thread | 64 | **128** |

The128-register x cache is the principal risk.160threads×2 admits at most about
200registers/thread by aggregate allocation arithmetic, not a guarantee from
the actual occupancy calculator. Scale records,16accumulators, addresses and
epilogue state must also fit. Apply a two-block launch bound only to this new
build and query both compiled specializations at the exact dynamic size. Any
capacity below2 is a failed experiment before launching a spin-barrier grid.
Do not infer unchanged register counts from NW8 or hide a failed gate.

## Exact implementation scope for review

1. Change only the three small-build define tuples: NW8→NW4; add NS2. Preserve
   all other flags, including the different tiny/MID WSBLK and FINPF choices.
   Add a new extension namespace. No model-weight repacking or extra copies.
2. In `skinny_ring_ws`, existing code addressing already generalizes through
   NCH=64/NW. For NW4, load the two existing8-chunk scale records at
   `UNIT_CODES+(2*warp)*512+lane*16` and that address+512. For chunk c0..15,
   select record c>>3, local chunk c&7, and its two scale bytes. Preserve both
   PRMT and general-scale LUT arithmetic. Prefer an outer two-group loop that
   keeps only one uint4 scale record live; do not retain eight scale registers
   unnecessarily. All eight code values per word and all K terms stay intact.
3. `okc_smem_bytes` returns base+skinny `xa` only for TINY+WS; unchanged max(xa,xb)
   elsewhere. Static assertions tie the new path to NW4/NS2/SKT8/WSNR4 and
   FAST1<=NS. Fat work remains impossible: tiny counts<=8, or MID splits them
   into jobs<=8. Preserve the runtime `nF>0` trap.
4. Before any enlarged launch, query actual CUDA attributes, opt-in shared
   memory and occupancy for BIGfalse/BIGtrue. Log threads/dynamic/occupancy/
   registers/local once. Require capacity>=2, scale only this extension's input
   grid nSM→2*nSM, and retain the parent PDL stream attribute/trigger/wait.
   A positive capacity query is necessary for admission, not a speed claim.
5. Keep all global scratch sizes and offsets. WGS fixup slots are not used by
   TINY+WS and its grid remains unchanged. No norm/FINCOL alternative in this
   first test. For FINCOL, FSEG4→8 keeps four columns/thread and all4096columns.

## Address, schedule and lifecycle evidence

`tools/audit_small_moe_residency.py` runs the actual inherited integer packing
helpers on CPU. Four coordinate-nibble patterns uniquely identify all65,536
weight coordinates in a complete16-row/full-K unit, in both DQ7/MULDEC layouts.
It checks524,288 decoded values,32,768 scale accesses and32,768 distinct
activation coordinates. The new map is exactly old_warp=2*warp+(c>>3),
old_chunk=c&7. The two scale records are512B apart, **not adjacent32B**.
Evidence: `experiments/iterations/20261104-small-moe-residency-cpu.json`.

One skinny job has512 independent A units and256 B units. Even one job offers
enough A units for264 CTAs. All A work is input-independent of other CTAs; each
helper publishes its entire A contribution before waiting for B partials.
Consumers wait on doneA[j]==512 only after their own A loop. Given residency,
there is no A→B wait cycle. Empty A ranges also have empty B ranges; zero-job
CTAs perform joins and the global barrier without ring waits or job indexing.

The abstract schedule proof covers J0..44 (the valid maximum at M16), grids
132/264/396/528,59,400 ranges and3,041,280 unit ownership checks. Every unit has
one owner and every job receives exactly512 A credits. NS2/FAST1=2 preissue and
refill cover the same concatenated A+B sequence once. WSBLK1/2 transfers only
the last min(WSBLK,nB) B epilogues; all A epilogues stay with the helper.
WSBLK<=WSNR4 remains valid. A new exact-source proof must verify conformance.

The grid counter target already uses gridDim.x. Shared metadata is per CTA;
doneA is per job; ss is per routed entry and alternates generation. Counter
reset loops use grid*NT, which remains132*256=264*128 for this change. The same
scratch generation must survive M1→M8→M16→M32 and reverse transitions, zero
local routes, hot experts and repeated fresh graphs. No CTA-dependent global
array grows when WGS is excluded. Preserve the initial PDL wait and all final
release/acquire ordering; the prior compact PDL admission claim was withdrawn.

## Costs and credible scope

Weight traffic, unit count and copy/barrier count per job stay unchanged. There
are twice as many independent helpers and more aggregate weight lookahead,
with half as many shared partial vectors per unit. This could hide the tiny
consumer/helper waits without compact WGS's fourfold stage count.

There are also twice as many routing/prologue copies and grid arrivals, and
more CTAs load each job's x/inter into registers. CPU examples show additional
activation-cache requests of0.5–7.4% of packed weight bytes for selected valid
histograms; these requests can hit cache and are not measured HBM traffic.
Do not claim activation traffic unchanged. The finalizer keeps total ybuf
values/bytes and four values/thread, but twice as many finalizing CTAs fetch
route/ss metadata. For M<=16, M*FSEG<=128<264 so each active CTA has at most one
segment; there are no extra serial segment-loop iterations. Different CTA
completion placement can still hurt M1 latency. FP32 K-partial reduction groups
change8→4, requiring fresh numerical checks despite unchanged precision/terms.

This targets M1/M8/M16, half the decoder's loads. With approximate historical
MoE shares30%/49%/46%, a20% small-MoE latency cut would mean4.27% aggregate
decoder reduction;25% gives5.36%,40% gives8.75%,50% gives11.08%. These are
profile-share scenarios, not predictions; PDL waits/overlap limit attribution.
The remaining leader gap is about14.1%, so even halving these expert times
would need another improvement. No byte-floor argument establishes such a gain.

## Bounded validation and kill gate

After independent plan approval, implement one standalone candidate, with exact
packed-byte/address checks and abstract schedule tests updated against its source.
Verify unchanged WGS/preprocessed controls, parent small math, full scratch
coverage, launch/resource guards and no unintended non-NW4 dispatch changes.
Root may then choose one official private check. Require both actual two-CTA
capacity and all numerical tests. Report compiler local/stack/spills as evidence,
not an excuse to waive occupancy or assert speed. No fallback to a different
geometry and no automatic NS/FINCOL sweep.

Performance target: at least15% lower geometric-mean normalized latency over
M1/M8/M16, no small case worse by more than3%, and stable WGS controls. A clear
regression kills this candidate. A coherent but weaker result needs an explicit
whole-decoder payoff review; promotion requires a separately reviewed full-layer
check against exact97af, including PDL. Do not score from occupancy or standalone
aggregate movement alone.
