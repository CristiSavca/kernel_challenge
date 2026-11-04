# Independent review: small-token MoE NW4 / NS2 proposal

2026-11-04. Source-only review by the attention/dense agent; no candidate edits,
GPU allocation, compilation or official actions. Implementation and final source
approval remain with the MoE author and independent critic respectively.

Reviewed active standalone source
`experiments/candidates/muse_moe_w4a16/247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0.py`
and plan `tools/SMALL_MOE_TWO_CTA_PLAN.md`, SHA256
`c5322412dc91d976f5967586239a0260f6453527a9ed9eb5c6bb00dafac92372`.
Also inspected the author's actual packing-helper CPU audit and saved output.

**Conclusion:** no additional blocking defect in the current plan. The proposal
is suitable for bounded implementation and exact-source review. This is not a
CUDA correctness or performance approval. The compiled two-CTA residency guard
is mandatory because phase-B dependencies and the final grid barrier can wait
for CTAs that have not yet run.

## Address and arithmetic requirements

- `_pack_scales` stores `[physical_warp8][lane][eight_chunks][row_half]`.
  NW4 gives each compute warp sixteen chunks. Its two scale records are at
  `UNIT_CODES + 2*warp*512 + lane*16` and that address **plus 512 bytes**.
  They are not one adjacent 32-byte record. Chunk `c` uses record `c>>3`, local
  chunk `c&7`, and bytes `2*(c&7)+h`. Both PRMT and general LUT scale builds
  need this same addressing.
- Existing code address `(warp*NCH+c)*512+lane*16` and activation address
  `warp*(64*NCH)+64*c+16*t` already generalize to NCH16. Keep existing NVFP4
  decoding, BF16 multiplication, token padding and all four MMA steps.
- An outer two-group scale loop must keep all four FP32 accumulator arrays
  live across both groups. Resetting accumulators per group, or accidentally
  reusing the first activation half, would lose terms. The new K grouping
  (sixteen chunks in each of four warps versus eight in each of eight) changes
  FP32 rounding order; do not claim bitwise equality to the parent.
- Confine NW4 to TINY+WS. Other skinny implementations still assume their old
  scale-vector width; an accidentally dispatched non-WS NW4 variant would not
  become correct just because `skinny_ring_ws` was fixed.

## Pipeline, route and finalizer corner cases

- NS2 is compatible with FAST1=2 only if the assertion remains explicit. The
  early preissue writes slot `i` directly rather than taking `i % NS`.
- With one job and 264 CTAs, phase B has 256 units: some CTAs have positive A
  work and zero B work. The parent's `Nh=N-min(WSBLK,nB)` and consumer guard
  `warp<WSBLK && warp<nU` correctly retain all A epilogues in the helper and
  avoid reading uninitialized B-tail constants. Exercise this case, N=2 with
  nA=nB=1, zero total jobs, odd ring wrap counts and WSBLK1/2.
- Preserve the compute-only barrier, partial-buffer free/ready parity and both
  helper joins. In WSB mode the compute barrier precedes refill of the consumed
  weight slot; the helper returns a partial buffer only after copying every
  compute warp's partials. Tail buffers are retained through the consumer
  epilogues; NS2 does not reduce the independent WSNR4 buffer count.
- The zero-job helper performs both joins but no ring wait. Compute threads
  still perform scratch resets, the generation barrier and zero-valued output
  finalization. Do not add an early return that bypasses those operations.
- `muldiv_g` in the parent computes an approximate quotient then corrects it by
  one using integer comparisons. The proposed CPU schedule covers products
  bounded by 44*512*528 = 11,894,784, below the helper's stated 2^24 bound.
  Exact-source coverage should confirm the same range formula is used by early
  preload, compute and helper paths; the author's schedule is an abstract model.
- The existing FINCOL formulas correctly produce FSEG8 at NT128. Each thread
  writes four columns; segments cover 4096 columns exactly. Keep FINPF's `hasp`
  guard and the first-item/remaining-items split. For M<=16 and grid264, there
  are at most128 finalizer items, so no CTA needs a second segment. The inherited
  comment saying 1024 columns is stale for NW4 and should be updated.

## Resources, scratch and decoder integration

- The proposed 101,632-byte dynamic allocation requires selecting the skinny
  workspace in TINY+WS. The parent's unconditional max with the unused fat
  workspace still requests too much memory for two CTAs. Preserve the original
  max formula in other builds and retain the unexpected-fat-work trap.
- Activation cache storage alone grows from64 to128 scalar registers per
  compute thread. Scale state, accumulators and addresses add to this. A
  two-block launch bound is not evidence that two blocks fit: query the actual
  false/true compiled specializations with exact threads, dynamic shared memory
  and static attributes; reject capacity below two before doubling the grid.
- The doubled small-path grid must not alter WGS launches or WGS fixup sizes.
  Small paths do not use per-CTA WGS fixups. Their scratch cleanup coverage
  remains complete: 264*128 = 132*256 compute threads. Grid-barrier counters
  already use `gridDim.x`; doneA remains per job and ss per routed entry.
- Keep the original initial PDL wait, helper final fence/join, dependent-launch
  trigger, generation release/acquire and scratch-reset order. Standalone PDL
  is off; a passing standalone check cannot establish decoder PDL correctness.
  Decoder TINY also differs in FINPF and MULDEC configuration. Integration must
  preserve those decoder flags and rerun the exact resource checks.
- Required transition coverage includes M1/M8/M16 to WGS M32 and back, with
  fresh routes, zero local routes, hot experts, and repeated graph replays.

## Performance interpretation

Two new CTAs provide eight compute warps per SM, the same number as one parent
CTA, and two helpers instead of one. The hypothesis is better pipeline overlap,
less cross-warp reduction work and a smaller serial unit tail. It is not a
doubling of compute-warps occupancy or a reduction in weight bytes. Routing and
grid-arrival work grow, and more CTAs load each job's activations. The plan now
acknowledges these costs. Require the stated small-case performance gate and
stable WGS controls; a resource fit alone is not grounds for publication.
