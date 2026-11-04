# Compact WGS residency proposal — 2026-11-04

Status: research and implementation plan, not a GPU result. Parent is the exact
scored standalone `ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a`
(working `muse_moe_w4a16/candidates/public_cuda_v6.py`). Do not mix KPERM. Root
authorized implementation after independent plan review; only root submits.

## Research that changed the plan

- [GC-MoE](https://arxiv.org/abs/2611.04222), submitted **November 3, 2026**, is
  the newest verified relevant arXiv result. It partitions communication and
  GEMM resources, with register-A FP4 WGMMA on Hopper. Its 8-H20/8-B300 results
  concern distributed dispatch, absent from this single-H200 task. No headline
  gain is transferable. The useful lesson is to account for resource contention.
- [MonoMoE](https://arxiv.org/html/2609.04244v1), dated **August 19, 2026** in
  the primary page despite its September identifier, uses weight-major small-N
  tensor operations and one persistent resident grid. Our parent already has
  swapped operands, persistent expert streaming, fused SwiGLU and readiness
  publication. Its FP8 activation quantization is not an eligible substitution.
- [LiquidGEMM](https://arxiv.org/html/2509.01229v1), **September 1, 2025**, is
  older but supplies counterevidence: its separate dequantize-to-shared WG adds
  data movement/synchronization; unified dequantize-and-MMA consumers outperform
  that arrangement on H800. Keep register-A decode; do not copy LiquidQuant/W4A8.
- [Humming](https://vllm.ai/blog/2026-10-21-humming), **October 21, 2026**, is
  an engineering release, not an arXiv paper. Current commit verified via GitHub
  API: `87b70fb99afe51867be2845ce4db0f87fc01bc42`, November 4 03:23 UTC. MXFP4
  group-32/power-of-two scale wins do not prove NVFP4 E4M3 group-16 gains.
- [Chord design](https://github.com/novitalabs/chord/blob/6199d44a863a165139d2fd8387b307bbe21f1ca9/docs/design.md),
  commit `6199d44a863a165139d2fd8387b307bbe21f1ca9` dated **October 20, 2026**,
  explicitly uses two resident WGMMA CTAs for small masked tiles, and three/four
  MMA CTAs for indexed decode. It uses INT4/group-32 weights, so only its
  resource decomposition is a lead. Its reported speedups are not forecasts.

The parent tiny/MID paths already use `mma.sync.m16n8k16` with weight on M,
register activations, helper-warp epilogues and persistent execution. Merely
switching to mma.sync or copying a small-token tile is not a new hypothesis.

## One falsifiable structural hypothesis

Two independently advancing, compact CTAs per SM can hide producer/consumer
latency better than the parent's one large CTA, without another weight pass or
different token-job partition. The goal is a useful gain on M32/M128/M256, not
an occupancy number alone. Tiny/MID M1/M8/M16 remain unchanged as controls.

| Property | Parent | Proposed compact |
| --- | ---: | ---: |
| Output rows per unit | 256 | 128 |
| K per stage | 256 | 128 |
| Stages per full K | 16 | 32 |
| Compute warpgroups | 4 | 2 |
| Producer warps | 2 | 2 |
| Threads | 576 | 320 |
| Max tokens per job | 64 | 64 |
| Joined ring slots | 3 | 3 |
| Packed weight bytes/slot | 36,864 | 9,216 |
| Activation bytes/slot | 32,768 | 16,384 |
| Metadata, rounded to 128 | 22,272 | 21,248 |
| Dynamic shared bytes | 231,168 | 98,048 |
| Intended resident CTAs/SM | 1 | 2 |

The metadata figures come from extracting the actual parent's Smem declaration
and running native clang record-layout evaluation with NW16/NW8, FT_T64,
EMAX3072, JMAX88: sizeof22240/21216, alignment8, rounded128. This checks static
layout arithmetic, not CUDA attributes or actual occupancy. Two compact CTAs
plus 1024 driver-reserved bytes each require198144B, within233472B per SM.
At a hypothetical96 allocated registers per thread, two320-thread blocks use
61440 registers, within65536; actual allocation and spills must be checked.

## Exact transformation and invariants

1. Keep all token counts, routes, FT_T64 splitting, scale decode, BF16 operands,
   FP32 accumulators and every output. No host input-value branch. Compact WGS
   replaces only the MAIN WGS build; tiny/MID/non-PRMT fallbacks remain exact.
2. A compact unit has two64-row WGs and32K128 stages. In the old packed array,
   new `(unit,stage,wg)` maps to old `(unit//2, stage//2,
   2*(unit%2)+wg)` and the first/second half of the old four load groups. Each
   piece is4096 code bytes plus512 scale bytes. This is a bijection; persistent
   packed bytes and logical weight requests per expert/job remain27MiB.
   Preparation changes only WGS copies12/13. No dequantized weights are stored.
3. Phase A has64 output units per job; phase B32. Each unit has32 stages.
   Four times more stage events each transfer one-quarter weight bytes. The
   activation stream is read twice as often because output units doubled;
   unique activation storage stays unchanged. Do not call this zero extra
   memory traffic. Logical weight bytes still scale with J=sum ceil(count/64),
   not just active experts U. Actual DRAM bytes remain unmeasured.
4. X layout becomes `[stage32][ntile][k8=16][token8][bf16x8]`. A WGMMA B
   descriptor uses128-byte leading offset and2048-byte N8 offset. Gather loops
   cover two16-byte chunks per consumer thread (NT256); both GATHB branches
   must be corrected. The tile epilogue publishes BF16 SwiGLU in that layout.
5. One WG writes32 hidden columns. A K128 B-stage depends on four WG
   epilogues, spanning two compact A-units. Use32 doneAs counters/job with
   acquire target4, release after WG barrier. Preserve deferred publication,
   designated finisher, joined producer arrivals and per-warp slot releases.
6. Grow fixup slots to at least grid+1 (e.g.320), use two WGs per slot and
  32-stage completion counts. Enlarge and separately address doneAs for32
   stages/job. Retain all old arrays for fallback compatibility. Recheck all
   reset and double-buffer generation offsets; no alias with ybuf/fixups.
7. Update stage-to-job bit shifts from9/8 to11/10. Old `tot*c` can overflow
  32 bits with264 CTAs; use an exact safe product/divide or quotient-remainder
   decomposition, never approximate a changed partition. CPU-enumerate complete
   disjoint stage coverage, fixup ownership and count conservation across
   nearby counts, zero/local-empty routes and maximum384-token chunks.
8. Query `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for both exact compiled
   specializations in prepare/first host setup, before a launch can spin.
   Throw on runtime error or less than two available blocks/SM for this
   experiment; launch exactly2*SM_count only after passing. `launch_bounds(320,2)`
   helps compilation but is not proof. Print bounded resource metadata once.
   Do not silently claim a one-CTA fallback validates the residency hypothesis.
9. Grid barriers and readiness spins require every grid CTA to be resident.
   Parent ordered stream/PDL wait and launch_dependents placement stay intact.
   An initial proposed delay of launch_dependents is withdrawn: the NVIDIA
   PTX griddepcontrol contract requires every primary CTA to trigger/complete
   before dependent work is eligible, so the supposed unadmitted-parent cycle
   was based on the wrong event semantics. The actual occupancy query remains
   necessary for this kernel's internal spin barriers. Validate ordinary private
   check and eventual decoder PDL integration separately. No independent work
   stream or extra launch is introduced.
10. The final norm currently caches8 columns×8 routes per thread. With NT256,
    naively doubling it to16 columns doubles live registers and risks spills.
    Use a two-pass streaming FP32 norm: first sum16 values per thread per route,
    reduce over8warps, then reread ybuf and accumulate16 output values. This
    adds L2 reads compared with the cached parent. All post-norm and route-weight
    math stays FP32 until one final BF16 store. Reduction order changes and
    needs numerical validation; no lowered precision is permitted.

## Critic and test gates

Reject before GPU if the byte mapping, stage coverage, counter/reset ownership,
asynchronous lifetime, progress argument or metadata budget fails. Independent
critic reviews the exact source before root submits one official private check.
The submitted candidate has no diagnostic instrumentation in timed execution;
one-time host resource logs are permissible and not score evidence.

Reject compiled hypothesis if actual two-CTA occupancy fails, WGMMA is forced
to serialize, or substantial spilling contradicts the planned resource fit.
Any compiler/runtime failure is retained, not silently fixed and reclassified.
Full official correctness and nearby shapes are required. Compare each changed
M32/M128/M256 against exact-parent private/scored timings, and use unchanged
M1/M8/M16 controls to identify job-level noise. An initial target is at least10%
lower aggregate affected-case latency, without more than3% loss on any affected
case; smaller wins need an explicit revised whole-decoder rationale, not an
automatic sweep. No scored promotion before a useful independently reviewed gain.

This is one broad WGS experiment, not a proof of a path to #1 by itself. A
hypothetical20% reduction of only large-WGS profile shares corresponds to about
5.1% aggregate decoder latency reduction; waiting time/overlap make that a
scenario, not a prediction. The remaining gap still needs another substantial
improvement. The latest papers justify the resource hypothesis, not its success.
