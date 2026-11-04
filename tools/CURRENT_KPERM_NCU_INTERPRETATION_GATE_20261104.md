# Current KPERM hardware diagnostic: interpretation fixed before results

2026-11-04. Read-only analysis; no candidate, allocation, provider query, kernel execution or profiling change. The forthcoming run is diagnostic evidence, not a performance submission. This note does not reopen SS-A, REG112, cache prefetch, SC6 or the final-scalar experiment.

## Provenance and comparison limits

Current source: `experiments/candidates/muse_moe_w4a16/247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0.py`; embedded CUDA `85fb417494c6eab5baabad6813275b256f7bae84c0329d0e94cb61a15030445c`.

Retained old measurement: `experiments/iterations/20261104-moe-hardware-profile-recovered.json`, SHA256 `0921485edb8a6494302288956fe1ef3614e7980f5fbf84fd831f27977a4c36b7`, recovered offline from immutable `experiments/modal/20261104T054636Z-ea14816c/result.json`. Its source was `ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a`, native CUDA `af4f0dba97e347f8afa8f27508e40d5c03f60b402825cadba9a9537700b20d6e`.

Require the new exact source/native hashes, numerical checks, actual route counts, H200 identity, image/tool versions, requested-versus-returned metrics with finite values/units, exactly two launch identities, actual replay-pass counts, and sandbox termination evidence before interpreting it as a complete measurement. Matching seeds alone do not establish matching routes. Missing optional counters remain missing; no substitute or second run is implied.

The old run used NCU 2025.3.1, four replay passes per launch, cache-control all, uncontrolled clocks, PDL=False and a 128 MiB read-only sweep. Preserve differences explicitly. These are separate measurements, not a controlled paired timing experiment. Neither ordinary timings inside the NCU-injected process nor profiled durations establish an official speed change or decoder overlap. Parent WGMMA serialization remains unknown.

## Old evidence, not a speed forecast

Here R is actual local token-expert pairs, U active local experts, J the FT64 job count, and padded rows are the sum of each job rounded to eight rows.

| Quantity | M32 | M256 |
|---|---:|---:|
| R / U / J / padded rows | 39 / 16 / 16 / 128 | 354 / 26 / 27 / 456 |
| Job histogram by NTT | 1:16 | 1:16, 2:5, 3:2, 4:1, 5:1, 8:2 |
| Nominal packed weight requests, J × 27 MiB | 452,984,832 B | 764,411,904 B |
| Actual DRAM read / write | 456,964,352 / 3,375,872 B | 769,691,136 / 4,264,192 B |
| Read / nominal request bytes | 1.008785 | 1.006906 |
| Profiled duration, contextual only | 128.576 µs | 214.304 µs |
| DRAM throughput (% sustained peak) | 74.43 | 75.06 |
| Issue active (%) / eligible warps per cycle | 59.38 / 1.40 | 59.48 / 1.39 |
| Tensor active (%) | 11.22 | 23.71 |
| Barrier / short-scoreboard / long-scoreboard / wait (%) | 7.49 / 1.89 / 17.97 / 8.11 | 4.89 / 2.35 / 12.81 / 9.58 |

The old read volume is already close to the packed-weight requests. Requested activation bytes cannot be treated as uncached HBM traffic. Eliminating the one repeated expert job at M256 would remove at most 1/27 = 3.704% of these nominal weight requests; M32 has no such repeat. This is a byte scenario, not a latency bound, and does not authorize reopening FT128.

## Decision rules for the new counters

1. Recompute `27 MiB × J` and `27 MiB × U` from actual routes; compare read volume to both. If excess remains small, do not propose large HBM savings from activation caching. If excess grows, quantify it before attributing it to any source path. Reads below nominal requests can reflect cache reuse and do not establish omitted work.
2. Interpret math-pipe, MIO and LG backpressure together with issue activity and eligible warps. They describe aggregate scheduling pressure, not a source PC or critical path. Do not sum stall percentages, multiply them by duration to invent recoverable time, or treat a low counter as proof instructions are free. NVIDIA specifically limits stall analysis to cases where schedulers fail to issue; replay and metric instrumentation also affect measurement. [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
3. The three proposed backpressure counters were not collected in the old run. Report current values without inventing an old-to-new delta. Math pressure alone does not identify decode/FMA/SFU work; MIO alone does not establish shared-memory bank conflicts; LG pressure is not equivalent to DRAM saturation.
4. Low tensor activity with sparse NTT1 jobs is compatible with limited arithmetic intensity. Our source already chooses NTT1..8 from actual `S.jC`; it does not execute a fixed 64-row tensor tile for every sparse job. Padding counts are useful work descriptors, not automatically removable arithmetic.
5. If the new optional group is absent or does not support an instruction-pressure investigation, conclude no new source target from this run. This is an allowed result.

## One conditional source-inspection target

Current `wgs_kperm_x` at source lines2883–2889 constructs each 16-byte chunk from four `__ldg(unsigned*)` loads. The pre-KPERM gather used one `__ldg(uint4*)`. Current `wgs_gather` invokes it for each real routed row; padding rows are counted but not loaded. Both versions move the same 8,192 bytes per actual row into the gather, with an equal-size output store. This is a source-level instruction difference; generated instructions and its exposed cost are not yet established.

For the retained routes, each direction is 319,488 B at M32 and 2,899,968 B at M256. Assuming the source loads lower separately, current gather has 64R versus 16R warp load instructions: 2,496 versus624 at M32 and22,656 versus5,664 at M256. These are counts, not measured instructions or timings.

Only if current LG pressure plus issue/eligibility warrants further investigation: inspect the exact lane permutation to determine whether a coalesced vector load followed by register exchange can construct the same KPERM bytes with less queue demand. This would add shuffle/address work and may worsen MIO or dependency pressure; the gather is a small fraction of total weight traffic. Aggregate counters cannot isolate it from epilogues/fixups/metadata, so they would justify a bounded source/compiled-instruction inspection, not a candidate or a performance claim. Keep the existing exact KPERM byte mapping, BF16 values and publication contract.

## RaMP v2: useful features, no new supported selector

[RaMP v2, revised 2026-10-30](https://arxiv.org/html/2604.26039v2) ranks a six-configuration menu using current expert counts, derived tile/wave counts and active experts, with coefficients fitted from measured workloads. The menu shares an occupancy class. Its GPU selector replaces an existing grouping kernel. The H200 implementation uses FP8 with 128×128 block scales; section3.3 says the kernel will be released later. Its reported speedups do not transfer to this W4A16 kernel.

The transferable question is whether two validated configurations have a routing-dependent crossover. Current source already forms jobs from actual counts, chooses NTT1..8 and partitions persistent stage work with phase-specific costs (`wgs_ranges`, line3252). It has no separate grouping launch to replace for free. A new selector would need actual competitive alternatives, compatible compiled resources and a bounded GPU publication path; the two diagnostic shapes cannot fit or validate such a model.

Therefore there is **no presently supported new route-aware specialization decision**. Record actual U, J, NTT histogram and padded rows alongside counters. A future independently justified configuration could be evaluated against those features, but neither a low-math/low-queue result nor padding alone creates a viable alternative. Do not manufacture a menu from the already rejected kernels or use benchmark identities, seeds, timing state or host reads of runtime routing.
