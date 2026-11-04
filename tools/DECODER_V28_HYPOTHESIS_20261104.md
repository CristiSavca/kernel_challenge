# V28: private boundary reduction in the public decoder

November 4, 2026. Root approved one fixed implementation and one private whole-decoder check after independent review. No device result exists for V28. This is an atomic-service hypothesis, not a forecast of first place.

The exact parent is `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`, public job `7a8c4941741645ee` and private job `b2a773e2d7194de8`. Public MoE remains `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0` at 2.373691x. The serving V26 result, 1.685110x against stock, is a different component and is not the public incumbent.

The two saved current-MoE hardware routes have **only two contributors per split unit**. They do not support a high-contention claim. Public WGS already uses native vector `red.global.add.v4.f32`; the large V26 improvement over V25's scalar CAS is not evidence that this public vector-atomic implementation is slow. The experiment tests whether eliminating the atomic read/modify/write dependency and payload clearing outweighs extra owner loads, FP32 sums and range enumeration.

## Actual scope and priorities

Read `muse_decode_layer/{AGENTS.md,TASK.md}`, `muse_moe_w4a16/{AGENTS.md,TASK.md}` and the current research loop. The public profile is `experiments/candidates/muse_decode_layer/profiles/97af…179c2.7a8c4941741645ee.json`; its exact path and hash are in the paired JSON.

| Affected load | Token count | MoE share, sliding | MoE share, full |
|---|---:|---:|---:|
| decode 32 | 32 | 50.03% | 25.62% |
| verify 16 | 128 | 58.50% | 42.02% |
| verify 32 | 256 | 61.71% | 40.90% |

These are rounded profiler-time shares, not measured recoverable critical-path time. Profiles can include PDL waits and overlap and do not equal benchmark latency. Indices `4,5,8,9,10,11` select MAIN WGS and change; indices `0,1,2,3,6,7` select the existing small paths and are controls. All source-legal larger/ragged MAIN inputs use the same algorithm; there is no benchmark-index branch.

The 20:56 snapshot has public decoder 2.181183135x versus leader 2.576162868x: +18.1085% score, or −15.3321% normalized latency, would be needed to match that leader. A **stale 13:42 snapshot** had decoder rank 50 at 2.3628x, requiring +8.3265% score/−7.6865% latency. That is not a current top-50 cutoff or a guarantee of overall top-50 rank. Root owns current board checks.

For prioritization only, scale each affected benchmark by `1 - reduction * its_profile_MoE_share`, retain the other cases, then recompute the exact six-load 46:16 scoring formula. A 5% reduction in the entire affected MAIN kernel gives a modeled 1.2515% whole-score increase; 10% gives 2.5519%; 20% gives 5.3126%. Matching the stale rank-50 score would require about 30.04% MAIN reduction; matching the saved leader about 57.43%. Neither is supported by reduction-service evidence. The unrealistic deletion of all affected MAIN time is a 1.40775x scenario, not a physical bound. This change can make measurable progress without claiming that it closes the ranking gap alone.

## Fixed code change

Candidate: `muse_decode_layer/candidates/decoder_private_partials_v28/submission.py`.
Frozen SHA256: `d1b1bcf5c241710697bc0d2fb39836b1d7064c262bff7f820b564085e2136d9b`.

Parent anchors: WGS layout around lines 7695–7701; `wgs_flush` around 8076; split store in `wgs_seg` around 8258; `wgs_phase` around 8274; weighted range functions around 8320–8360; host `doneAs` offset around 8894; `_ma_ext` around 8990; Python scratch constant around 12922.

1. A split fragment writes an immutable private `float4` payload indexed by phase, CTA, one of two boundary units, warpgroup, token tile and lane. Whole units retain their original epilogue and never touch this payload. The existing phase-A rotation cannot create a third boundary unit.
2. Keep the inherited deferred publication, 128-thread warpgroup barrier, release counter, owner acquire and counter reset. Phase A still elects the final arrival; phase B still uses the first-stage designated owner. No new global barrier or launch is added.
3. The owner enumerates increasing CTA ranges from the first to last stage owner, skips empty/nonintersecting ranges, loads each contributing private payload and adds FP32 values in that deterministic order. Two token tiles use eight FP32 sum registers at a time. Payloads are not cleared; every consumed element is overwritten and published within the current invocation. No slot is reused for another unit before kernel completion.
4. Preserve the public reduction's RN/FTZ policy with explicit `add.rn.ftz.f32`. The initial draft's copied V26 non-FTZ addition was corrected before any measurement. Global `red.add.f32` flushes subnormal inputs/results to signed zero; the owner sum must not silently import serving-specific non-FTZ behavior. [NVIDIA PTX `red` documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red) supports this policy. Deterministic addition order still differs from atomic arrival order and needs fresh GPU numerical validation.
5. Double the partial payload reservation from 20 to 40 MiB, adjusting both host/device counter and `doneAs` offsets plus the Python scratch constant. No other state layout changes. All weights, KPERM packing, scale operands, BF16 boundaries, activation expressions, fused normalization/finalizer, routing, PDL, grid/block size, WGS ring depths, GS2/wait1/FCI3, cost tables and compiler flags remain the public parent's.

The module cache name changes. An `OKC_WGS`-only macro renames the real MAIN function to `okc::moe_private_partials_v28<…>` without a wrapper. Small kernels retain `okc::moe_fused<false>`. No standalone-serving cooperative launch or serving output/finalizer is imported.

## Source-level costs on saved routes

These counts reconstruct the actual saved 247888 NCU routes, not the decoder's unknown routes. They use the parent's expert-order fat jobs, then skinny jobs, fixed cost tables, grid 132, and exact weighted stage boundaries. Counts are **dynamic per-thread source operations**, not emitted SASS instructions, transactions, timing or a claim that all traffic reaches HBM.

| Both phases combined | Saved M32 | Saved M256 |
|---|---:|---:|
| Jobs | 16 | 27 |
| Split units / fragments | 240 / 480 | 251 / 502 |
| Contributors per split | 2 | 2 |
| Parent vector atomic payload operations | 245,760 | 647,168 |
| Parent owner vector loads | 122,880 | 323,584 |
| Parent owner vector zero stores removed | 122,880 | 323,584 |
| Candidate private vector stores | 245,760 | 647,168 |
| Candidate owner vector loads | 245,760 | 647,168 |
| Candidate scalar FP32 additions | 983,040 | 2,588,672 |

Private stores/readback cover 3,932,160 bytes each for M32 and 10,354,688 bytes each for M256. Parent atomic service includes a hardware read/modify/write whose physical traffic/cost cannot be inferred from these counts. The new code doubles the owner's payload reads and adds range-search/address instructions; it can lose even with fewer atomic operations. Saved DRAM reads are already only about 1.008x nominal packed weight requests. This is not a weight-bandwidth reduction or evidence of a DRAM stall bottleneck. New compiler register/local allocation and cache footprint remain unknown.

## Why this is distinct, and its falsifier

Closed compact/NW4/R16 variants changed geometry, occupancy or work decomposition; shared-A added large shared transport; wait/register/prefetch variants changed the pipeline; SC6 changed scale packing; scalar-finalizer and attention experiments removed different work. This changes only split-reduction ownership and storage. No prior public 247888/97af private-partial test was found in the reviewed ledger. V26 provides reusable contributor/publication code and a tested design precedent; its serving result supplies no public speed estimate.

The strongest competing explanation is that native vector atomic service already hides behind the inherited deferred flush, while extra private loads, sums, range enumeration and scratch footprint slow the owner. The frozen whole-decoder check distinguishes these outcomes. A failure closes this fixed candidate; no quota, order or geometry sweep is bundled as a rescue.

Root's pre-result gates compare against **both** exact-97af jobs. For each load first compute its 46:16 weighted latency, then its normalized speed ratio:

- Affected three-load GM speed ratio ≥1.03; each of the six affected case latency ratios ≤1.03.
- Control three-load GM latency ratio in [0.97,1.03]; each control case latency ratio ≤1.03.
- Official six-load GM speed ratio ≥1.015.
- Clean 13 tests, 12 benchmark checks, 12 profiles, evaluator/source binding and rules checks; unique MAIN activation and unchanged control symbols.

The 1.015 whole floor is intentional; it is slightly stricter than `sqrt(1.03)` with unchanged controls. The screen is operational, not a statistical significance claim; paired CI, different fresh seeds and compiler/PDL profile limitations must remain visible.

## CPU evidence and limits

`build.py` reconstructs the candidate from pinned 97af and the pinned V26 reduction body. `edits.json` reverses the complete candidate byte-for-byte to 97af. Only the `MOE_CUDA_SRC`, `_ma_ext` namespace and `WGS_FIXF` Python AST nodes change. Fourteen device bodies, including full `moe_fused`, producers, routing, phase/range functions, dequantization and epilogue, remain exact. The partial-MMA prefix and publication/reset blocks are exact. Six small-build and two explicit non-WGS control preprocessings match the parent.

The native CPU test executes the actual candidate's range/phase/private-store/owner snippets on 92 cases, 17,347 partial contributions, 5,919 split units and 70,863 vector loads, checking 283,452 FP32 values with poisoned unreadable regions. It covers NTT1–8, rotated execution, zero-length ranges, small unit spans, mixed jobs and three bad source mutations (wrong boundary, lost range carry, reading empty ranges). Actual payload-pointer slices are disjoint over phases/CTAs/boundaries/warpgroups. Directed positive/negative subnormal and subnormal-result checks bind the RN/FTZ shim. Representative WG0/WT0 arithmetic plus affine slice coverage is not exhaustive GPU arithmetic or memory-order proof. Oversubscribed empty-intermediate-range fixtures are explicitly outside the host launch domain. The inherited all-remote route behavior is not newly proved.

Final GPU compilation, resource allocation, CUDA publication/progress, numerical tolerance and performance remain unproved. Root owns one official private check only after independent source GO and independent CPU reproduction.
