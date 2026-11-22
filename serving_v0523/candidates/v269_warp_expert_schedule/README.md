# V269 warp lookup for grouped decode experts

V269 uses one CTA barrier to prepare the tile schedule for each Humming W13 and W2 projection. V263 uses three. On the archived 256-row H200 route histogram, the old scheduler checks a shared-memory expert-count condition 4,976 times across a projection’s persistent CTA jobs. V269 resolves those same 1,056 jobs with one warp vote each. Each lookup executes in every participating warp. Hardware serving throughput, output-token agreement and elapsed kernel time are unmeasured. The official export is 409,283 bytes and applies cleanly to the published SGLang pin. [Patch](patch.diff) · [Rules verification](validation.json) · [Route replay](work.json)

The first warp loads the real expert offsets and computes an exclusive prefix of padded tile counts. A single CTA barrier publishes that prefix in the two existing shared-memory arrays. Producer and consumer warps look up the last expert whose tile start is at or below the job’s row tile. Choosing the last match handles consecutive empty experts. The scheduler uses the selected expert’s true row interval, including any remote prefix. V269 uses the native 32-expert grouped, warp-specialized, non-stream-K geometry. Other expert counts use Humming’s original scheduler. W13 and W2 evaluate the original routed rows, channels, K slices and reductions in the same order. The public model entry still selects the previous constructor fallbacks. [Production change](delta.diff) · [Startup cases](startup.json)

## Why the candidate retains weight scheduling

| Geometry per rank | W13 or W2 jobs | Incumbent condition checks | Packed weight savings if all repeated M tiles could share |
|---|---:|---:|---:|
| Recorded 32 rows, width 8 | 512 | 4,092 | 0 MiB |
| Recorded 256 rows, width 24 | 1,056 | 4,976 | 189 MiB |
| Modeled 16 users, 128 verification rows, width 16 | 1,006.4 expected | — | 0.00050 MiB expected |
| Modeled 32 users, 256 verification rows, width 24 | 1,023.7 expected | — | 0.00057 MiB expected |

The weight figures combine both projections over one layer. The modeled rows assume independent, uniform top-eight routing across 256 experts. The observed stress histogram had 354 local routes, 26 live experts and 33 row tiles. Giving one CTA each complete expert/output band reduces independent jobs from 1,056 to 832. Under the native cyclic assignment the slowest CTA then owns ten row tiles instead of eight. Perfect retention could remove 189 MiB of packed and scale requests at that skew, including 54 MiB beyond V216’s two-tile pairs. The separate standalone H200 implementation read 734.03 MiB from DRAM, 32.03 MiB above the 702 MiB unique weights. It used different kernels and a cold-cache counter replay. Neither the modeled requests nor those counters establish actual Humming HBM savings. [Replay and uniform assumptions](work.py) · [Prior two-tile multicast](../v216_decode_weight_multicast_feasibility/README.md) · [H200 profile](../../../experiments/iterations/20261104-moe-hardware-profile-recovered.json)

## Machine and correctness checks

| Native tile rows | W13 and W2 static CTA barrier sites, V263 → V269 | W13 / W2 shared bytes per CTA | Registers and spills |
|---:|---:|---:|---:|
| 8 | 15 → 13 | 82,944 / 91,136 | 96 registers, zero spills |
| 16 | 19 → 17 | 91,136 / 107,520 | 96 registers, zero spills |
| 24 | 23 → 21 | 99,328 / 123,904 | 96 registers, zero spills |

All 12 parent and candidate CUDA modules compiled for SM90a under NVCC 12.9.86. Each applicable residual width retains 32 BF16 HGMMA sites, eight drains and four ordered stage releases per K loop. The six V269 binaries each contain two new vote/most-significant-bit lookups for the producer and math paths. The per-CTA memory, register use, thread count and native one-CTA-per-SM launch agree between versions. The 132-CTA grid executes 264 fewer block-wide barrier executions per projection, counting each CTA separately. Warps now pay for a fixed vote and shuffle, and for reading the published bound during the persistent loop. The machine-code change alone gives no speed estimate. Two alternate register lifetimes spilled across all six modules. [Matched assembly](assembly.json) · [Discarded variations](rejected.json)

A separate C++ SIMT fixture executed the submitted scheduler on 288 route layouts and 24,368 CTAs, checking 4.8 million lane results across 150,000 tile visits. The input cases include both offset strides, remote prefixes, all-empty ranks, consecutive empty experts and multiple tiles per expert. The original K-tree fixture matched 1,978,368 FP32 words, including nonfinite values. The public grouped model fixture compared 20,234,240 staged and output words with identity projections across 320 rank-forwards. [Independent schedule](independent_schedule.json) · [FP32 numerical results](numerical.json) · [Grouped routing](dispatch.json)

V269 uses V263 as its parent. Combining V268 and this scheduler exports at 410,251 bytes against the 409,600-byte limit. The eight inherited launch settings are recorded in the verification receipt. V269 has no judge job or score. V64 remains the measured four-load control at 1.4119×. [V263](../v263_token_major_expert_decode/README.md) · [Measured V64](../../runs/abd217baf0ed49fd.json)

To reproduce, apply each official patch to the published pin in a separate scratch root containing the pinned Humming 0.1.12 package. Use the existing V213 renderer and compiler wrapper for each root’s six live W13/W2 sources. Place the resulting source, cubin, log and disassembly files in sibling directories named `parent` and `live`, then run `assembly.py` with their common directory. Run `independent_schedule.py` with the patched checkout and extracted Humming package directories, `numerical.py` with the scratch root, and `verify.py` with the patched checkout. [Renderer](../v213_live_decode_expert_reduction/render.py) · [Compiler wrapper](../v213_live_decode_expert_reduction/compile.sh)
