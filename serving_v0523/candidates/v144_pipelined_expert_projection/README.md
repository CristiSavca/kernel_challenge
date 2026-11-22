# V144 pipelined sparse expert projections

V144 issues packed NVFP4 weight, scale, and BF16 activation loads while the warp contracts an earlier K slice. It applies to V141's low-density target expert verification path. The gate/up projection looks one 16-channel slice ahead and the down projection looks two ahead. Each warp consumes slices in ascending K order, retaining the existing BF16 activation boundary and FP32 accumulators. The full 96,499-byte official patch has SHA-256 `c6f2938fae1d63325122f70a922b14e58d4f542f29b0de43e2be7079ab4cce09`. Physical H200 throughput and serving agreement remain unmeasured. [Official export](patch.diff) · [One-header change from V141](delta.diff) · [Manifest and four inherited flags](validation.json)

## Compiled load schedule

NVCC 12.9.86 compiled both complete TVM FFI wrappers for SM90 and SM90a. The table counts static SM90a instruction sites between each packed weight or scale load and the first instruction that waits on its encoded write barrier. The waits move later in the instruction stream, allowing the warp to issue other work while the memory request is outstanding. Data-dependent stalls, memory latency and cache behavior determine the actual time saved. [SASS measurements](sass.json) · [Disassembly](sm90a.sass.gz) · [Control-field decoding](https://github.com/cloudcores/CuAssembler/blob/master/CuAsm/CuControlCode.py)

| Projection | First encoded wait, V141 | First encoded wait, V144 | Hot-loop sites, V141 → V144 | Registers/thread, V141 → V144 |
| --- | --- | --- | ---: | ---: |
| W13 gate/up | 24, 26, 138, 42 | 156, 151, 146, 156 | 328 → 327 | 63 → 78 |
| W2 down | 14, 16 | 150, 150 | 170 → 189 | 40 → 56 |

Both builds issue eight HMMA instructions in W13 and four in W2 per steady iteration, with zero spills. The extra registers reduce the theoretical register-limited residency for 64-thread CTAs from 16 to 12 for W13 and 25 to 18 for W2. Code size grows from 984 to 1,320 instruction sites for W13 and from 600 to 976 for W2. NVCC emitted identical machine words for V141's separate 5,584-instruction medium-density path and the 440-instruction low-density planner. [Compiler and binary fingerprints](sm90.json)

With uniform distinct top-eight routes, one and four concurrent native requests yield about 7.18 and 20.41 live experts per rank. At 32 channel bands per live expert, that gives about 230 and 653 useful CTAs per projection. Both cases can leave little independent work available per SM. Their modeled cold packed-weight payloads of 193.8 and 551.2 MiB imply ideal H200 bandwidth floors of 42.3 and 120.4 microseconds. The pipeline changes when those bytes are fetched. The larger W2 loop and the lower residency may offset the latency benefit. [Low-density work model](../v129_routed_projection_pair/work_model.json)

## Correctness and reach

The production warp bodies passed six portable coordinate and numerical cases including 4,096-channel K, partial route tiles, all-remote routing, and two order-sensitive cancellation fixtures. A separate schedule audit checked 132,096 slices and 1,769,472 pointer bounds across 256 widths and modes. The production adapter passed 72 startup and 81 dispatch cases including all four JIT availability combinations. Both the official patch and parent delta reconstructed their expected trees and reversed cleanly. [Numerical checks](warp.json) · [Independent review](independent_review.md) · [Dispatch](dispatch.json) · [Export validation](validation.json)
