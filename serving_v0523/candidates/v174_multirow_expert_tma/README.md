# V174 shares expert weights across up to 32 verify rows

V174 runs one gate/up kernel and one down kernel for eligible medium-density verification. V173 launched a separate grid for short and long expert lists in each projection. At hidden and intermediate widths of 4,096 and 32 local experts, V174 launches 2,048 CTAs across the pair, down from 4,096. Each expert keeps its ordered list of active rows. One producer warp fetches a packed 128-channel weight panel, its scales and the BF16 activations, then eight consumer warps apply that panel to up to 16 gate/up rows or 32 down rows. The work allocated to each expert still includes every K dimension and every requested destination. [Full patch](patch.diff) · [Delta from V173](delta.diff)

The compiler confirms that each warp decodes an operand once and reuses it across row groups. Eight K steps issue 128 paired BF16 multiplies in the gate/up consumer loop for either 16 or 32 MMA instructions. In the down consumer loop, 64 paired BF16 multiplies feed 8, 16, 24 or 32 MMAs. A 17–24-row down tile therefore avoids the fourth padded MMA group used by V173. All 20 retained functions across the two CUDA modules have identical machine words on SM90 and SM90a. The new W13/W2 kernels use 53/55 registers, 45,600/35,872 bytes of static shared memory, and zero spills. [Native compiler and loop counts](sm90.json)

## Where the work changes

| Per-rank route distribution | Parent packed-weight bytes on the hot path | Nominal BF16 input loads saved per projection | Down-projection MMA groups saved |
| --- | ---: | ---: | ---: |
| 1,024 total slots, uniform distinct top-eight | 1.99% | 1.49 MiB / 4.4% | <0.01% |
| 2,048 total slots, uniform distinct top-eight | 40.88% | 34.87 MiB / 35.3% | 0.22% |
| Archived 256-row standalone H200 histogram | 54.72% | 73.75 MiB / 45.5% | 3.39% |

Both candidates request the same number of full packed expert weight sweeps. V174 cuts duplicate activation requests on hot experts by staging one copy for each 128 output channels. Those requests can already hit cache on V173. The new nine-warp CTA has a register-residency ceiling of four blocks per SM, compared with eight two-warp blocks for V173's hot kernel. Small down projections also go from three buffered panels to two. The 32-user geometry exposes more hot experts, while the 16-user geometry mostly tests the saved launches against the shorter down-prefetch window. The recovered H200 histogram comes from a different standalone kernel. The scored serving run reports no expert histogram. End-to-end throughput, physical output agreement, latency and peak memory remain unmeasured. [Reproducible model and data](work_model.json) · [Model source](work_model.py)

## Verification

The native host selector issued exactly one projection launch in all 25,600 tested cases, including 23,680 ordinary vector fallbacks. The inherited Python preparation/dispatch fixture passed 108 layouts, including absent and stale descriptors. The normal low-load path and ordinary medium fallback keep their matched machine code. [C++ launch results](launch.json) · [Python dispatch results](dispatch.json)

Independent CPU execution matched 2,515,968 live BF16 values across 54 projection pairs. Protected pages, three asynchronous copy schedules and 136,917 deliberately stale positive tail slots exercised the unpadded planner. Six cancellation fixtures checked full-K ordering. Each transfer count matched a separate byte oracle. [Independent numerical and memory review](independent_review.md)

The official export SHA-256 is `7dcbcb538043c14a0c525a9a419c1d92b25dfa23666faae109b82b4df3cd1374`. The patch uses V173's five flags. Both complete and incremental exports applied and reversed to the pinned SGLang trees. The current published source, path, flag, character and size restrictions passed. [Validation and flags](validation.json) · [Challenge rules](https://openkernelchallenge.dev/api/v1/serving)

Reproduce the host checks with `launch_check.py STAGE`, the parent V173 `dispatch_check.py STAGE`, `independent_warp.py STAGE` and `work_model.py`. For the compiler reproduce the V173 scratch layout with the V173 and V174 official headers under `prior/` and `new/` in `/tmp/v174`. The pinned Git include archive, Linux NVCC 12.9.86, host compatibility includes and TVM-FFI layout are documented in the parent compiler receipt. Run the saved wrapper script and then `compiler_check.py /tmp/v174`. [Compiler commands](compile_sm90.sh) · [Compiler audit](compiler_check.py) · [Compiler inputs](../v173_medium_expert_tma/compiler_review.md)
