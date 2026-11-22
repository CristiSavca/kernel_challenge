# V69 Hopper activation multicast for grouped experts

V69 shares one BF16 input tile between two Hopper thread blocks that compute
adjacent output tiles of the same grouped expert GEMM. It applies to the W13
and W2 projections in V68's grouped decode and target verification path. The
explicit Humming backend and inherited grouped-decode gate enable the path.
Set `SGLANG_ENABLE_MUSE_SPARK_HUMMING_DECODE_A_MULTICAST=0` to select V68's
decode tuning without changing the backend.

- Base SGLang commit `297e54fd105e974cdf401e5b4d20253a22c16f90`.
- Parent V68 patch SHA-256 `6248a58842e1a22c1cda5a957d5ab464a6ec824efcdf8308bdcc9c6a1046e505`.
- Exact V69 patch SHA-256 `ab4d289b3a4f7ced6695d69bdf3fe1d00c36e0ff90c03e5717804a887ef8f29f`.
- 86,388 bytes across 15 permitted files. The V68 delta changes two Python files.
- V68 flags `moe-runner-backend=humming`, `speculative-moe-runner-backend=marlin`,
  `enable-layernorm-sp`, and `swa-full-tokens-ratio=0.25`.

## Scheduling and numerical boundaries

The load-time guard requires SM90, an even visible SM count of at least two,
TMA A loads, warp specialization, a transaction barrier, and aligned output
width for pairs of N tiles. An explicit nonzero tuned SM count must also be
even. The selected decode variants compile through Humming's existing
preparation step, before graph capture. Indexed, large prefill, deterministic,
and Marlin draft execution continue through their existing tuning.

In Humming 0.1.12, `scheduler.cuh` assigns each two-CTA cluster one logical
MN tile. Both ranks therefore see the same expert ID, compact row offset,
row tail and K traversal. Only their output N tile IDs differ. The loader
issues the A transfer on cluster rank zero with both ranks in the mask. Each
rank expects the full A bytes in its transaction barrier. Before reusing the
A stage, rank zero waits for its local math readers and rank one's remote
arrival. It consumes all such stage arrivals before a tile transition and
publishes the next first-stage A through the distinct first-stage load
barrier. This handles empty ranks and partial expert groups under the same
live-suffix routing schedule. The CUDA launcher uses the current stream and
supplies the cluster attribute to `cuLaunchKernelEx`.
[Humming 0.1.12 release](https://pypi.org/project/humming-kernels/0.1.12/)
[CUDA thread-block clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)

The cluster preserves one CTA per original output tile and uses the same
K sequence, FP32 accumulation, NVFP4/E4M3 weights, BF16 scale rounds, and
SiLU and finalization code. No extra weights, activation tensors, or GPU
launches are required by the delta. B multicast is excluded because the
pinned grouped scheduler counts M tiles independently per expert and then
would multiply the global M tile index by the cluster size.

## Traffic estimate

For EP8 with 32 local experts, W13 and W2 each have 32 output tiles at the
current decode geometry. Each scheduled M tile issues approximately
`tile_M × 4096 × 2 × 32 × 2` BF16 bytes over both projections before
zero-filled descriptor tails. Multicast halves that issued A request
volume and preserves 27 MiB of logical packed weight and scale loads per
M tile. Under uniform independent top-eight routing, the expected volumes
per rank and sparse layer are:

| Target rows | M tile | Packed B MiB | A MiB before multicast | A MiB saved | Combined logical savings |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 193.8 | 28.7 | 14.4 | 6.45% |
| 32 | 8 | 551.2 | 81.7 | 40.8 | 6.45% |
| 128 | 16 | 849.2 | 251.6 | 125.8 | 11.43% |
| 256 | 24 | 863.7 | 383.9 | 191.9 | 15.38% |

Actual routing changes the number of M tiles. Repeated reads can hit L2,
the cluster still writes both shared-memory destinations, and paired
residency can reduce effective occupancy. H200 CUDA compilation, captured
replay, measured memory traffic, output agreement, throughput and peak
memory remain unverified. There is no measured serving score for V69.
[NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)

## Verification

The [CPU receipt](proof.json) records 4,617 configuration assertions using
the exact vendor SM90 tuner, 168 device/flag cases and 880 paired grouped
schedules. The scheduler model checks identical tile ownership and one A
request per pair for empty, single-expert, concentrated, mixed and tail
routing. A separate source review checked CTA pairing, phase order and
the first-stage transition. The CPU checks do not execute CUDA barriers or
compile cubins.

To repeat the CPU checks, apply the [patch](patch.diff) to the pinned checkout
and pass the resulting `muse_spark_humming_moe.py` and an extracted Humming
0.1.12 `humming/` directory to [cpu_check_v69.py](cpu_check_v69.py). Verify
the sdist SHA-256 recorded in the receipt. The official local
`okc serve diff` export matches the frozen bytes. Forward/reverse application,
whitespace checks, Python compilation and exact parent-file parity passed.

The V69 scratch and export checkout were removed after checking open files,
Git state and frozen artifact hashes. Allocated scratch was 46,936 KiB.
Available space rose from 46,442,060 to 46,489,012 KiB during removal,
an observed increase of 46,952 KiB.
