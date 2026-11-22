# V61 Hopper column-parallel decode projections

V61 extends Muse Spark's existing column-parallel BF16 pooler and unpooler
to Hopper target decode and verification on multicast-capable TP4/EP4 and
TP8/EP8. Each rank uses the existing BF16 GEMM dispatcher to compute its
output columns with the full reduction dimension. The existing gather
restores canonical column order. Routing, expert finalization, and the
original expert all-reduce still run in the same order. The draft, prefill,
Blackwell, deterministic, scattered, and sequence-parallel paths use their
incumbent dispatch. The existing gather's signed-zero sentinel convention
and possible GEMM accumulation-order changes apply.

- Pinned SGLang commit `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Exact parent V44 scored 1.1847 with job `c3b7bd142c264853`
- Patch SHA-256 `eab9720a002103d2deef924be266d474ebefd9a1e899cbfe171ef909e80f4800`
- Patch size 30,716 bytes across nine permitted paths
- Test flag `--flag swa-full-tokens-ratio=0.25`
- Hopper feature toggle `SGLANG_ENABLE_MUSE_SPARK_HOPPER_DECODE_GEMM_AG`

An 8192-by-4096 BF16 projection stores 64 MiB of weights. TP8 reduces
the weight reads from both replicated projections by 112 MiB per routed
layer per rank. At 62 routed layers the total is 6.78 GiB per verify pass.
At eight rows, the incumbent unpooler already shards the output columns,
so the incremental amount is 3.39 GiB. With 256 rows, gathering both new
projections brings 325.5 MiB from peers per rank over 62 layers. The ideal
H200 bandwidth scales for 6.78 GiB HBM and 325.5 MiB incoming NVLink are
about 1.52 ms and 0.76 ms respectively. Kernel launch, synchronization,
cache behavior, and achieved bandwidth determine the net effect.
[Model dimensions](https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/meta-models/Muse-Spark-open-816B-A42B)
[H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)

The gather concatenates every rank's slice in one phase of the existing
two-phase push workspace. At the 512-row cap that means 1 MiB per slot on
TP8, which adds 14 MiB of final push storage over the tuned 128 KiB slots.
The all-reduce thresholds and arithmetic are preserved. One consumer
resets each gather vector, and the grid stays at or below the SM count.
The CUDA source uses the incumbent SM90-compatible multicast store.
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red)

CPU validation and an independent source review passed. The [test receipt](proof.json)
covers selection, numerical rank ordering, gather address bounds, and alternating
all-reduce/gather phases. GPU JIT compilation, device synchronization, judge
numerical agreement, peak memory and performance remain unverified. V61 is
a reserve because it addresses decode projection work alone.

Removed the closed `okc-v61-projection-r8vEoM` staging directory after checking
its contents, alternate Git index, and open-file state. It held nine materialized
source files, two CPU test scripts, and the index, totaling 1,760 KiB. Available
space on the shared volume changed from 35,756,372 KiB to 35,758,092 KiB.
The frozen patch preserves every candidate source change.
