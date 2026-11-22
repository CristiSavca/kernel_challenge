# V71: produce directly into the multicast reduction workspace

V71 removes the two staging copies around Hopper's existing multicast
all-reduce for eligible captured Muse decode and verification shapes.
The attention output projection and grouped expert finalizer write their
BF16 results directly into the existing symmetric pull buffer. The unchanged
reduction kernel runs in that buffer. The gated attention boundary or unpool
projection immediately consumes the result before the next collective.

- Base SGLang: `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Exact parent: V68, `6248a58842e1a22c1cda5a957d5ab464a6ec824efcdf8308bdcc9c6a1046e505`
- Patch SHA256: `31f15bb4e125f8574efe2ae2e2ed83f385825a47c4395716b55d7fa65b0d24e9`
- Payload: 96,956 bytes, nineteen permitted source files, maximum added line 106 characters
- Same Humming, Marlin-draft, LayerNorm SP and SWA ratio flags as V68
- Feature switch: `SGLANG_ENABLE_MUSE_SPARK_WORKSPACE_REDUCE`, default true

## Arithmetic and ownership

The explicit mutating FFI entrypoint checks that its input is exactly this
rank's pull-buffer base, with the required alignment, dtype and capacity.
It invokes the incumbent `all_reduce_2shot_pull_kernel<MC>` with the same
block count, 512 threads, packed-vector type, multicast reduction and two
barrier phases. The device reduction implementation is byte-for-byte unchanged.
The ordinary out-of-place custom op is never given this borrowed tensor.

The attention producer accepts only the unquantized auto/torch BF16 GEMM
route with no bias. The expert producer is the V66 grouped finalizer with
the same normalization, BF16 scale boundaries and original routing-slot sum.
If it falls back, it returns its ordinary allocation and retains its normal
all-reduce. The unpool projection is required so borrowed MoE output cannot
escape the immediate consumer. The decoder owns the borrowed attention view
and consumes it before entering the FFN.

Guards retain the original path for other devices, dtypes, communicator
policies, FlashInfer or other collective precedence, explicit AR overrides,
determinism, compilation, eager execution, prefill, DP/CP/DCP/PP/SP, and both
single-batch and two-batch overlap. The original pull-buffer views are retained
after workspace construction; a reserve-induced reconstruction refreshes them.
No extra GPU allocation or multicast workspace is introduced.

On each rank, producer, reduction and consumer use the same CUDA stream.
The unchanged final acquire barrier orders every peer's multicast stores
before the consumer. The next collective's entry barrier follows each rank's
next producer, which follows its prior local consumer. PDL consumers must
wait for the producer grid before reading the borrowed result; the existing
Muse norm kernel does so. [NVIDIA PDL synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

## Opportunity and checks

At TP8, attention output is eligible from 32 rows and the 4096-wide expert
output from 64 rows under the pinned 512 KiB multicast crossover. At 128/256
rows, both reductions qualify. Across 62 routed layers this removes up to
248 copy launches and about 744/1,488 MiB of logical read-plus-write traffic.
The ideal HBM scales are about 0.16/0.33 ms; launch gaps and actual cache and
transport behavior determine any additional gain. This is a component saving,
with no measured whole-serving speedup. The retained V44 one-user log confirms
that the judge's Hopper communicator has multicast enabled.

The official exporter reproduced the exact patch. Python syntax, source
boundaries, whitespace, and reverse application pass. The executable
[CPU check](cpu_check.py) passed 387 actual selector cases, 28 helper rejection
cases and three alias/mutation checks, and verifies that the CUDA device
reduction body is unchanged. V66's routing/numerical suite and V64's 54-case
prefill integration passed against this source. The independent reviewer
executed 48 actual caller traces and checked two-use phase ordering for TP4
and TP8. [Independent review](independent_review.md)

These are CPU and source proofs. CUDA compilation, captured replay, pointer
and stream behavior on H200, output agreement, memory peak, TTFT and speed
remain unverified. V71 is a reserve component; the scheduled V68 check is
unchanged. V72 combines this source with tree speculation and A multicast.
