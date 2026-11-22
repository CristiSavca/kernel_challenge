# V55 token-sharded BF16 prefill projections

V55 adds one model hunk to V50. On ordinary replicated Hopper prefill with a
multicast-capable NVLink TP group, each rank computes the expert pooler and
unpooler for a contiguous shard of the token rows. The existing TP all-gather
restores their original order. The projections retain the BF16 GEMM dispatcher,
the attention input is still replicated, and the existing expert sum runs
before the unpool projection. Small, speculative, sharded, context-parallel,
and deterministic execution takes the existing path.

- Base SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Patch SHA-256 `5ca9d26522ba84bc183537f819811962cac21de546244a48bf86a3a60dc5543e`
- Patch size 32,253 bytes across six allowed source files
- Serving flag `swa-full-tokens-ratio=0.25`

At 16,384 tokens, the two projections perform about 2.20 TFLOP per layer per
rank in V50. With TP8, V55 performs one-eighth of that work and each rank
receives 336 MiB from both gathers. An ideal-rate illustration using H200
dense BF16 and unidirectional NVLink peak gives about 1.2 ms per layer,
or 72 ms per full chunk, available to save. The savings depend on achieved
GEMM and gather bandwidth. The smaller GEMM can change cuBLAS accumulation
order, and the extra collectives can change prefill graph startup and peak
memory. [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)

The exact patch passed forward and reverse application, whitespace, path and
line limits, and all five Python source compiles. The extracted helper matched
the replicated GEMMs in 18 CPU rank/order cases. The extracted forward
dispatch passed 13 fallback cases. An independent source audit reviewed
replica ownership and the existing graph-aware group gather, after which
the ordinary-extend, attention-TP, sequence-sharding and multicast guards
were tightened and retested. Temporary alternate indexes were removed
after confirming they were closed. No CUDA compilation, NCCL runtime
result, or judge measurement has been observed.

For a private 16-user check, compare same-run stock-normalized throughput,
TTFT, agreement and peak memory against V50. Consider pooler-only row
sharding if gather time or memory is limiting.
