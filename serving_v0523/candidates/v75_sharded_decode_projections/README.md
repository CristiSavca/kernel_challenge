# V75 sharded Hopper decode projections

V75 adds V61's column-parallel BF16 pooler and unpooler to V72's tree
verification, grouped Humming experts, activation multicast and direct
all-reduce workspace output. Each TP rank computes its output columns over
the complete reduction dimension, then the existing gather restores column
order. The combined model passes the Hopper projection selection through
both the direct-workspace and ordinary expert-output paths.

- Pinned SGLang commit `297e54fd105e974cdf401e5b4d20253a22c16f90`.
- Parent V72 SHA-256 `e90c47ab73e66158b678893bb6fc7950c3b52f38cc3213ac6d7b7716d1b19705`.
- Frozen V75 SHA-256 `fe02db5dc89b286b914766c0a4079feba0027cbdfe960817ca648d885920bdaa`.
- Patch size 137,577 bytes across 26 source files, maximum added line 106 characters.
- CPU and independent source review passed. H200 compilation, replay and performance remain unverified.

Use the V72 launch flags:

```text
--flag swa-full-tokens-ratio=0.25
--flag moe-runner-backend=humming
--flag speculative-moe-runner-backend=marlin
--flag enable-layernorm-sp
--flag speculative-dflash-tree
```

The Hopper projection switch
`SGLANG_ENABLE_MUSE_SPARK_HOPPER_DECODE_GEMM_AG` defaults to true.
Setting it to zero selects V72's projection dispatch. Admission requires
compatible Hopper multicast communication, TP4/EP4 or TP8/EP8, supported
BF16 projection geometry and decode or target verification. Guards cover
compiler tracing, deterministic or batch-invariant execution, sequence
sharding, scattered input, overlap and additional parallelism. Prefill
continues through V72's sequence-parallel entry point.

For TP8 and 62 routed layers, sharding both 8,192-by-4,096 BF16 projections
removes 112 MiB of logical projection-weight reads per layer and rank,
or 6.78 GiB per verification pass. At eight target rows, the incumbent
unpooler already shards its columns, leaving 3.39 GiB of incremental weight
savings from the pooler. At 256 rows, the two gathers communicate up to
325.5 MiB of incoming data per rank over those layers. The corresponding
ideal H200 bandwidth scales are roughly 1.52 ms for 6.78 GiB of HBM traffic
and 0.76 ms for 325.5 MiB of peer traffic. Kernel launches, synchronization,
cache behavior and achieved bandwidth determine the net effect.
[V61 component arithmetic](../v61_hopper_decode_projection_gather/README.md)

The projections retain the full K accumulation and BF16 output boundary.
The gathered rank slices restore canonical channel order before the next
operation. Expert normalization, routing-slot order and collective
accumulation use their inherited implementations. Different GEMM geometry
can affect floating-point accumulation order, and the existing gather uses
its incumbent signed-zero sentinel convention. Judge agreement remains a
required hardware check.

V61's projection gathers use the symmetric allocation's push plane and
alternating phases. V71's direct expert and attention reductions use the
pull plane. Projection reservation reconstructs both regions before graph
capture, and direct reduction borrows the current pull view. Attention
output is consumed by the gated norm boundary and expert output by unpool
before another collective can reuse that plane. The router side stream
computes routes, and its main-stream join precedes expert execution.
[Independent review](independent_review.md)

The TP8 reservation grows final push storage by 14 MiB relative to the
128-KiB slots. The fused gather-plus-norm boundary remains limited by its
existing single-slot capacity. With the one-MiB slot and 8,192-wide
unpooler it applies through 64 rows. Larger admitted shapes gather into
the ordinary full output before the norm boundary. V64's measured 16-user
peak was stock plus 1.7 GB. V75's complete graph pools, post-capture KV
sizing and measured peak need validation against the two-GB limit.

The [CPU validation summary](cpu_validation.md) records the combined
component runs and fixture adaptations. The independent review exercised
384 model/collective combinations, ten projection fallbacks, 2,048 gather
shapes, 270,336 counter assignments and 2,000 mixed collective uses. The
source and CPU models support the ownership and dispatch contracts.
H200 JIT, multi-GPU progress, changing-batch tree replay, numerical and
later-turn agreement, exact requested output lengths, peak memory, TTFT
and throughput remain open. V75 has no measured serving speedup.
