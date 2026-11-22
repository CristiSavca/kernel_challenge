# V83 sharded target router and pooler

V80 already distributes the expert-input projection across eight GPUs. Its router still evaluates all 256 experts on every rank. V83 gives each rank 32 full-K router outputs and sends them with its local pooler columns in a single all-gather. The original routing operator receives the assembled FP32 logits in global expert order. Eligible decode and target verification use this path. Prefill keeps its incumbent dispatch.

At TP8, this removes seven eighths of the router matrix multiplication on the ordinary inference path. Across 62 routed layers it avoids up to 434 MiB of logical router weight reads per GPU per verification pass under a one-read-per-matrix model. At 128 target rows, the extra incoming router payload is 6.78 MiB per GPU across those layers. The complete weights remain resident because prefill uses them. The smaller GEMMs, pack and unpack, ordinary all-gather and join before top-k can cost more time than the work saved.

## Arithmetic and communication

The direct FP32 linear path retains every input feature. The split-BF16 branch retains the two BF16 weight components and adds their two FP32 GEMM outputs in the incumbent order. The sharded GEMMs can select different reduction trees or tiles. Near-tied logits therefore need end-to-end route and token checks.

The side stream computes local logits while the main stream projects the expert input. The main stream joins before packing, gathers rank-major int32 words and unpacks contiguous outputs. FP32 router words pass through unchanged. BF16 pooler words retain the multicast path's signed-zero behavior. A hook on one rank can use a separate full local result while every GPU participates in the same collective.

## Reproduction and checks

Use the seven [V80 flags](../v80_fused_experts_sharded_draft_filters/README.md) at the same request load. The additional target path defaults on under the guarded Hopper Humming configuration. The environment switch is `SGLANG_ENABLE_MUSE_SPARK_SHARDED_ROUTER`. The frozen [patch](patch.diff) is 157,318 bytes across 29 permitted files, SHA-256 `bae9bc7a0a26ea3aadbbef8c80cd5049e8dd078673a3986d4407e110050a9dc4`.

The [CPU harness](cpu_check.py) runs the production packer and router on real PyTorch CPU tensors and emulates the Triton unpack operations. Its split-BF16 GEMM test widens BF16 operands to FP32 on CPU. It exercised the full 8,192-feature reduction, rank-local hooks, strided inputs, signed zeros, NaN payloads and word order. All 175,808 words and 16 router comparisons passed, with maximum absolute FP32 difference 5.96e-7. The [independent review](independent_review.md) covered 27,043,652 additional transport words and the stream handoff. H200 compilation, collective progress, peak memory, token agreement and speed require official measurement.
