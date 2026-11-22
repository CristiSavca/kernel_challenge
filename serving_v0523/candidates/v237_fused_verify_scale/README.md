# V237 fuses the model scale into greedy verification

V237 removes a full read and write of the target’s FP32 vocabulary shard on eligible ordinary DFlash decode graphs. The released Muse target projects each verification row through the native BF16 head with FP32 output. V235 scales that entire result in place, then reads it again to compute softcapped winners. V237 performs the correctly rounded FP32 multiplication when its existing winner kernel loads a score. It preserves the original GEMM, softcap, per-tile reduction, tie policy and cross-rank winner exchange. V91 changed that reduction and transport. V225 packed the separate draft top-16 exchange. [Three-file change](delta.diff) · [Prior target study](../v121_target_head_feasibility/README.md) · [Draft selector](../v225_packed_draft_candidates/README.md)

| Users | Target rows | Logical scale traffic removed per GPU | Transfer time at 4.8 TB/s |
| ---: | ---: | ---: | ---: |
| 1 | 8 | 1.54 MiB | 0.34 µs |
| 4 | 32 | 6.17 MiB | 1.35 µs |
| 16 | 128 | 24.66 MiB | 5.39 µs |
| 32 | 256 | 49.33 MiB | 10.78 µs |

Each rank still projects all 25,256 vocabulary entries in its 394.625 MiB head shard. The table divides logical bytes by peak bandwidth. Cache residency, launch overhead and contention determine elapsed GPU time. This change alone provides no evidence for 2× or 3× serving speed. [Resource model and historical decode periods](work.json)

The captured target output records the exact model scale. The native full-graph replay preserves it when slicing padded rows. Greedy verification consumes it once. When sampling, penalties, grammars, logprobs or other adjustments need the complete vocabulary, the worker creates a fresh scaled shard before the original full gather. This preserves shared graph storage and can add one transient FP32 shard, up to 24.66 MiB per GPU at 32 users. Eligible greedy calls add no persistent allocation. Eager, mixed and instrumented paths retain ordinary scaled output. The same behavior applies on recapture and on legacy untagged outputs. [Graph, fallback and instrumentation checks](independent_graph.md)

Sixteen Triton 3.8 SM90 compilations assembled for SM90a with zero spills. Unscaled variants produced the same PTX instruction streams as V235. Scaled variants retained an explicit FP32 round-to-nearest multiply before the original division and CUDA tanh. A paired CPU replay used the Triton interpreter on 3,002,800 live scores and matched all 480 local-winner rows. The interpreter used host implementations of FP32 multiplication and tanh. Its native all-NaN reduction cannot execute an all-NaN tile in the ordinary mode, so that case retained one finite lane. The first-NaN mode exercised completely NaN tiles. [Compiler evidence](sm90.json) · [Interpreter results](interpreter.json)

The official exporter produced a 392,694-byte patch with two context lines. Its SHA-256 is `3523359b53eb4f9ee237fa328e9db09257060eef6f741434f8bf2727b1f7b9b5`. The full patch and three-context donor delta reconstructed and reversed the exact pinned trees using ordinary Git. V235’s eight launch flags carry forward. No H200 check ran for V237. [Frozen patch](patch.diff) · [Rule checks and source identities](validation.json) · [Inherited CLI registration](flags.json)
