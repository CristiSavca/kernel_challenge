# V50 compact dense prefill finalization

Base SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`. Parent candidate V44
`0822e1279e17`. Patch SHA-256
`7ba1708e049dbfae89eac613efe0bdeb9b7c8ccbcf0c11859ccad49f7f4441ae`.
Keep `swa-full-tokens-ratio=0.25` and the other V44 flags for the initial comparison.

V44 writes each dense GEMM row to a sparse `[tokens, topk, hidden]` tensor, which
the fused finalizer reads in original slot order. V50 builds a compact int32
slot-to-row map from the same stable permutation, then calls that finalizer while
the GEMM rows are still compact. The finalizer applies the same normalization,
routing score, accumulation order and BF16 output cast. An all-inactive rank
writes zeros. Small batches and other finalizer callers retain the original
Marlin and Triton paths.

The break keeps V44's graph bridge contract. The preceding captured segment
allocates a fixed-size output, the eager body writes it in place, and the later
segment consumes the same address. The stub returns no dynamic tensor. Route
counts, mapped rows and temporary GEMM buffers remain local to each eager call.

For 16,384 tokens, top-k 8 and 4,096 expert output features, the fixed buffer
shrinks from 1,024 MiB to 128 MiB. The live set inside the dense break therefore
drops by about 896 MiB, with a sub-MiB inverse row map added. For 16,224 local routes the
eliminated scatter read and write total about 254 MiB per MoE layer. Graph-pool
reuse, KV-pool sizing, allocator reservations and runtime scheduling can change
the savings the judge reports. Estimated direct whole-score gain is 0.2–1.0%
relative to V44. No H200 score or memory result is measured for V50 yet.

Local verification preserved the exact V44 and V50 eager Python bodies. Their
live rows and combined BF16 outputs were bitwise equal in 40 CPU Torch cases
using modeled NVFP4 dequantization, SiLU and Triton execution. Cases covered
all-inactive, fully active, uneven, single-live and repeated slots, both norm
orders and contiguous and strided layouts. A separate fallback probe selected
Marlin when no fused norm weight was available. The Triton AST kept its live
arithmetic and store, with the equivalent slot-based score address. An
independent review found no blocking issue in the graph bridge or slot mapping.
The official CLI exported the 27,242-byte five-file patch, all four Python
sources compiled, and an exact pinned-base patch apply and whitespace check
passed. Added lines were at most 110 bytes.

Prioritize a free official 16-user check against the V44 launch settings. Inspect
peak GPU memory, throughput, TTFT, first/later-turn correctness, startup and
capture before considering larger chunks or another performance change.
