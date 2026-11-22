# V236 grouped verification merge

V236 shares one request-length scan across adjacent verification heads. It changes the second kernel of the existing Hopper target verification attention, after the full or sliding key/value scan has produced its FP32 partition outputs. The first kernel already shares each key/value tile across all eight query positions and eight grouped heads per local KV head. [Delta](delta.diff)

| Concurrent requests | Packed rows per merge CTA | V235 → V236 merge CTAs per layer | Fewer merge CTAs per 62-layer step |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 128 → 128 | 0 |
| 4 | 2 | 512 → 256 | 15,872 |
| 16 | 4 | 2,048 → 512 | 95,232 |
| 32 | 4 | 4,096 → 1,024 | 190,464 |

Each grouped CTA computes the request's partition boundaries once and then processes its live rows. On a full /32 verify grid this removes 18,284,544 logical int32 reads of lengths and query offsets over 62 layers, alongside the repeated scans. These small arrays can reside in cache. The F32 partials, keys and values remain the dominant data arrays. Each CTA works longer, with fewer independent CTAs. A single request retains its 128-CTA launch. [Launch and coverage replay](geometry.json)

The 16-split FP32 maximum, weighted numerator, denominator and per-request scale use the original calculation and reduction ordering. Offline Triton 3.8 generated 16 parent/candidate SM90 objects across four batches and two windows. Every pair matched all three resolved reduction layouts and its per-loop FP32 add, FMA, exp2 and butterfly-shuffle counts. All candidates had zero spills and 1,024 bytes of shared memory. An earlier vectorized prototype changed reduction layout and differed by one BF16 word on a ragged case. The final source uses independent scalar-row reductions. [SM90 compiler receipt](sm90.json)

The Triton interpreter produced identical bits for 1,521,856 BF16 words across 40 comparisons, including ragged rows, empty keys, zero live queries, optional scales and untouched output guards. CPU launch replay covered 14,976 admitted geometries and 54,464 request-row ownership proofs across batches 1–64. A separate source-level graph/backend replay exercised 128 full and 368 sliding layer calls, alongside 32 fallback checks. All used host substitutes for GPU execution. [Differential](interpreter.json) · [Launch replay](geometry.json) · [Backend dispatch](backend_replay.json)

An archived H200 standalone attention implementation took 100.0 µs per full and 10.8 µs per sliding layer at /16, and 185.3 µs and 13.9 µs at /32, respectively. Sixteen full and 46 sliding layers give about 2.10 ms and 3.60 ms per step in that implementation. It has a different fused kernel, so these times only give scale for attention as a whole. V236 has no physical H200 timing, accuracy result or serving speedup. [Historical profile](../../../experiments/candidates/muse_attention_decode/profiles/87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c.c21d391dbf604bb5.json)

The official export on V235 is 407,719 bytes under the 409,600-byte cap. The full patch and delta reconstructed and reversed on the public SGLang pin. The only changed production file relative to V235 is the verification attention implementation. The launch flags match V235. Full SHA-256 is `c38b23ad967bd3ba862f2d4a7fee0885f62649dd1f27758fc64b97524e9791a8`. [Patch](patch.diff) · [Public-rule and source validation](validation.json)

For local reproduction, run `verify.py <stage>` and `geometry.py <stage>` on the exact published source. Run the V218 `dispatch.py <stage>` for graph binding. Run `check_gpu.py compile <v235-source.py> <v236-source.py>` under Triton 3.8 with its NVIDIA backend, NumPy and the serving runs directory on PYTHONPATH. Select `interpret` with `TRITON_INTERPRET=1` to compare the kernel bodies.
