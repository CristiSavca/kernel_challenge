# V218 reads contiguous sliding-window KV tiles directly during verification

V218 extends V215’s Hopper attention transport to the target’s 46 sliding-window layers. The existing graph replay kernel already translates the physical slots for the sliding KV pool. It now checks every address in each complete 64-row tile in the active suffix. A consecutive tile can feed two tensor-map loads for a KV head. Fragmented tiles and both partially visible boundaries use the indexed path. The attention kernel keeps the parent’s query positions, masks, descales, K tile order, accumulation and output merge. [Source delta](delta.diff) · [Full patch](patch.diff)

The certificate table is allocated with the ordinary target graph. At window 2,047, verify width eight and batch capacity 32 it occupies **4,352 bytes** per GPU. Each replay rewrites the live entries after resolving the current request-to-slot mapping. A shorter actual query can further trim the first eligible tile. This prevents a bulk load from touching expired sliding-cache rows. [Capture and replay checks](dispatch.json) · [Independent allocator analysis](independent_geometry.md)

## What the verification established

- The production graph methods launched the metadata builder on eight replays. All 368 sliding-layer calls received the sliding table and its own certificates. All 128 full-layer calls kept their original table. Thirty-two fallback cases exercised the layer window, query width, alternate page table and ragged verifier. [Runtime entry tests](dispatch.py) · [Results](dispatch.json)
- Across 24 CPU Triton interpreter replays, the actual producer translated 52,343 page IDs. The verifier executed 1,396 emulated tensor-map loads at exactly the expected physical addresses. Its 430,560 FP32 partial and log-sum-exp words and 227,328 BF16 output words matched the V215 indexed kernel bit for bit. A separate dense reference checked 3,456 output values. [Interpreter and oracle](interpreter.py) · [Results](interpreter.json)
- All 22 offline Hopper compiler builds passed, including the dual full/sliding producer on static and virtual mappings. Every direct partition emitted four static 2-D TMA load sites and used 33,048 bytes of shared memory. The direct specializations took 218–236 registers and all 22 builds had zero spills. [Compiled instructions](sm90.json)

## Performance evidence

On the archived request lengths, a fresh page allocator kept most of the visible KV bytes in eligible tiles. Fragmentation substantially changes coverage. In the 32-user later-turn replay, 92.33% of KV bytes qualified with a fresh allocator, 69.53% with shuffled 256-row runs and 1.96% with shuffled 64-row runs. The trace records lengths and request order. Physical addresses in this comparison come from allocator replay. [Geometry and assumptions](independent_geometry.md)

The official export is 327,697 bytes across 47 paths on the published pin. It retains V215’s six launch flags and passed the published static restrictions. Throughput, exact served tokens, peak memory and first-token latency remain unmeasured on H200. V218 has no judge submission. [Validation](validation.json) · [Serving contract](https://openkernelchallenge.dev/api/v1/serving)
