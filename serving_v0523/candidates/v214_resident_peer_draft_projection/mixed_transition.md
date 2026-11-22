# Multi-round mixed scheduler comparison

V214 and V212 completed 88 paired scheduler comparisons using real four- and eight-process Gloo. Each round exercised native mixed admission, the prepared target call, the eager graph tail or eager model, both sampling roles, physical writes to five draft layers, publication and the following filtered tick. The helper and its rank-specific weight persisted across requests. Prompt IDs, tail IDs, accept lengths, next lengths and all physical cache words matched V212. [Reproducer](mixed_transition.py) · [Machine results](mixed_transition.json)

On a normal warm TP4 round the actual transcript contained zero weight sends per rank, down from one. At TP8 it contained two, down from three. The omitted destination was the rank halfway around the TP group in every case. Across the two worlds, six shorter rounds passed only after warming the weight bank. The tests also mutated one rank’s local bank in place, replaced it with a new tensor, and retired a helper on one rank. The mixed scheduler fell back before any asymmetric exchange, and subsequently repopulated the bank. Weak references confirmed release of obsolete peer tensors after mutation and replacement. The full target logits for prompt sampling and the greedy tail score/ID gathers remained correct.

The Full graph cases used two deliberately padded buckets within the native 2× graph-admission limit. The captured layer supplied a contiguous rank-major layout, with NaNs in every padding row. Both cold and warm V214 calls selected paired K/V; V212 used its normalized-feature gather. The oracle checked every actual point-to-point message against expected peer, shape and order. All paired GEMM inputs and sent tensors were finite. It compared every physical slot, including rejected tails and untouched slots, against the original result.

| TP size | Live / captured rows | Live rows per rank |
|---:|---:|---|
| 4 | 529 / 1,024 | 256, 256, 17, 0 |
| 8 | 1,041 / 2,048 | 256, 256, 256, 256, 17, 0, 0, 0 |

The tested Python CUDA boundaries used two alternating stream probes. Each world observed 13 producer events, 19 warm waits and 13 cross-stream waits. Each successful use recorded the local weight on the consuming stream, and warm uses recorded the retained peer tensor as well. Gloo executed the transport with CPU tensors. Actual H200 event timing, asynchronous allocator behavior and throughput remain unmeasured. The JSON byte counts cover paired point-to-point sends. V212 fallback performs a separate normalized-feature collective. [Weight-load invalidation and stream lifecycle](independent_lifetime.json)
