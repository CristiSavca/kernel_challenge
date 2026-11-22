# Mixed scheduler and five-layer physical cache replay

The frozen V212 source completed 78 paired comparisons with V211 across 156 scheduler rounds at TP4 and TP8. The fixture exercised scheduler admission, preparation and the actual `try_forward_proposed_mixed` call, eager or replayed target body, the real prompt sampler and tail verifier, five layers of draft-cache writes, publication and the following filtered tick. Every comparison matched the independent physical cache expectation, sampled IDs, acceptance and next lengths. The source hashes for all three changed files are recorded. [Fixture](mixed_scheduler.py) · [Results](mixed_scheduler.json)

| Test outcome | TP4 | TP8 |
|---|---:|---:|
| Completed V211/V212 comparisons | 39 | 39 |
| V212 mixed forwards returning projected K/V | 11 | 11 |
| Physical BF16 cache words checked per rank across both versions | 2,364,480 | 4,576,320 |
| Rejected tail slots checked per rank across both versions | 546 | 546 |
| Third-tail-layer failures recovered across both versions | 6 | 6 |
| V212 falls back after one rank’s GEMM workspace failure | 3 | 3 |

The oracle derives K/V from the full normalized target taps and rank-specific, five-layer BF16 weight bank. It initializes every cache word to a sentinel, computes all physical prompt positions and accepted running prefixes, and compares the complete resulting pool. The tails exercised every accepted length from one through eight. Both fused and per-layer recovery paths kept rejected suffixes intact. The normal path gathered complete prompt scores and only tail score/ID pairs. A width-sensitive head hook received the full vocabulary and disabled the optimized representation.

A TP8 Full example combined 1,056 target rows for two prompts and two running requests. V211 returned 64 normalized channels and projected them in each writer. V212 returned 20 already projected channels, using 56,256 bytes of paired sends per rank in the small fixture. Both committed one and four tail tokens and preserved 11 rejected slots. The JSON’s `wire_bytes` counts only the new point-to-point sends. The parent uses normalized-feature all-gather traffic outside this counter.

The suite exercised disabled context, a retired helper or unavailable communicator on one rank, a rank-local normalization hook, non-unit value scaling, short rows, an unfavorable byte budget, multi-rank recovery after a GEMM workspace error, and bucket padding greater than one TP group. Full replay selected the normalized gather when excessive padding made its captured shard inconsistent with the live extent. Breakable used the actual serving bucket and trimmed the output. Real Gloo supplied point-to-point exchanges and collective agreements. CUDA kernels, target layers and graph bodies were simulated. H200 equality and latency remain open.
