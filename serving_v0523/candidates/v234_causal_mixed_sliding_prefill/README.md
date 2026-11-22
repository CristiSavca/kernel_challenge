# V234 preserves direct sliding-window attention in joined prompt and verification batches

V234 runs V223's packed FP8 attention on the 46 sliding layers of an eligible V232 joined target forward. V232 otherwise sends the same mixed batch to native paged FA3. The mixed worker marks the live forward only after all ranks agree that every running request supplied a linear, top-1 causal proposal with a valid reservation. The marker reaches the static batch for that replay. Warmup, noncausal masks, full attention, unsupported cache layouts and declines continue through the original code. [Source changes](delta.diff) · [Proposal and replay ownership](ownership.json)

The packer consumes each request's retained sliding prefix and current K/V, including all eight reserved verification slots. It reads prefixes from the authoritative translated SWA page table. The original writer still receives the translated slots allocated for those same tokens. Verification subsequently accepts or rejects the ordinary prefix. The stage holds no request data across forwards. In the mixed extension a layer must use the actual sliding pool with a positive window. All V223 guards for the native dtype, hardware backend, tensor layout, live sizes, capture mode, and 32 MiB allocation remain active. [Physical-cache and attention replay](attention.json) · [Native FA3 and storage audit](../v223_batched_direct_prompt/independent_replay.md)

| Concurrent users | Oversized joined opportunities in archived schedule | Maximum scratch including current writer inputs |
| ---: | ---: | ---: |
| 1 | 0 | — |
| 4 | 25 | 10.26 MiB |
| 16 | 72 | 17.06 MiB |
| 32 | 145 | 26.11 MiB |

This upper bound gives every request its full 2,047-key prefix and 256 guard rows, including cold requests that may have shorter prefixes. The V41 archive contains ordinary arrivals on an eight-H200 node. It did not execute V234. Actual mixed eligibility, H200 timing, rank peak memory and official throughput are still unmeasured. [Trace calculation and limits](work.json)

Thirteen parent/candidate comparisons executed the native model attention, Radix callback, real FA3 metadata and cache writer on CPU-backed tensors. Positive cases included one 16,384-token prompt and 31 width-eight tails, two prompt requests, heterogeneous prefix lengths, fragmented slots, scaled cached K/V and poison padding. All 81,123,840 inspected physical cache bytes and 17,664 sampled FP64 attention scalars matched. The fixture models FA3's launch boundary and the packer's storage operation. The unchanged Triton packer's existing interpreter and Hopper assembly evidence are available separately. Thirty-two worker rounds reached eight agreed target calls and exercised rank disagreement, stale proposals, absent graph keys and insufficient capacity. [Attention oracle](attention.py) · [Target-call ownership](ownership.py) · [Triton interpreter](../v223_batched_direct_prompt/interpreter.json) · [SM90a compile](../v223_batched_direct_prompt/sm90.json)

The complete [official export](patch.diff) is 387,269 bytes. Its SHA-256 is `0c8cf27cf07fd2545e12d6b2ff8369948a0dee06d4ee59267f7f6b7064855201` on [V232](../v232_mixed_prompt_overflow/README.md). The four changed source files add 653 bytes to the parent's canonical full patch, with the same launch flags. Full and relative patches apply and reverse on the public pin. [Release validation](validation.json)
