# V239 splits full attention by request role

A certified mixed prompt and verification batch can retain long-prompt dense FA3 on its 16 full-attention layers. On the eligible Breakable callback, the paged call handles all running requests using their full physical histories. A second call packs only the prompt cohort's visible K/V and runs native dense FA3. Both calls write their disjoint BF16 query rows into the original graph output. The implementation preserves per-request causal alignment, both cache scales, softcap and the original physical writer. The 46 sliding-window layers retain V235's packed path. [Official patch](patch.diff) · [Code delta](delta.diff)

The prompt count originates in the target's output-row planner and travels through replay with the same causal certificate. Missing certificates, masked proposals, unsupported caches, missing preallocated output and over-budget cohorts retain parent behavior. The tail offsets are rebased once per live metadata object. An older FA3 provider that returns its own tensor has its result copied into the assigned output slice. [Model and graph replay](replay.json) · [Native Python provider comparison](independent_fa3.json)

| Archived cold wave | Full chunks with running requests | Prompt packs proven within 32 MiB | Maximum bounded pack plus converted writer inputs |
| --- | ---: | ---: | ---: |
| 4 users, two repetitions | 25 | 21 | 27.33 MiB |
| 16 users, one repetition | 72 | 58 | 30.55 MiB |
| 32 users, one repetition | 145 | 115 | 30.15 MiB |

The bound uses the longest actual first-turn requests from the same wave. At 16 and 32 users, the eligible counts represent 950,272 and 1,884,160 long-prompt rows respectively. Packing every decoder history as well would exceed 32 MiB in at least 51 and 109 of those batches. Additional multi-prompt batches may fit. One user has no competing verification request. Each split adds an attention launch per full layer and gathers prompt keys from their native slots. The archived V41 GPU did not execute this candidate. [Trace, assumptions and bounds](work.json)

Ten paired CPU model/graph comparisons against V235 matched 133,264,128 visible FP8 K/V bytes and 23,040 sampled FP64 attention values. The independent provider comparison covered 144 Python FA3 calls, including 36 legacy retries. The actual Triton packer matched 87,057,920 bytes in the interpreter, including its newly exercised one-prompt grid. The 409,513-byte export passes the published source, flag, line-length and patch limits. H200 peak memory, token agreement, TTFT and serving speed require a physical check. [Provider ABI](independent_fa3.json) · [Triton cases](pack_interpreter.json) · [Contract and hashes](validation.json)

