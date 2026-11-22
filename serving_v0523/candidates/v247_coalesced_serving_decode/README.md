# V247 combines prompt routing with direct FP8 prompt storage

V247 integrates two independent changes on V241. Eligible long MoE prompts exchange exact router IDs, scores and the peer-packing vote in one collective. On a mixed verification pass with a single long prompt, the model writes the new FP8 K/V rows directly into the buffer that dense FA3 consumes. The physical writer still publishes the entire batch. [Full patch](patch.diff) · [Delta from V241](delta.diff)

| Work per eligible call | V241 | V247 |
| --- | ---: | ---: |
| Routing collectives per expert layer | 3 | 1 |
| Routing collectives over 62 eligible layers | 186 | 62 |
| Extra logical FP8 prompt copy for 16,384 new rows, per full-attention layer and GPU | 8 MiB | 0 |

The archived eight-H200 V41 wave had 75 and 150 long prompt passes at 16 and 32 users. If every pass enters the peer path, the routing change removes 9,300 and 18,600 collectives. The routing model also adds 2.25 MiB of local packing traffic per rank and layer at 16,384 tokens. Achieving a 5% whole-wave throughput gain from routing alone would require 844 and 779 µs of net savings per removed collective. The independent mixed-attention replay found 58 and 115 eligible single-prompt joins at those loads, removing 7.25 and 14.375 GiB of logical copying per GPU across the wave. These studies estimate work, and neither measures V247 throughput. [Routing model](../v242_coalesced_prompt_routing/work.json) · [KV model and conservative memory bounds](../v246_direct_mixed_prompt_storage/work.json)

On the integrated tree, Gloo TP2/4/8 routing matched 4,353,728 BF16 components. Twenty paired executions through the production model and host graph checked 83,356,928 cache bytes. Both a missing-output ABI retry and a provider returning fresh storage completed correctly. A separate test covered 21 allocation thresholds and reproduced stale physical V bytes when it deliberately removed a return-stream wait. [Numerical and provider tests](independent_host.md) · [Lifetime and budget](independent_lifetime.json)

The official exporter produced 401,616 bytes out of 409,600 allowed. Both donor orders produced identical code and reversed on the pinned SGLang commit. The flag parser accepted every ordering of the eight flags, and 17 server configurations resolved. V64 holds the official four-load score of 1.4119×. At 03:21 UTC all four earlier private jobs were queued and all three daily private checks were used. V247 was built locally. Its H200 token agreement, TTFT, peak memory and throughput remain open. [Source and public contract](validation.json) · [Flags](flags.json) · [Server configuration](server_resolution.json) · [Judge observation](judge_observation.json)
