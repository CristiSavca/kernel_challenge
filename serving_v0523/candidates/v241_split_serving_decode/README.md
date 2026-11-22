# V241 splits full attention between prompts and running decoders

V241 combines V236's verification merge, V237's greedy vocabulary reduction, V239's split prompt attention and V240's bounded verification tail on top of V235. An eligible mixed batch processes its long prompts with dense FP8 FA3 and its running requests with paged split-KV attention. Each role writes its own rows into the same captured output buffer. The ordinary decode path also shares partition planning across adjacent heads and applies the model scale while reducing each vocabulary shard. [Complete patch](patch.diff) · [Four component deltas and rule validation](validation.json)

## Work on the archived hardware shapes

| Concurrent requests | Full-chunk arrivals with running requests | Prompt packs proven within 32 MiB | Merge CTAs per ordinary verification layer, V235 → V241 | Local greedy score traffic removed per rank per eligible step |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 128 → 128 | 1.54 MiB |
| 4 | 25 | 21 | 512 → 256 | 6.17 MiB |
| 16 | 72 | 58 | 2,048 → 512 | 24.66 MiB |
| 32 | 145 | 115 | 4,096 → 1,024 | 49.33 MiB |

The arrivals are from the initial waves of the archived eight-H200 V41 run. The 16- and 32-user loads each have one repetition. Their bounded single-prompt packs account for 950,272 and 1,884,160 prompt rows. Splitting avoids copying every decoder's long prefix into those packs. The pinned FA3 C++ selector enables Hopper TMA KV loads for the split dense prompt. It selected page-one loads for the earlier mixed call. Both used 192 × 160 tiles, packed grouped-query attention and one split across the modeled long-prompt cases. The split adds a second attention call. Its cost and the net saving need a hardware measurement. [Physical trace and memory bounds](../v239_split_mixed_full_attention/work.json) · [Pinned FA3 host selector replay](fa3_dispatch.json)

At 32 users, sharing the partition scan reduces merge CTAs across 62 ordinary verification layers from 253,952 to 63,488. The per-row FP32 reductions keep the same operation order, and the first attention kernel retains every visible key/value. Eight paired SM90 builds resolved the same reduction layouts and used zero spills. A Triton interpreter compared 1,521,856 BF16 words. The vocabulary change removes a separate FP32 scaling pass on the eligible greedy graph and carries its scale through graph trimming. Adjusted sampling reconstructs the scaled shard before the native consumer. [Verification merge](../v236_tiled_verify_merge/README.md) · [Score reduction and fallback](../v237_fused_verify_scale/README.md)

## Correctness, resource limits and release

The target verifies a rank-agreed linear cohort and the original physical slots. A split-KV tail requires an eligible SM90 attention backend, uniform real verification width, supported geometry, zero attention softcap and sufficient workspace. The partial and LSE buffers are charged alongside the live converted writer inputs before the later prompt pack. An unsupported tail takes native paged FA3. Native providers that return a separate output tensor have it copied into the proper graph slice. [Composed dispatch at 3, 15 and 31 tail requests](mixed_merge.json) · [Memory admission and legacy providers](../v240_mixed_paged_verify_tail/limits.json)

Seventeen paired CPU replays ran the model and graph path and exercised both specialized kernel bodies on actual mixed operands. Across 409,600 tail scalars, maximum relative L2 error against FP64 causal attention was 0.002244. A separate comparison against V235 matched 133,264,128 visible FP8 K/V bytes and 23,040 sampled FP64 attention values for the native split path and its fallbacks. The full graph and sampling suite passed 70 paired replays. These checks leave GPU reduction agreement, end-to-end tokens, peak memory, TTFT and throughput for the H200 judge. [Integrated numerical replay](mixed_merge_numerical.json) · [Full-prompt and native fallback comparison](split_full_replay.json) · [Graph and sampling](independent_graph.json)

The official exporter produced 397,454 bytes across 58 permitted files using Git's standard two context lines. A three-context diff reconstructed the identical source. All 12 dependency-valid component orders reconstructed and reversed on the published pin. All 40,320 orderings of the eight launch flags parsed and 17 runtime configurations resolved. The full patch SHA-256 is `c1b938cd93f2cfdbaba9b0e6dce7e5d16f95410d811db9a8e8570693739f0acf`. [Public-contract validation](validation.json) · [Flag and positional defaults](flags.json) · [Runtime configuration](server_resolution.json) · [Global name checks](global_names.json)

V64's measured four-load score is 1.4119×. V241 has no hardware result. The 02:57 UTC judge observation showed four queued checks, three private checks used for the day and one scored attempt available. V229 remains the conditional November 20 private choice on a qualifying V204/16 result. V198 remains the fallback. The source changes were prepared locally and consumed no submissions. [Official V64 result](../../runs/abd217baf0ed49fd.json) · [Judge observation](judge_observation.json) · [Existing selection and thresholds](historical_selection.json)

