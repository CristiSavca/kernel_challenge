# What the archived H200 results establish

The archive contains 24 official serving job JSON files. Twenty-two startup logs say `cut: first request`. Each shows one 151-token prefill and one one-token prefill, each with one new sequence and zero cached tokens. Another log ends when its server stops and contains no prefill batch. The timed-out job has no startup log. **Zero of the 44 observed batches pass V171’s 1,024-token threshold at TP8.** This observation covers startup only. The public scored rungs report workload concurrency and aggregate TTFT, throughput and cache-hit figures. They do not expose query rows or sequence counts for individual timed forwards. [Archived job audit](traffic.json) · [Scored V64](../../runs/abd217baf0ed49fd.json)

V64 reports a varied workload with a 74,160-token first turn and 753 new tokens per subsequent turn. Its startup configuration sets both prompt chunk limits to 16,384. A single uncached request of exactly 74,160 rows, processed alone at full chunks, would produce four 16,384-row forwards plus one 8,624-row forward. All five clear the row thresholds. A single 753-row forward falls below the switch. Two such extensions scheduled together would produce 1,506 rows and pass both cardinality checks. Admission timing, cache state, graph padding, module compatibility and rank consensus affect actual eligibility. The archive cannot supply an activation percentage. [V64 startup](../../runs/abd217baf0ed49fd.log) · [V171 production delta](delta.diff)

| Conditional TP8 case | Prior two all-gather outputs | V171 two all-gather outputs | V171 endpoint output |
| --- | ---: | ---: | ---: |
| 16,384 rows, one sequence | 1,536 MiB | 256.125 MiB | 0.125 MiB |
| 16,384 rows, 32 sequences | 1,536 MiB | 260 MiB | 4 MiB |
| 1,506 rows, two sequences | 141.75 MiB | 23.875 MiB | 0.25 MiB |

These quantities are the logical all-gather destination sizes on one rank, excluding allocator reuse, local views, collectives’ link protocol and FC work. The original gathers carry five BF16 tap vectors and one final hidden vector per row. V171 carries one projected context vector per aligned row and one final hidden vector per endpoint per rank. Each vector has 8,192 BF16 elements. The ratio cannot be assigned to TTFT or peak memory from these receipts. [Reproduction script](traffic.py)
