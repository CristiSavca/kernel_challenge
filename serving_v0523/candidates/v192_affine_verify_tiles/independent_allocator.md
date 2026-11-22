# Coverage of the contiguous history path under allocator churn

This replay feeds 767 archived request lengths through the pinned page-size-one allocator and replays one representative verification block per request. Each same-turn cohort is scheduled together in the model. A 128-row tile qualifies only when every row is contiguous and in bounds. V192 uses bulk reads only when every tile intersecting the request’s partition segment qualifies. The denominator below includes all history tiles. [Replay program](independent_allocator.py) · [Input and result hashes](independent_allocator.json)

| Initial physical free list | 1 user | 4 users | 16 users | 32 users |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 94.92% | 82.65% | 36.67% | 3.81% |
| Shuffled 4,096-row runs | 94.57% | 54.05% | 13.01% | 0.02% |
| Shuffled 256-row runs | 93.71% | 33.61% | 12.88% | 0.00% |
| Shuffled 32-row runs | 0.00% | 0.00% | 0.00% | 0.00% |

The allocator and the prompt, cache-hit and output lengths come from serving. The interleaving, four committed tokens per append round, retirement policy and starting free lists are modeled. Live physical address histories and hardware timings are unavailable. Fragmentation and partition size can therefore materially change how often bulk reads execute. [Full counts by turn](independent_allocator.json)
