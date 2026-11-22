# V220 prompt-bucket credit feasibility

**Keep V215’s final-bucket gate.** Starting each prompt with zero credit and spending only savings from earlier successful mixed forwards produced **zero additional full-chunk joins**. This held for all four continuity-identifiable chains in the archived eight-H200 V41 chronology and for 1,256 first-turn request/cohort combinations. The latter simulation fixes a running cohort at every later continuation and grants success to every eligible join. [Reproduction](replay.py) · [Results](replay.json) · [V194 baseline and trace provenance](../v194_mixed_decode_chunk_budget/README.md)

For a full 16,384-row chunk, a reservation displaces eight prompt rows per running request. The debit equals the increase in the **eventual last prompt bucket**, recomputed after every earlier reservation. A successful mixed forward earns at most its separate prompt bucket plus its separate verify bucket minus the joint prompt bucket. The debit must be covered before the allocator shortens this chunk. This is already an optimistic accounting unit because one padded prompt row and one separate verification row can take different amounts of time.

| Archived chain | Running requests | Identifiable continuations | V215 joins | Prior-credit joins | Joins if the current forward is credited in advance |
|---|---:|---:|---:|---:|---:|
| 4-user first repetition | 3 | 2 | 0 | 0 | 0 |
| 4-user second repetition | 3 | 4 | 4 | 4 | 4 |
| 16 users | 15 | 5 | 0 | 0 | 2 |
| 32 users | 31 | 4 | 1 | 1 | 4 |

At 16 users the first debit is 64 rows. At 32 users the first join earns 256 rows, then moving the next chunk crosses a 512-row boundary. On each chain the previous partial prompt calls in the same repetition provided zero observable opportunities to seed an account. A hypothetical pre-existing 512-row balance would enable five, versus zero, at 16 users and four, versus one, at 32 users. Other full chunks in the log have ambiguous request ownership and remaining lengths. V41 did not run this mixed policy, so the archive provides no actual cross-request credit or elapsed-time comparison. [Archived scheduler evidence](../../../experiments/iterations/20261118-serving-mixed-graph-capacity.py)

V215 reserves the shorter prompt before its final cross-rank graph admission. A late veto can execute separate verification and prompt forwards after the reservation. Both paths return a `MixedDFlashResult`; the constructor accepts `shared_target` as a consistency check and does not retain it as a completion record. A ledger would need the agreed three graph sizes, explicit successful joint execution and normal result completion before awarding any credit. It would still have to honor ordinary intermediate prompt handling, which decrements the outstanding middle-chunk count and withholds streaming until the prompt completes. The lookahead and cross-request scenarios above do not justify changing that lifecycle. V220 ends at the replay stage. [V215 source export](../v215_decode_reduction_paired_context/patch.diff)
