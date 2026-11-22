# Independent V229 standby replay

The staged November 20 worker preserves all 131 tested ordinary V215 poll outcomes, status-query sequences, evidence fields and attempt counts. Fourteen cases that selected V215 now select V229. Forty-nine select V198, 67 hold without a claim and one finds an existing claim. The counts describe synthetic scenarios. Some simulated submissions return an error or an ambiguous result.

The fixture loads the archived V215 and staged V229 polling functions directly. It supplies the same inputs to each, uses actual file locks and exclusive receipts in temporary directories, and replaces every CLI call with an in-process fake. It reuses archived V68 results and the bound V204 submission identity. Its positive completed V204 result is synthetic. It also checks the exact patch, sorted flags, 16-user command and receipt that each selected candidate would use.

- A V229 offer requires the bound V204 check at 16 users to pass and improve both raw and stock-normalized throughput over V68 by more than the measured 1.63% V64 repeat spread. Correctness, later turns, TTFT, memory, reference identity, configuration and local lineage checks still apply. A stable nonqualifying V204 selects V198 when the rest of the allocation is eligible. A changing selection between observations holds.
- The poll requires the earlier V129 and V141 allocations, an eligible V68 reference and a pending V120b control. It rechecks the exact third-slot quota and V120b immediately before issuing the command. The replay exercises the 04:00–08:00 UTC window, quota reset, stale responses, terminal status, transport failures, unknown or malformed responses, equivalent or partial prior receipts and an ambiguous submission acknowledgment.
- Two concurrent acquisition orders let either V215 or V229 enter the shared allocator first. Each produced exactly one simulated command and one receipt. A retry following a successful or uncertain mock dispatch found the existing receipt.

## Clock correction

An additional nine cases replay the two workers with independently controlled wall and monotonic time. V215 issued a mock command when the final control observation moved from 04:10 back to 04:05, as well as when UTC remained fixed through an expired freshness budget. The revised V229 denies even a one-microsecond sampled clock rollback. It denies at 120 elapsed seconds from preflight or 30 elapsed seconds from the final observations. The controls at 119.999 and 29.999 seconds authorize one mock command. An anomaly after exclusive receipt creation suppresses delivery and preserves the receipt for manual reconciliation. An earlier anomaly leaves the slot unclaimed.

## Reproduction

From the challenge repository, run the following command. The default uses the historical staging directory even if a different worker is subsequently deployed.

```sh
python3 -B serving_v0523/candidates/v229_sharded_decode_serving/independent_standby.py
```

| Artifact | SHA-256 |
| --- | --- |
| Archived V215 worker | `ee80f0bd9e785b326cd0eaad948c6626a26dcbc5443e4c71f03d875ba801e8f5` |
| Staged V229 worker | `72e4d0b86709452534a65560669a111b8b78bcf2d318c28625d44be7039162e0` |
| Staged V229 selector | `344ed160ca81eeb248af3ba8d2eac25e72f0b64b5c7951db03c99aba12898595` |
| V229 challenge patch | `d81634df5c43230867d314540c7ae9ebe5ff8ba40c8fdc59482104ee3d75adcd` |

The [machine receipt](independent_standby.json) records the complete case list and both workers' clock outcomes. This offline replay measures worker behavior. H200 performance, live quota and official submission acceptance require separate evidence.
