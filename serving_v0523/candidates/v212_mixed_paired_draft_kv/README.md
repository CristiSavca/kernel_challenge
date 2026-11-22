# V212 projects draft K/V during mixed target forwards

V212 connects V211’s live mixed scheduler to the paired K/V projection already available for ordinary prompts. On a compatible TP4 or TP8 group, sequence owners project normalized target features with the released draft weights and exchange the narrower K/V rows. The prompt writer commits every prompt row. The running-request writer commits only each accepted prefix, leaving rejected slots available for the next step. The ordinary prompt sampler and greedy sharded-vocabulary verifier keep their existing inputs. [Source delta](delta.diff) · [End-to-end scheduler replay](mixed_scheduler.md)

V212 also fixes a value-scaling error. When a layer specifies a non-unit model V multiplier, the worker selects the per-layer projection that applies it. With a multiplier of two on one layer, V211 wrote 330 incorrect BF16 values to the physical cache. V212 matched the sequential reference on all 14,880 K/V words. [Scaled-V counterexample](worker_prefix.json)

## Where the exchange applies

The worker requires compatible BF16 projection, a fused draft helper with unit model V scales, working transport and agreement across ranks. Its byte estimate must beat gathering the normalized features by at least one eighth. V212 requires a tight contiguous row layout with at most TP−1 trailing padding rows. Extra rows in a Full graph bucket can send the mixed call through the normalized-feature gather. Each qualifying forward exchanges its peer weights anew. [Geometry and assumptions](geometry.json)

| Concurrent users | Possible mixed forwards | Eligible exchanges with tightly packed rows | Eligible exchanges with captured row pitch |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 4 | 70 | 5 | 4 |
| 16 | 99 | 0 | 0 |
| 32 | 59 | 7 | 4 |

These are opportunities reconstructed from seven archived V41 H200 repetitions. The V41 server used a different schedule. All 99 possible 16-user mixed targets still favored the normalized-feature gather under this cost gate. [Chronology replay](geometry.py)

The actual scheduler matched V211 across 78 paired CPU/Gloo comparisons, including acceptance lengths one through eight and recovery after partial physical writes. The frozen 305,032-byte patch strictly applies and reverses on the official pin with V211’s six launch flags. V212 has no H200 timing or challenge score. [Scheduler results](mixed_scheduler.json) · [Official patch](patch.diff) · [Validation](validation.json)
