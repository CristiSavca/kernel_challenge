# V214 reuses one peer’s draft projection weights

After a successful V212-style exchange, each TP rank retains the packed draft K/V weight belonging to the opposite rank. A warm forward projects fresh token features and skips exactly that weight transfer. At challenge TP8 geometry, the extra persistent allocation is **20 MiB per rank**, or **160 MiB across eight ranks**. Version, identity and rank agreement protect reuse. Draft-weight updates invalidate the packed banks before writing, including a load that subsequently fails. The next append rebuilds the local bank. V212’s scaled-V correction and accepted-prefix cache writes carry forward. [Source delta](delta.diff) · [Weight and stream lifecycle checks](independent_lifetime.json) · [V212 scaled-V counterexample](../v212_mixed_paired_draft_kv/worker_prefix.json)

V214 also handles Full graph buckets with uneven and entirely empty token owners. The code preserves the captured rank-major row pitch and sends and multiplies each owner’s live prefix. The TP8 scheduler test placed 1,041 live rows in a 2,048-row bucket, leaving four owners with 256 rows each, one with 17, and three with zero. Exact peer-message shapes, NaN-poisoned padding, sampling and every physical cache slot were checked. [Scheduler transition replay](mixed_transition.md) · [Transport tests](transport.json)

## Reconstructed serving opportunities

The model replays ordinary and possible mixed targets in chronological order across seven V41 H200 repetitions, with a cold weight bank at each repetition’s start. A qualifying ordinary prompt can prepare the bank for later mixed requests. The table uses the captured row pitch and shows cumulative logical send volume for this endpoint, averaged over TP8 ranks. [Geometry and assumptions](geometry.json)

| Concurrent users | Eligible ordinary prompts, V212 → V214 | Eligible mixed targets, V212 → V214 | Endpoint GiB per rank, V212 → V214 |
|---:|---:|---:|---:|
| 1 | 15 → 36 | 0 → 0 | 2.44 → 1.87 |
| 4 | 30 → 35 | 4 → 42 | 5.13 → 3.94 |
| 16 | 75 → 110 | 0 → 58 | 10.86 → 8.44 |
| 32 | 145 → 265 | 4 → 45 | 22.71 → 17.33 |

The frozen V214 and V212 sources matched in 88 paired scheduler comparisons using real four- and eight-rank Gloo. Warm TP4 omitted its sole peer-weight send. Warm TP8 sent two of its three peer banks. A rank-local mutation, replacement or helper retirement triggered coordinated fallback. CUDA event and allocator boundaries were emulated. H200 throughput, TTFT, actual peak memory and challenge score remain unmeasured. The 314,264-byte official patch retains V212’s launch flags and strictly applies and reverses. [Comparison results](mixed_transition.json) · [Official patch](patch.diff) · [Validation](validation.json)
