# One-user verification on eight H200s

The matched V41 one-user verification round took **15.841 ms** and V44 took **15.895 ms**. To double V41's steady-state decode rate at the same acceptance, a change must remove 7.920 ms from each round. In nine independent official H200 attention jobs with eight verification positions and roughly 80K context, the entire 62-layer stock attention operation cost 1.605–1.643 ms when weighted by the released model's 46 sliding and 16 full layers. Deleting even the largest of these proxy times would imply 1.116× one-user streaming. The fastest attention submission took 0.529 ms, and the calculation includes projection input processing, cache writes and output gating alongside the attention itself. A partition-planning change reaches a fraction of that time. [Reproduction and per-job hashes](check.json) · [Standalone attention contract](../../../muse_attention_decode/TASK.md)

The other standalone verify cases use 16 and 32 requests. At those loads the largest stock attention proxies are 4.640 and 7.836 ms. Against V64’s illustrative periods of 79.840 and 96.998 ms, deleting those proxies would imply 1.062× and 1.088× streaming. The standalone shape suite has no four-user verification case. [Per-load measurements](check.json)

## Context sensitivity in the actual service

Each cell contains the median active context and median round time across complete 40-round CUDA graph intervals. A prefill boundary excludes any interval spanning it. The separate V288 matching procedure supplies the nine-pair figure above. All 50 complete windows extend the context comparison across both private runs.

| Increasing context | V41 context | V41 round | V44 context | V44 round |
|---|---:|---:|---:|---:|
| Shortest repetition | 70,971 | 15.850 ms | 70,369 | 15.765 ms |
| Middle repetition | 96,470 | 15.941 ms | 97,026 | 15.891 ms |
| Longest repetition | 114,918 | 16.012 ms | 113,963 | 15.892 ms |

A roughly 44K-token difference coincided with just 0.163/0.127 ms more wall time. Each repetition also changes requests, expert routes, acceptance and cache layout. These observations measure the entire round and do not isolate its history-dependent cost. [Windows, context ranges and source](check.json) · [Original paired-window method](../v288_paired_cold_and_verify_intervals/README.md)

## What an outer graph or draft pipeline would have to hide

The same shape appears in 27 official H200 whole-target-layer jobs. After weighting their gate, attention, routing, expert and residual operations by the layer counts, the stock results range from 15.854 to 16.976 ms, with a median of 16.533 ms. Even their minimum exceeds the 15.841 ms service measurement. The standalone target uses synthetic routes, identity collectives, separately replayed layers and an independent numerical contract. Subtracting it from a private round cannot identify the actual share of draft work, collectives or host delay. [Whole-layer measurements](check.json) · [Single-rank layer contract](../../../muse_decode_layer/TASK.md)

The fastest measured standalone target totals **6.497 ms**. Under the explicit assumption that its time transfers unchanged, a 2× V41 streaming target would leave 1.424 ms for all remaining work. A 3× target would already exceed its 5.280 ms budget by 1.217 ms. V64's separate official one-user ratio of mean acceptance to median user speed gives an illustrative 19.685 ms period. At that anchor the same projection leaves 3.346 ms at 2× and 0.065 ms at 3×. These figures are break-even conditions for streaming at fixed acceptance. The competition also charges prompt work and measures a four-load frontier. [Thresholds for the measured round and official ratios](check.json) · [V64 receipt](../../runs/abd217baf0ed49fd.json)

In V297, the draft graph computes proposal IDs before the target can consume them. The next anchor and committed length follow the target prediction. The target's auxiliary features feed draft KV writes, and the graph can overwrite those features on its next replay. The scheduler can publish committed lengths by device event before the KV append completes. A combined graph must preserve that early event, the captured output lifetimes and the backend's existing fallback paths. It could remove exposed launch gaps. It would still execute the target and draft arithmetic. Sharing a vocabulary-weight read between those two heads for the same request would require target hidden states before the draft has supplied target input IDs. [V297 source hashes and call order](check.json) · [Worker event tests](../v140_decode_overlap_feasibility/README.md) · [Bonus and tap dependency](../v200_cross_block_draft_feasibility/README.md) · [Shared head contract](../v121_target_head_feasibility/README.md)

The missing observation is the critical path of a complete ordinary, warmed, single-request verification round across all eight ranks. Device-side activity time, CUDA graph launches and copies need the same timestamp basis. Distinguish time spent computing the draft from GPU gaps after its output is ready and before target launches, then follow the target and the committed KV write into the next round. An outer graph alone needs 7.920 ms of **removable** delay per 15.841 ms round to double this historical rate. A CPU call duration includes time overlapped with GPU work. A future profile can test the threshold directly. The archived service logs contain no such timeline. [Archived eight-rank profile inventory](../v265_native_serving_profile/README.md)

Run `python3 -B calibrate.py` in this directory. It reads the local public-base Git checkout and the existing job records. The study changed no serving code.

The retired temporary Git index and generated V297 source copies occupied 2,854,912 allocated bytes. The observed change in free volume space across cleanup was +2,949,120 bytes.
