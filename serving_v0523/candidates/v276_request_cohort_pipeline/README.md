# Can independent requests or pipeline stages approach 4×?

The five scored receipts provide small and large batch streaming rates. The V41 H200 client and server receipts also provide cold misses, first-token time and per-request streaming intervals. The calculations below combine those records under stated scenarios. Actual target and draft GPU-stage durations were unavailable. [CPU replay](cohorts.py) · [Inputs and results](cohorts.json)

## Separate draft and verify stages

Across requests, one cohort may draft while another verifies. Within a request, the next anchored proposal consumes the verified bonus, accepted length and target features from the preceding step. The pinned worker enforces this causal order. Its full CUDA graph uses a single executable per shape. Concurrent replays of one executable serialize. A real implementation would need distinct graph resources and buffer ownership, plus compatible collective ordering across ranks. [NVIDIA CUDA graph execution](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__GRAPH.html) · [Pinned worker paths and hashes](cohorts.json)

For a full batch split into g cohorts, let Rsmall and Rfull be the official median output rates per user and f be the draft fraction of a small cohort's service time. On two independent resources, steady-state streaming speed scales by Rsmall / (g × max(f, 1−f) × Rfull). This assumes enough live requests, no contention, no extra work and predictable service periods. Pinned rates already include accepted outputs.

For V64 at 32 users, two 16-user cohorts break even only with f between 42.15% and 57.85%. Perfect balance yields 1.157× streaming. At 16 users, four cohorts of four yield 1.122×. The calibrated four-rung end-to-end replay moves from 1.41190× to 1.44596×. With a 25% draft share, keeping full batches wins at all four modeled rungs. Eight- and two-user child-cohort rates were unavailable and the scenario cannot bound those shapes. In the V41 closed loop, at least two clients streamed for 29.84% of the 32-user wall interval, and all 32 streamed for 0.31%. Client activity provides exposure, not a device-occupancy measurement. [V64 score receipt](../../runs/abd217baf0ed49fd.json) · [Replayed frontier and per-rung rates](cohorts.json)

## Two independent four-GPU replicas

In the one-copy V41 cold workload with existing cache representations, the 32-user rung computes 2,455,793 cold rows. Even distributing these evenly over two servers and retaining only the measured 2.56 GiB draft weight footprint per rank, TP4 target weights and KV caches sum to 143.91 GiB per rank including the modeled draft KV. The server exposed 140.40 GiB physical. With a hypothetical window-only draft KV, the estimate falls to 138.21 GiB against 138.03 GiB available before loading. Both sums omit extra sharded draft/vocabulary weights, captured graphs, pending slots, fragmentation and workspaces. The numbers constrain this particular topology and retention scheme. [Memory inputs and arithmetic](cohorts.json)

## Pipeline 31 target layers per four-GPU group

A cold 16K-token chunk can advance through the second group while the following chunk enters the first group, provided per-layer KV dependencies remain in order. The V41 server's batches after each cache flush sum exactly to the independently computed client misses. The seven first waves contain 6, 5, 7, 15, 21, 76 and 150 chunks. In a two-machine flow-shop, if each stage takes half the existing per-chunk time multiplied by r (the same layer's four-GPU/eight-GPU cost ratio), and work scales with new rows, pipeline time is r × (sum + max) / 2 in row-time units. The model grants balanced layer cost, free boundary transfers, immediate scheduling, zero setup and no extra draft feature gathering.

| Cold users | Source chunk count | Cold gain at r=1 | Cold gain at r=1.5 | Cold gain at r=2 |
|---|---:|---:|---:|---:|
| 1 | 5–7 | 1.60–1.74× | 1.07–1.16× | 0.80–0.87× |
| 4 | 15–21 | 1.87–1.91× | 1.25–1.27× | 0.94–0.95× |
| 16 | 76 | 1.97× | 1.32× | 0.99× |
| 32 | 150 | 1.99× | 1.32× | 0.99× |

At 32 users, doubling cold throughput would require a four-GPU stage to run each layer at least as fast as 0.9934 times its eight-GPU time. Reaching 4× would require 0.4967. Actual per-layer TP4 time and attention costs by prefix position are unmeasured. DFLASH's pinned validator requires pipeline size one, so native launch arguments alone cannot implement this split. [Source-bound flow-shop replay](cohorts.py) · [Native constraint and full scenario data](cohorts.json)
