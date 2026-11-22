# V176 hardware comparison and score headroom

V64 holds the measured 1.4119× four-load score. V176 has no physical H200 result. It changes both prompt execution and target verification, so it is a credible hardware experiment. A 4× score would require another 2.833× in the geometric mean of the four stock-interactivity points. If only two points improved and improved equally, each would need 8.026×. A uniform 2.833× fixed-work throughput gain corresponds to removing 64.70% of elapsed time. Actual scoring also depends on interactivity, TTFT and correct complete outputs. [V64 judge report](../../runs/abd217baf0ed49fd.json) · [Calculation](independent_frontier.json)

| Concurrent users | V64 score point vs stock | V64 throughput at stock point, tok/s/GPU | Additional multiplier for 4× at this point |
|---:|---:|---:|---:|
| 1 | 1.1700× | 7,524.1 | 3.419× |
| 4 | 1.3160× | 13,146.0 | 3.040× |
| 16 | 1.5919× | 19,520.7 | 2.513× |
| 32 | 1.6213× | 20,547.4 | 2.467× |

V48b and V64 were scored on the same node with the same stock references and three paired repetitions. At one user, V64’s median streaming rate fell from 236.5 to 203.2 tok/s. The reported end-to-end rate rose from 147.4 to 153.7 tok/s, first-turn p90 TTFT fell from 8.310 to 5.419 s and stock-interactivity throughput rose 10.1%. Across the ladder the score rose from 1.1037× to 1.4119×. Client-visible latency can change the score more than a streaming-rate comparison suggests. [V48b judge report](../../runs/8ebed2bced8942ff.json) · [V64 judge report](../../runs/abd217baf0ed49fd.json)

The matched private workload at 16 users ran once each on two different NYC nodes with the same four launch flags. V64 reached 17,349.9 tok/s/GPU against local stock at 12,284.8. V68 reached 17,512.9 against local stock at 12,298.3. Raw throughput rose 0.94%. The judge’s indicative ratios at the stock interactivity point were 1.412× and 1.424×, a relative increase of 0.85%. The V64 scored 19,520.7 frontier value uses the three-repetition curve. For V176/16, use the same private schema and report both the raw and node-paired numbers. [Private V64](../../runs/ac89fddc4ee147ab.json) · [Private V68](../../runs/04b01f958aa5487c.json)

## What the client traces constrain

The archived V41 run contains 767 requests across four loads. Each user ran thirteen sequential turns. The table reduces one phase duration per request, holds that user's observed gaps fixed and takes the maximum time across simultaneous users in each repetition. Ratios are total original wall time divided by total transformed wall time. Baseline reconstruction agrees with the recorded makespans to within 25 ms. TTFT includes waiting for service. The residual after the first token includes all later response delays. Both can change when scheduling changes. These are fixed-timeline sensitivity tests on the earlier model build, with no estimate of V176 kernel time. [Reproducer](independent_frontier.py) · [Raw V41 client trace](../../../experiments/modal_node/20261113T204343Z-bench/results__20261113T204343Z-bench__v41__load.json)

| Users | Repetitions | Halve each request’s TTFT | Halve time after first token | Remove all time after first token |
|---:|---:|---:|---:|---:|
| 1 | 3 | 1.328× | 1.339× | 2.025× |
| 4 | 2 | 1.414× | 1.261× | 1.673× |
| 16 | 1 | 1.514× | 1.189× | 1.450× |
| 32 | 1 | 1.575× | 1.156× | 1.360× |

The V44 one-user trace gives 1.355× when time after first token is halved. V175’s separate diagnostic removed the whole measured pool/unpool projection cost from an approximate V64 decode period, yielding 1.129×, 1.092×, 1.043× and 1.042× at the four loads. Neither diagnostic establishes a ceiling for V176. V176 also changes long prompt execution, expert weight loading and the verification scheduler. [V44 client trace](../../../experiments/modal_node/20261113T213641Z-bench/results__partial__v44__load.json) · [V175 component model](../v175_decode_collective_pipeline/check.json)

V171’s single-sequence, 16,384-row prompt geometry cuts two logical gathered tensor outputs from 1,536 to 256.125 MiB per rank and cuts the replicated context FC calculation from 11.00 to 1.37 trillion FLOPs per rank. V172 can remove 60 repeated GPU scheduler-prepare nodes over 62 verification layers at eligible geometries. V174 halves the number of launched thread blocks in its medium expert projection pair from 4,096 to 2,048 and reduces prefetch depth for small down projections. Serving receipts omit internal prompt batch sizes, expert histograms and per-kernel timings. The measured shares of these opportunities remain unknown. [Prompt geometry](../v171_prompt_endpoint_context/README.md) · [Verification scheduler](../v172_streamed_experts_verify_schedule/README.md) · [Expert tradeoff](../v174_multirow_expert_tma/README.md)

## Hardware priority and quota

A 16-user private V168 comparison against V64 is the direct way to establish the common parent’s performance. If V168 passes on Nov19, V173 retains the disjoint third-slot comparison on Nov20. All usual Nov20 paths already have private reservations. V176 can use one otherwise unclaimed third slot only when the accepted Nov19 V98/V95 alternative displaced V168, both separate V120b and V139 controls failed, V129 occupied the first slot, V163 passed a complete 16-user private run in slot two, competing receipts are absent, and the live quota is exactly two used and one left. A passing V163 is an ancestor control for the whole V176 package. V163 lacks the later V168 changes as well as V176’s additions, so this comparison cannot isolate them. [Conditional V176 plan](../../runs/nov20_v176_runner.plan.json) · [Existing V173 plan](../../runs/nov20_v173_runner.plan.json)

The Nov18 status read at approximately 20:38 UTC showed three of three private checks used, with the distinct daily scored allowance still available. V93/16 was queued. No date has a guaranteed free V176 slot in the existing allocations. Use the private result to examine correctness, complete later turns, raw/normalized throughput, both TTFT guards and peak memory. V64 had only 0.1 GB of reported margin. A scored four-rung run is what can establish a new frontier. [Official serving track](https://openkernelchallenge.dev/api/v1/serving) · [V176 validation](validation.json)
