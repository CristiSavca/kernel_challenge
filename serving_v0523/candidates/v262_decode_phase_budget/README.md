# V262 · Decode phase budget at one and 32 users

The archived H200 runs do not establish a route to 4× serving throughput from the target’s dense and expert projections alone. A uniform 2.833× improvement over V64’s 1.4119× score at fixed interactivity would require saving 64.70% of the time at every rung. This is one scoring scenario. To price its decode component, divide average committed tokens by median user decode rate. V64 yields 19.69 ms at one user (4.0 / 203.2 tokens/s) and 97.00 ms at 32 users (4.2 / 43.3 tokens/s). These quotients combine separate aggregates. [Official eight-H200 serving result](../../runs/abd217baf0ed49fd.json)

| Per target verification pass, ms | One user, 8 rows | 32 users, 256 rows |
|---|---:|---:|
| Saving required by the uniform scenario | 12.74 | 62.76 |
| Five dense projections across 62 layers | 5.91 | 7.52 |
| Original Marlin expert operations across 62 layers | 8.05 | 49.07 |
| Combined historical operator time | 13.96 | 56.59 |
| Modeled cold expert weight reads at peak H200 bandwidth | 2.62 | 11.70 |

For each shape, the dense control sums the **raw original baseline median** from 60 H200 measurements for each of QKV/gate, output, pooler, unpooler and router, then multiplies by 62. Subtracting the harness overhead reduces the dense totals to 4.64 and 6.30 ms. The separate expert control replayed a single-H200 Marlin graph with 10 and 252 local routes. The original expert path took 129.79 and 791.49 µs per layer. V40 took 123.62 and 381.63 µs. Adding these measurements assumes serial execution across the model. V64 used eight H200s and Humming, with routing imbalance, overlapping work and collectives. Its published result gives no native operator timings. [Dense operator measurements](../../../experiments/runs/b5538a0dd25d46be.json) · [Expert measurements](../../../experiments/modal_serving/20261112T191039Z/result.json)

At 32 users the combined original operator budget falls 6.17 ms below the required saving even at zero cost. At one user it leaves 1.22 ms for both families. Uniform independent top-eight routes touch an expected 7.18 local experts per layer at eight rows. Reading each selected expert’s 27 MiB bank once per layer at 4.8 TB/s gives the 2.62 ms estimate. That scenario depends on cold weights and the route distribution. The most informative missing decode measurement is the slowest rank’s captured Humming W13/W2 and routing/reduction time alongside the separate DFLASH draft pass, with live route histograms at both loads. [Routing model](../v129_routed_projection_pair/work_model.py)

On the independent eight-H200 V41 client trace, making all post-first-token streaming instantaneous yields 2.041× and 1.360× raw throughput at one and 32 users. This replay preserves observed TTFT and per-user gaps, then recomputes the last-finishing user in each repetition. The wall time includes prompts, queueing and runtime between requests. Operator timing, decode rate and whole-service throughput each have a different denominator. [Archived client requests](../../../experiments/modal_node/20261113T204343Z-bench/results__20261113T204343Z-bench__v41__load.json)

Run `python3 -B check.py` in this directory. The script reads local historical data, replays the original client intervals and prints the receipt including the SHA-256 of every input. V262 is an investigation. [Calculation](check.py) · [Machine-readable receipt](check.json)
