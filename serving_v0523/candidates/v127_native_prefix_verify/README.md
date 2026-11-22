# V127 shorter target verification with a native-width DFLASH draft

V64 can safely commit a prefix of its eight-row greedy proposal **once the equivalent target predictions are available**. Its production graph, attention metadata and KV plan all assume the same width as the draft. Its scored receipt gives too little information to justify adding a second target runtime state. This is an offline feasibility result. No serving patch or GPU run was produced. [V64 scored receipt](../../runs/abd217baf0ed49fd.json) · [CPU study](prefix_study.json)

## What the shorter prefix returns

The first input row is the known anchor. The other seven are proposals. With a target width of `w`, the target evaluates the anchor and the first `w − 1` proposals. If an early comparison fails, it commits the matching proposals followed by the target's correction. If every proposal in this prefix matches, it commits them and the target's prediction from row `w − 1`. At width one it commits exactly one target prediction. For equal target predictions on identical causal prefixes, the committed output is exactly the first `min(D, w)` tokens of the full-width result, where `D` includes the bonus. The shorter pass resumes from that committed position.

The fixture loaded V64's actual PyTorch greedy acceptance and commit functions from the immutable SGLang pin. Across all 128 possible seven-comparison patterns and all eight widths, all **1,024** results matched both the clipped full-width output and the target token sequence. The fixture supplies fixed target IDs. Changing GEMM row counts or an attention kernel on Hopper can change finite-precision values, so this neither certifies a complete model nor tests near-tie argmax decisions. [Executable fixture](prefix_study.py)

## How much target work must disappear

At one user, V64 reported acceptance **4.0**, median decode **203.2 tokens/s/user** and scored throughput **6,979.2 tokens/s/GPU**. The judge receipt provides neither the accepted-depth histogram nor a decode phase breakdown. The following calculation models the reported acceptance as a mean of four commits per full-width call. The server's native per-request acceptance-length metric includes the bonus, and rounded aggregate reporting can differ from a pooled per-call mean. [V64 scored receipt](../../runs/abd217baf0ed49fd.json)

With only `1 ≤ D ≤ 8` and `E[D] = 4`, the exact same-anchor bounds for `E[min(D, w)]` are `1 + 3(w − 1)/7` and `min(4, w)`. The lower extreme mixes lengths one and eight. Constant length four attains the upper extreme. For width four alone, those distributions allow **2.286 to 4.000 commits**. For a fixed width at the same anchors, an idealized target whose entire pass cost scales linearly with rows can improve by at most `8 × min(w, 4)/(4w) ≤ 2`. Actual reruns can reach different anchors and trigger different kernels. This algebra does not bound a new full-system score.

The table models independent per-step matches at `p = 0.786666`, chosen to give four expected commits across eight rows. Each target row chooses eight distinct experts uniformly out of 256, independently across rows. The expected unique experts on each of eight ranks are `32 × [1 − (31/32)^w]`. The specified fraction of baseline pass time scales with this count. All other time, including the entire draft forward, stays constant. The one-user target attention still reads the committed KV prefix for the same full/sliding windows. Actual HBM reads, routing correlation, communication latency, cache reuse and tensor-core dispatch can all change the curve.

| Target width | Expected commits | Expert-time fraction needed to break even | Modeled speed at 50% | Modeled speed at 75% |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 87.1% | 0.439× | 0.705× |
| 2 | 1.787 | 76.2% | 0.701× | 0.980× |
| 4 | 2.892 | 59.1% | 0.944× | 1.115× |
| 6 | 3.577 | 46.7% | 1.008× | 1.077× |
| 7 | 3.814 | 41.8% | 1.010× | 1.040× |
| 8 | 4.000 | Baseline | 1.000× | 1.000× |

A second synthetic distribution mixes low-quality and high-quality calls with match probabilities 0.25 and 0.96, calibrated to the same mean of four. At a 50% scalable time share, a hypothetical classifier that knows the current regime perfectly can select widths one and eight and obtain **1.236×**. This assumes complete classification accuracy without a synchronization or dispatch cost. The paired examples demonstrate why a single acceptance value cannot select a profitable width. [Full sensitivity and formulas](prefix_study.json) · [Expert weight traffic model](../../runs/low_load_expert_traffic_feasibility.md)

## Production barriers on the measured V64 source

| Pinned component | What changes at shorter width |
| --- | --- |
| Draft startup and graph | DFlashWorkerV2 initializes its block size from the configured draft token count, passes it into the model's grouped convolutions, and folds the selector into that width's CUDA graph. A smaller global flag changes the checkpoint's draft computation as well. The intended algorithm needs an eight-row draft throughout. |
| Target graph and FA3 | The decode runner rejects uniform inputs unless their declared per-request width equals its capture width. On its ordinary eager path FlashAttentionBackend uses its own configured draft width for cache lengths, maximum query length and query offsets. A smaller token slice alone therefore cannot safely run even after graph rejection. V64 uses native FA3. V62's later split-KV kernel is absent from this parent. |
| KV, acceptance and overlap | The current worker makes one eight-column KV plan for the draft writes, target verify and catch-up. Prefix verification needs the first `w` columns **of every request** for target writes, positions and hidden rows. Taking the first `B × w` flattened slots aliases adjacent requests whenever `B > 1`. The existing worker also reshapes target predictions and hidden states with its draft width, double-buffers the eight-column accepted output, and reports that width to the scheduler. The scheduler uses the returned stride when it unpacks results. It is possible to retain the full reservation and output stride while changing the actual writes, but the two geometries must be explicit at their consumers. |
| Tier selection | The bundled adaptive configuration accepts EAGLE and EAGLE3 only. The ragged graph switch supports DSPARK only. DFLASH does not implement the adaptive worker protocol that chooses matching target backend/graph states. A policy that varies uniform width across TP ranks must choose one shared state before the collective forward. A host decision based on the current draft's device logits introduces a synchronization unless the entire branch is captured on device. |
| Memory | V64 reported +1.9 GB/GPU against the +2.0 GB cap. Its target decode graphs used 0.336 GB and draft decode graphs 0.082 GB. The extra cost of a restricted shorter target state cannot be inferred by multiplying the whole existing pool, and the receipt does not establish room for it. |

The nine-file V64 patch leaves the worker, decode graph runner, FA3 metadata and scheduler result-stride code at SGLang revision `297e54fd105e974cdf401e5b4d20253a22c16f90`. The worker's greedy commit semantics are usable across widths, while those system interfaces require more than request-level slicing. A defensible integration would preserve the trained eight-row forward, isolate a width-specific target backend and graph, report a coherent output stride, and demonstrate exact KV ownership and peak usage before allocation. The public score currently supplies neither the depth distribution nor the routing-time fraction that would justify that investment. [Measured parent patch](../v64_humming_grouped_sequence_parallel/patch.diff) · [Prior native tree budget study](../v118_native_tree_budget/README.md) · [Serving rules](../../RULES.md)
