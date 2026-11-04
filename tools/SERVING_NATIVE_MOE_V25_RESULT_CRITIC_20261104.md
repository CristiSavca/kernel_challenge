# V25 terminal independent critique — November 4

**Close the fixed attempt. No retry, shape-policy rescue, public promotion or serving integration.** Official private job `033a2d2eca374d82` passes correctness but fails the predeclared per-case performance limit. Frozen source remains `3a8c973b737dd074bcf5115f7418f5f0b34139fd4f9f637410ad1765ccbe050d`.

The raw receipt, archived source and external profile JSON agree. There is one passing setup, five passing tests, six passing benchmarks and six profiles, return code0, no flags and clean checks. The complete evaluator file inventory and image match the public247888 anchor. I independently reproduced the official six-load score **1.2537949062**, CI `[1.2349879826,1.2726018298]`, from the paired benchmark ratios. These are official adjusted timings; profiler durations and raw medians are different measurements.

| Tokens | Stock µs | V25 adjusted µs | Stock / V25 | V25 latency change |
|---:|---:|---:|---:|---:|
| 1 | 80.208 | 91.143 | 0.880022 | **+13.63%** |
| 8 | 141.936 | 131.963 | 1.075576 | −7.03% |
| 16 | 186.080 | 165.320 | 1.125576 | −11.16% |
| 32 | 202.240 | 170.274 | 1.187731 | −15.81% |
| 128 | 471.152 | 284.620 | 1.655372 | −39.59% |
| 256 | 612.288 | 330.156 | 1.854539 | −46.08% |

The aggregate exceeds the fixed1.15 speed requirement. M1 normalized latency is **1.1363354821**, failing the1.02 ceiling decisively. All six cases are affected; there is no unchanged control family. This rejection does not depend on the profile issue and does not require a causal explanation for the regression.

All six profiles contain the intended `native_serving_v25` core with the expected BIG specialization. The three large cases explicitly contain `_post_expert_norm_weighted_sum_kernel`. Small cases contain two generic reductions and elementwise work, but their displayed names are abbreviated before the required `where_kernel_impl`, `MeanOps` and `sum_functor` tokens. The frozen activation marker gate therefore remains unmet. This is missing direct evidence, not proof that finalization was omitted: frozen source calls the same qualified helper as the same-run stock baseline. No replacement marker rule or additional diagnostic is needed to close an already failing performance screen.

Both compiled pointers report576 threads,231,168 dynamic shared bytes,96 registers,24 local bytes, zero static shared, resident capacity1,132 CTAs and cooperative support. These establish valid launch admission; they do not measure achieved occupancy. Local24 is an actual attribute, not measured spill traffic or an isolated explanation for the slowdown. No C75xx warning appears in the retained log, but verbose compiler diagnostics are absent, so serialization or spill-instruction absence is unproved. The saved two-line resource log is an exact prefix of the receipt tail, followed only by profiler start/stop messages.

The core accounts for66.7%/76.6%/75.8% of profiled M1/8/16 GPU time and97.9–98.7% at the large shapes. This localizes most work to the new core without separating CAS, whole-grid scheduling, precision changes, packing or local-memory costs. Do not claim that any one of those caused the regression. The5 correctness errors are approximately0.00535–0.00560 relative L2, with worst reported row0.006067: passing official tolerance, not bit-identical Marlin or universal serving validation.

The one-bank, corrected-boundary implementation is useful architectural evidence: it compiles, runs cooperatively and improves several complete stock-component loads. It nevertheless falsifies the fixed uniform-all-M performance prediction. Its score is only0.528205 of publicnative247888's score; every normalized case ratio is below that incumbent. The public comparison uses different runs/seeds and different serving arithmetic, so it is descriptive, not matched serving A/B. It gives no reason for public promotion.

Preserve this source/result and stop the fixed attempt. Do not bolt on a small-M fallback, combineV22, relax the gate or retry. Any future materially different core must separately justify small-M work/parallelism, maintain exact BF16 boundaries and one-bank ownership, and freeze its own screen before measurement. Full-model weight replacement, generic EP/empty-route behavior, shared workspace/graph ownership, prefill/TTFT, actual serving alignment and eight-H200 performance remain unvalidated. No defensible whole-service gain or upper bound follows from this run.

Only the two assigned result-critic artifacts were written. The source-critic pins remain immutable. No provider, GPU, submission or polling call was made during this audit.
