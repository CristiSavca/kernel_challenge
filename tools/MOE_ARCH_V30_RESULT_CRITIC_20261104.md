# V30 independent terminal review — close the fixed architecture

**V30 activated and passed correctness/resource gates, but its performance hypothesis failed.** Exact source `ac03c2cab756b1ed9dee884c3104a345661c64ec7da6af1894d9ee244c94e0f4`, job `d0510bc0412a4537`, scored **2.0335370900** with CI [2.0030340336, 2.0640401463]. The 1 setup / 5 tests / 6 benchmarks / 6 profiles and all checks are clean. I independently reproduced the six-load score, every normalized case ratio and both parent comparisons; the analyzer output exactly equals the saved root result.

| Measure | Versus public247888 | Versus private247888 |
|---|---:|---:|
| Whole score ratio | 0.856698 | 0.850911 |
| Affected M32/128/256 speed GM | 0.730721 | 0.724071 |
| Small-control latency GM | 0.995625 | 1.000031 |

| M | Normalized latency vs public | vs private |
|---:|---:|---:|
| 32 | 1.418532 | 1.420418 |
| 128 | 1.349304 | 1.366074 |
| 256 | 1.339047 | 1.357581 |

All three affected cases violate the 3% cap; the 4% affected-GM and 2% aggregate targets fail against both parents. The unchanged small controls satisfy their GM and individual caps. This is not an activation failure: all six new phase kernels appear with positive duration in each affected profile.

The explicit resource markers also satisfy the intended design. Cold A/B use 64 registers, no local allocation, 89,856 dynamic shared bytes and theoretical capacity two blocks/SM. Hot A/B use 96 registers and 231,168 shared bytes with capacity one; hot A has eight bytes of local allocation. The retained compiler output independently reports eight-byte stack/spill stores/spill loads for hot A, zero for the other three tile kernels. That spill is real evidence, but it is not a measured causal explanation for the slowdown. Capacity is not achieved occupancy.

The observed phase costs are:

| M | Cold A | Hot A | Cold B | Hot B | Route + final | Sum of displayed rows |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 72.0 µs | 26.3 µs | 41.8 µs | 24.1 µs | 7.6 µs | 171.8 µs |
| 128 | 62.5 µs | 51.0 µs | 41.6 µs | 33.8 µs | 10.1 µs | 199.0 µs |
| 256 | 84.1 µs | 84.1 µs | 41.6 µs | 60.7 µs | 13.4 µs | 283.9 µs |

The four tile kernels consume about 95% of these displayed kernel times; route and final kernels contribute only 4.4–5.1%. This rules out describing the result as merely expensive route/final kernels. It does not isolate launch gaps, cold/hot scheduling, duplicated activation requests, tile imbalance or lost overlap. The separate profiler samples differ from benchmark medians, and current-run cold/hot route counts were not logged. Prior NCU route counts must not be substituted.

The source preserves packed-weight bytes while doubling cold TMA issue count and activation copies per original output unit. It also replaces interleaved coordination with six ordered launch phases. The result rejects the **net benefit of this fixed tradeoff**, despite real two-block cold admission. It does not prove that every whole-K schedule is inferior or that one spill/local parameter is the cause. No tuning rescue, shape selection, same-code retry, decoder port or public promotion is justified.

The upload-order exception is material and remains explicit. Official acceptance was **22:17:47.135301 UTC**, start **22:18:08.161391**, and the plan was written **22:18:14.604524**; commit `dd4b6bc` is timestamped 22:18:14. Official completion was **22:21:25.489360**. The plan/source bytes match that commit. Thus the plan/commit occurred after upload and start, though before terminal completion. Root disclosed that missing optional README evidence made an earlier step fail and unchecked orchestration continued. Pre-upload messages, critic and analyzer already stated the same fixed gates; this does not permit calling the committed plan pre-upload or backdating it. Future submission orchestration must stop on any prerequisite failure.

The next separately scoped V31 diagnostic can investigate V29’s absent compressed-path activation. It does not reopen V30 or provide a speed result. Active public implementations remain unchanged. Exact provenance, resources and six paired cases are retained in [the result JSON](../experiments/iterations/20261104-moe-arch-v30-result-critic.json).
