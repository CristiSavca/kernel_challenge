# V248 partition planning and hardware evidence

V248 computes the request tile boundaries once per eligible ordinary verification forward. V247 recomputes these boundaries in each partition CTA and each live merge CTA. Across 62 full and sliding layers, the fully occupied parent runs the planner 42,040 times at 16 requests and 73,784 times at 32. The latter performs 7,083,264 logical reads of small int32 metadata values. The new producer launches two grids with 14 CTAs in total. Its consumers still read individual plan entries, scan physical KV and merge partial attention results. The logical storage for the two plans is 1,320 bytes per 32-request metadata object. [Parent-derived counts and archive hashes](independent_reach.json)

I executed the two production partition-ownership branches and both merge-range branches against the candidate's actual integer producer. The comparison covers 240 shapes, 24,352 plan entries, 3,456 partition pairs and 8,834 merge pairs. Empty queries at the start, middle and end, all-empty batches, partial tiles, uneven lengths and same-buffer rewrites agreed. An independent scalar calculation also checked tile counts, partition starts and right-inclusive ownership of exact ends. This test exercises geometry on CPU arrays. [Source-bound geometry results](independent_geometry.json) · [Reproducer](independent_geometry.py)

| Concurrent requests | Archived V41 log samples ending at this batch | V41 median step-duration proxy | Separate optimized H200 attention over 62 layers | Separate FA3 preparation kernels over 62 layers |
|---:|---:|---:|---:|---:|
| 1 | 69 | 21.2 ms | 0.507 ms | 0.132 ms |
| 16 | 6 | 83.3 ms | 2.097 ms | 0.132 ms |
| 32 | 1 | 142.9 ms | 3.604 ms | 0.137 ms |

The V41 log samples summarize reporting intervals. The proxy divides the ending batch size times reported mean accepted length by reported token throughput. Batch churn, prefill and idle time affect that estimate. The standalone kernels have a different implementation from V247. Their timings give scale for comparison, not a measurement of V248's gain. [Per-shape measurements and formulas](independent_reach.json)

The archive contains 767 timed requests across seven load repetitions. Fixing each user's TTFT and scheduling gaps, then eliminating **all** time after first token, raises raw throughput by 1.450× at 16 users and 1.360× at 32. A new scheduler run can change those timings. The available measurements do not support attributing a fourfold serving gain to shared partition planning. [Fixed-chronology sensitivity](independent_reach.json)
