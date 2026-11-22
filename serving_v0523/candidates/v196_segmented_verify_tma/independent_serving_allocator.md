# Segmented verification in a serving allocator replay

The native DFlash scheduler reserves two verification widths per request and allows host sequence lengths to lag by a round. This replay runs that allocation arithmetic and the page-one reservation code on 767 archived prompt, cache-hit and output lengths. It spans 7,072 allocation rounds, sampling 91 initial and 347 later verification batches per starting layout. The later samples include batches that have shrunk as requests finished. Each baseline and candidate sees identical addresses. [Implementation](independent_serving_allocator.py) · [Measured counts in the replay](independent_serving_allocator.json)

| Starting layout | Users | First-batch direct tile coverage, V195 → V196 | Later-batch coverage, V195 → V196 | V196 accumulation loops per later intersection |
| --- | ---: | ---: | ---: | ---: |
| Fresh free list | 16 | 36.67% → 95.94% | 44.35% → 95.63% | 2.60 |
| Fresh free list | 32 | 3.81% → 96.69% | 8.98% → 96.42% | 4.22 |
| 4,096-row free runs | 16 | 13.01% → 93.28% | 16.02% → 92.92% | 4.48 |
| 4,096-row free runs | 32 | 0.02% → 94.48% | 2.77% → 94.20% | 6.99 |
| 256-row free runs | 16 | 12.88% → 33.54% | 15.90% → 33.21% | 1.29 |
| 256-row free runs | 32 | 0.00% → 45.19% | 2.74% → 44.85% | 1.74 |
| 32-row free runs | 16 | 0.00% → 0.00% | 0.00% → 0.00% | 1.00 |
| 32-row free runs | 32 | 0.00% → 0.00% | 0.00% → 0.00% | 1.00 |

One warp can check every tile in a short partition intersection, so the production condition keeps V195’s whole-segment decision through 32 tiles. In longer intersections V196 can visit several runs. Each direct run needs four or more certified tiles and the same FP32 softmax state must cross every boundary. At the two smaller concurrency settings used by this trace, all intersections fit one warp’s check. A CPU execution of the two production mask helpers verified 291,768 tile classifications, including boundary-spanning runs and individual invalid origins. [Helper execution](independent_run_scan.json)

The interleaving of same-turn clients and the commit sequence are modeled. Each simulated active request advances by one to seven tokens per verification, excess reserved slots return after each turn and previously reusable prompt prefixes stay allocated. The client archive contains timings and lengths. Physical slot histories, CUDA timing and device-cache residency for the actual judge workload remain unknown.
