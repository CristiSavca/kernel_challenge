# V238 tests whether hot expert weights can survive between target passes

Keep V235’s cache policy. An oracle allowed to retain **50 MiB per GPU** from one target pass to the next can cover only **0.095%** of the expected first-touch expert weights at 16 users. At one user it covers 0.416%. The cache budget gives the proposal the entire Hopper L2, with perfect route prediction, no competing allocations, and zero policy overhead. Adding persistence directives to production is unjustified by this bound. [Hopper cache documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/) · [Source-bound results](results.json)

Each active expert needs 18 MiB of packed W13 weights and scale words plus 9 MiB for W2. Each layer owns a distinct bank. Whatever happened in earlier passes, an address first requested in the current pass can hit because of earlier work only if it was among the resident bytes at the boundary. After its first request, any further reuse belongs to this pass. This argument also applies when the exact same experts fire on every pass. Across eight GPUs there are at most 400 MiB of initial resident data. Even if every token routes to the same eight experts per layer, the first sweeps across 62 layers are at least 13,392 MiB across the node. Earlier work on repeated row tiles and overlap within a pass remains applicable. [Within-pass multicast](../v216_decode_weight_multicast_feasibility/README.md) · [Projection overlap](../v226_expert_overlap_feasibility/README.md)

| Users | V64 scored tokens/s/GPU | Verify rows | Uniform expected first-touch GiB/GPU per pass | Full-cache share |
|---:|---:|---:|---:|---:|
| 1 | 6,979.2 | 8 | 11.73 | 0.416% |
| 4 | 12,531.4 | 32 | 33.37 | 0.146% |
| 16 | 17,347.5 | 128 | 51.41 | 0.095% |
| 32 | 20,547.4 | 256 | 52.30 | 0.093% |

The model uses distinct top-eight expert draws for each row and independent uniform routing across rows. V64’s 16-user figure above comes from the three-repetition scored report. V68’s independent one-repetition private check returned 17,512.9 at that load. These serving measurements supply scale and do not time a V235 kernel. [Scored V64](../../runs/abd217baf0ed49fd.json) · [Private V68](../../runs/04b01f958aa5487c.json)

Two physical standalone H200 histograms had 16 active experts at 32 rows and 26 at 256 rows. They imply 432 MiB and 702 MiB of distinct first-touch bank data per layer. The second shape requests 891 MiB when repeated row tiles are included. The profiling implementation reached 74.43% and 75.06% of DRAM throughput; it differs from V235. As a time sensitivity, 50 MiB takes 14.56 µs at 3.6 TB/s. Even placing all eight GPUs’ savings sequentially on the critical path gives 116.51 µs at that bandwidth, versus an illustrative V64 16-user verification period of 79.84 ms. Latency, cache stalls, and concurrent device work can change the time effect. [Hardware counters and histograms](../v213_live_decode_expert_reduction/physical.json) · [H200 bandwidth](https://www.nvidia.com/en-us/data-center/h200/)

Run `python3 check.py` to verify seven native decode source files and four dispatch/planning functions on V235 against V213, replay the two physical histograms, and enumerate 1,024 small oracle/cache configurations. The fixture separates within-pass hits from inherited hits and checks that the optimal inherited count equals the smaller of cache capacity and the number of previously touched pages required again. V238 is a feasibility result. [Reproducer](check.py)
