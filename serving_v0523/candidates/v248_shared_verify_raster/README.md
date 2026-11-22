# V248 reuses the verification tile plan across attention layers

During ordinary multi-request DFlash verification, V247's split-KV partition and merge kernels each scan the batch's lengths and cumulative query offsets to locate requests. V248 runs that integer calculation once per attention window in the forward metadata update. A small table gives each consuming CTA its interval boundaries and gives each partition its starting request. The computation reads the actual padded length and query-offset tensors on every forward. [Production change](delta.diff)

| Fixed-width batch | V247 per-forward batch scans across 62 layers | V248 producer CTAs | Device bytes per graph bucket |
| ---: | ---: | ---: | ---: |
| 1 | 18,232 scalar cases | 0, scalar path retained | 0 |
| 4 | 26,168 | 14 | 872 |
| 16 | 42,040 | 14 | 1,064 |
| 32 | 73,784 | 14 | 1,320 |

These counts assume eight queries per request, two local KV heads, 16 full-attention layers, 46 sliding layers and 132 SMs. The 14 CTAs run in two short launches per eligible forward. They replace the repeated in-kernel scans. The attention, partition and merge grids still process the same query rows and KV tiles. The short length arrays fit in cache, so the scan count cannot be translated directly into HBM bytes or serving time. [Geometry, measured comparators and score sensitivity](independent_reach.json)

Eager initialization owns the tables for one step. Full CUDA graphs retain two allocations at the existing metadata-bucket addresses, including across recaptures of the same batch size. Native replay updates the lengths first, then refills the original tables. Only matching native input tensors, window and partition count can use a published table. A failed refresh revokes publication, while a retry can refill the addresses held by captured consumers. The mixed-prompt path and one-request scalar case still run the original implementation. [Host graph replay](independent_graph.json)

The checked CPU replay matched 1,174 producer tables, 879,872 output components and 1,651,520 scratch words against V247. It compared indexed reads and tensor-descriptor reads, including partially covered tiles, empty rows and fragmented page tables. A second numerical comparison exercised the actual production partition and merge bodies. Offline Triton generated 72 Hopper specializations with zero spills. For 32 requests, the tensor-map partition dropped from 219 to 216 registers for full attention and from 218 to 214 for sliding attention. [CPU replay](raster.json) · [Independent numerical replay](kernel_replay.json) · [Hopper code generation](hopper.json)

The frozen patch is 408,604 bytes on public SGLang pin `297e54fd105e974cdf401e5b4d20253a22c16f90`. It inherits V247's eight flags. The full patch and the parent delta applied and reversed to the exact recorded trees. SHA-256 is `fc6a445d70bcbabe89031c1e1063785d2e9953f93792301de7810e43c3323e89`. H200 correctness, elapsed time and serving score remain unmeasured. [Official patch](patch.diff) · [Source and rules validation](validation.json)

The Triton interpreter ran 18 planner updates on reused buffers. Across 36 planned and fallback comparisons with V247, 13,334,880 FP32 intermediate words and 1,589,760 live BF16 output words matched bit for bit. A separate dense FP64 reference checked 7,104 components across both indexed and tensor-map reads. All 72 parent, planned and fallback Hopper builds assembled with zero spills. [Triton numerical replay](kernel_replay.json) · [Compiler evidence](hopper.json) · [Bound inherited configuration](inherited_configuration.json)
