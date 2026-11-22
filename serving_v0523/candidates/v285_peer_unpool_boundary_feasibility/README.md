# Can the Hopper FFN boundary read unpool columns directly from peers?

A direct reader can retain the full-K projection and both signed zeros. It would still transfer every remote BF16 column to every consumer. The existing SGLang fused operation publishes through a multicast push plane. It replaces each positive zero with negative zero as its slot-arrival marker, and its Python selector requires compute capability 10 or newer. The sign can survive normalization into the FP32 residual. Adding a positive-zero branch to a negative-zero stream gives positive zero, while a branch converted to negative zero gives negative zero. The existing operation therefore needs a different transport to match V283 at its larger Hopper shapes. [Source hashes and signed-zero witness](check.json) · [V283 boundary](../v283_rank_major_unpool_boundary/README.md)

## Work at eight ranks

The model uses 61 nonterminal boundaries, 8,192 columns and eight target verification positions per user. At one user the existing small-row Hopper multicast has priority. Every larger V283 shape already consumes the rank-major result from its all-gather. A direct reader could remove the local receive tensor, including its write and subsequent normalization read. Each consumer would continue to fetch seven eighths of every branch from peers. One rank serves its slice to the other seven ranks. [V283 shape and eligibility model](../v283_rank_major_unpool_boundary/work.json)

| Users | Rows | Incoming branch payload across 61 boundaries per rank | Receive tensor writes plus following reads per rank | Time to move those local bytes at 4.8 TB/s | Approximate V64 verify interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 32 | 26.69 MiB | 61 MiB | 13.3 µs | 35.6 ms |
| 16 | 128 | 106.75 MiB | 244 MiB | 53.3 µs | 79.8 ms |
| 32 | 256 | 213.50 MiB | 488 MiB | 106.6 µs | 97.0 ms |

The last column divides published acceptance by median streaming tokens per second per user. These are rough intervals assembled from aggregate metrics. The local read/write price excludes caching, barrier delays, allocation, collective launch gaps and the throughput of remote loads. A fourfold reduction in those intervals through 61 calls alone would need actual savings averaging 438, 982 and 1,193 µs per boundary. The archive has no synchronized H200 measurement isolating the V283 gather and FFN boundary. [Official V64 report](../../runs/abd217baf0ed49fd.json) · [Calculated rows](check.json)

## Pointer lifetime and ordering

V61 and V75 use the multicast push convention for the column projections. V71 can give an eligible producer the local symmetric pull workspace and invokes the existing in-place all-reduce on it. The ordinary full-K column GEMM dispatcher takes an input, weight and bias and returns its own tensor. Writing it directly into the pull buffer would require a verified output-capable dispatch or an extra copy. V230 owns a separate symmetric allocation and uses a copy and rendezvous barriers around eager or breakable long-prompt work. Its sparse expert consumers can omit rows. All 8,192 unpool columns contribute to each replicated FFN boundary. [V61](../v61_hopper_decode_projection_gather/README.md) · [V75](../v75_sharded_decode_projections/README.md) · [V71](../v71_direct_multicast_workspace/README.md) · [V230](../v230_peer_packed_prompt_routes/README.md)

The pinned communicator can register otherwise ordinary GEMM outputs. During actual full CUDA graph capture, `_allocate_graph_row` records the producer address and size. The IPC or VMM path populates a device-side row of peer pointers when capture completes and synchronizes before replay. Warmup executes immediately and needs a working ordinary path. The fused reader would have to reuse the Hopper single-CTA norm’s vector mapping, FP32 arithmetic, BF16 roundings and output casts. Per-block communication needs the same persistent row assignment on every peer. The pinned Hopper tuning provides 64 ordinary pull blocks, so 128 and 256 rows require two and four rows per block with that grid. Collective ordering, captured pointer reuse and retirement across ranks still need device testing. [Pinned communicator, barrier and kernel hashes](check.json)

A two-rank replay enumerates publication, remote reading and reuse of one allocation over two epochs. With corresponding publication and retirement ordering, all 16 legal interleavings use the correct values. Removing publication admits 20 bad interleavings among 36. Removing retirement admits eight among 24. The latter includes a producer writing its next epoch before the peer has read the previous one. A release after all relevant lanes complete their last remote load and an acquire before a peer can reuse the allocation are essential. [Executable witnesses](check.py)

V285 is a feasibility study. A new peer-read kernel and hardware timings would be required to justify a check. Run `python3 -B check.py` from this directory to reproduce its receipt.
