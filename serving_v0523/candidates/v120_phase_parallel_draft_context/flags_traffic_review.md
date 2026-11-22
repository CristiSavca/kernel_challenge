# V120 launch flags and verify traffic

The frozen V120 launch manifest preserves all nine V111 flags and their values. It adds two enabled switches, `--speculative-dflash-direct-prefix-kv` and `--speculative-dflash-parallel-verify-context`. An independent match of every flag against each of the 76 fixed patterns returned zero matches across 836 comparisons. The official 96,445-byte patch changes 20 permitted Python sources. Its longest added line is 121 bytes. Both patch and delta digests match the manifest. [Exact flags and independent receipt](flags_traffic.json) · [Serving rules](../../RULES.md) · [Fixed flags](../../flags)

For the native eight-token verify width, a full 16- or 32-request pass supplies 128 or 256 rows. Each rank reads a contiguous row view of its existing BF16 weight, retains all 40,960 input channels and exchanges 1,024 of 8,192 output channels. The table counts one projection per rank, assumes one logical pass over weights, and counts a multiply-add as two operations. Ring traffic is an idealized send or receive volume per rank. [Projection and dispatch source](delta.diff)

| Per verify pass | 16 requests | 32 requests |
|---|---:|---:|
| Logical weight bytes, V111 → V120 | 640 → 80 MiB | 640 → 80 MiB |
| Projection work, V111 → V120 | 85.90 → 10.74 GFLOP | 171.80 → 21.47 GFLOP |
| Extra collective traffic, each direction | 1.75 MiB | 3.50 MiB |
| Local output + rank-major gather + assembled output | 4.25 MiB | 8.50 MiB |
| Excess of those three buffers over one ordinary output | 2.25 MiB | 4.50 MiB |

The full 640 MiB static weight already exists on every rank and remains available for prompt processing. On the CUDA path, the pinned gather allocates a contiguous rank-major buffer. Moving the rank dimension between rows and channels and reshaping to combine channels requires another contiguous output for these multirow cases. The table excludes NCCL workspaces, graph pools and allocator reuse. V64 measured +1.9 GB/GPU over stock against a +2 GB cap. V120 needs its own judge reading for memory, token agreement, TTFT and throughput. [Gather implementation](../../sglang/python/sglang/srt/distributed/parallel_state.py) · [V64 scored result](../../runs/abd217baf0ed49fd.json)

The second added switch inherits V104's prefix-valid draft KV store. At four committed positions per request it removes 320 KiB of BF16 intermediate writes and 160 KiB of reads at 16 requests, or 640 KiB and 320 KiB at 32 requests. The destination cache still receives each committed key and value. These byte budgets and the projection counts describe distinct work in the draft worker. Physical HBM and elapsed time depend on the target. [Prefix writer analysis](../v104_v64_direct_draft_kv/README.md) · [V120 cache-byte equivalence](prefix_bytes.json)
