# V286 alternates peer input buffers during long prompts

V286 removes 62 device barrier launches per rank from an eligible 62-layer, 16,384-token prompt. V279 publishes each input shard, lets all ranks read their locally routed expert rows, and synchronizes again before reusing the buffer. V286 alternates two buffers in the same symmetric allocation. The next publication on every rank follows that rank’s previous reads. Completing that publication makes the older buffer safe to overwrite. The original 62 publication barriers, peer input copies, routing scores and expert arithmetic all still execute. [Production delta](delta.diff) · [Pinned PyTorch CUDA barrier](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L158-L185)

Empty readers also publish and advance the buffer index. A rank-local hook or shape veto sends all ranks through the existing gathered-input fallback. A capture stub never changes the index, and a breakable graph executes the Python publication during each replay. When the larger allocation or either publication probe fails on any rank, all ranks try the original single buffer and two barriers. At the TP8 limit of 16,640 backed prompt rows, the second buffer adds 16.25 MiB of BF16 payload per GPU, shared across layers, plus an eight-entry device address table. The allocator rounds the physical reservation. [Control flow and ordering](ordering.json) · [Memory and exposure](work.json)

| Concurrent users | Archived size-eligible prompts per repetition | Removable barrier epochs per repetition |
|---:|---|---|
| 1 | 6, 4, 7 | 372, 248, 434 |
| 4 | 15, 20 | 930, 1,240 |
| 16 | 75 | 4,650 |
| 32 | 150 | 9,300 |

V41 supplied the prompt sizes and wall times. The table assumes every eligible call uses two-buffer peer packing. At 16 and 32 users, net savings of 1.688 and 1.559 ms per eligible layer call would be required for a 5% throughput improvement at fixed work. Reaching a raw speed factor of 4/1.4119 would require 22.94 and 21.18 ms per call under the same assumption. Device barrier latency, end-to-end speed, per-load TTFT, token agreement and combined peak memory require H200 measurement. [Historical comparison](work.json)

## Verification

Real Gloo ranks replayed the production Python setup and publication methods at world sizes 2, 4 and 8, with shared BF16 host slabs and synchronous stream stand-ins. All 31,465,472 compared BF16 words matched. Coverage includes rank-local failure of large allocation, registration and probing, both allocation attempts failing, deliberately slow readers, empty ranks and skipped calls. Per-stream ordering plus collective publication gives a read-before-reuse path for all 13,608 checked rank pairs across 12 grid and epoch combinations. Removing the retirement barrier from a single-buffer protocol lost such a path in every combination. [Gloo transport](transport.json) · [Ordering control](ordering.json)

The 407,889-byte official export reconstructs against the public pin. It changes one source file relative to V279 and keeps its eight launch flags. The full diff and the parent delta apply and reverse with strict whitespace checks. The maximum added line is 142 characters. [Official patch](patch.diff) · [Rule and hash checks](validation.json)

## Cold expert scheduling alternative

A replay of the 12K and 16K route fixtures measured issued tensor rows per persistent CTA and N tile. At 16K, pooling all W13 and W2 jobs into an ideally balanced grid with no dependencies gives 1.097× modeled projection headroom with 128 resident blocks and 1.119× with 132. Keeping the original projection geometry but splitting each rank into four sequential windows of eight experts reduces rank 3’s live gate/up intermediate from 126.75 to 31.79 MiB. It raises projection launches from two to eight and increases modeled issue time by 1.2% to 11.5% across the eight ranks and two grids. These estimates assume equal throughput per issued full-K tensor element. They omit synchronization, register effects and cache reuse. The compact-window design was left out of the serving patch. [Per-rank schedules](cold_schedule.json) · [Archived route fixture](../v274_four_block_prompt_multicast/routes.json)
