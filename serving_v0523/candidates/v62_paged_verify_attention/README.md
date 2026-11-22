# V62 Split-KV target verification across requests

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- V44 parent `0822e1279e1710bbb30f2e0e541dcf0055a105e7bc1032ae5b081638c1d27b4c`
- Patch SHA-256 `f0595f55c6c13380f67ed033bb4174780276188c44d779dfd9fee00bf517541f`
- Patch size 39,041 bytes across six files
- Check flag `--flag swa-full-tokens-ratio=0.25`

The Hopper DFLASH target verifier groups queries by KV head and divides
the live key tiles of the whole batch into nearly equal partitions. A
partition can finish a request and begin another in the same program.
A second kernel merges the per-query states with FP32 log-sum-exp. The
partition and request intervals intersect in at most `P + B - 1`
places, so the partial buffers grow with the sum of their counts. At
batch 32, TP8 and 132 SMs, that is about 5.2 MiB for full attention and
3.1 MiB for sliding-window attention in one invocation.

Existing projection, RoPE, E4M3 query conversion, E4M3 KV writes, page
tables, causal limits and sliding windows are preserved. K descale is
applied to scores and V descale to the output. Products of Q and K
accumulate in FP32. Softmax tracks actual row maxima in FP32. The
probability/value dot takes BF16 operands and accumulates in FP32.
Every E4M3 value widens exactly to BF16, and BF16 probability rounding
uses more precision than FA3's E4M3 probability operand. The new split
and probability rounding can change individual numerical results. The
output is BF16, and agreement still requires the serving judge.

The two kernels read live per-step lengths and IDs. Out-of-window and
padded positions are masked before page or KV reads. Per-invocation
scratch has no shared pool of counters or partial states. The existing
full CUDA graph backend runs each forward shape twice before capture,
covering full and sliding-window Triton specializations on that shape.
Native FA3 remains the route for other modes, devices, unsupported
layouts and attention features, including deterministic, ragged,
distributed context and modified-score attention.

## Performance evidence

At 32 live requests of 80,000 tokens, one full-attention layer has about
655 MB of unique E4M3 K and V to read per GPU. Sixteen full and 46 local
layers have about 11.26 GB of unique K and V per verify step, with a
2.35 ms read floor at 4.8 TB/s. Both V44 and V62 cover these same keys.
In the equal-length 132-SM example, FA3's default dynamic plan uses
roughly two splits per KV head per request, or about 313 tiles per work
unit. V62 offers 264 programs with about 152 tiles each, while allocating
parts in proportion to all live lengths. These are counts from the
algorithms. Software pipelining, occupancy, cache locality and split
merge costs need measurement. [Pinned FA3 split planner](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_prepare_scheduler.cu#L130-L167)

The existing H200 attention-board candidate V4 measured 4.63 to 2.15 ms
for a weighted 16-user verify step's attention kernels and 7.81 to 3.65
ms at 32 users, in private run `c21d391dbf604bb5`. Those cases use
random physical page slots and report isolated QKV-through-gate timings.
V4 uses a different probability path and a persistent atomic workspace.
V62 contains an independent implementation of the split and merge and
imports no V4 source. The measured attention savings of 2.48 and 4.16 ms
would be 6.2% and 10.4% of a hypothetical 40 ms verify step. The actual
serving step, attention savings and overall score are unknown.
[V4 benchmark source](https://github.com/rohanarun/Auto-research-agent/tree/018b7a102348ea3b73701e8be52d24f82d8c91ee),
[upstream CUDA source](https://github.com/vjeux/okc-2026)

## Verification

V44 patch hunk parity, pinned-base application and reversal, Python
compilation, and Ruff formatting and lint passed. The CPU harness
executed the exact Triton function bodies with NumPy-backed checked
pointers. All 110 small full and sliding-window cases matched a full-key
FP32 oracle to relative L2 0.001629 or better after the explicit BF16
probability rounding. Cases include empty rows, padded and ragged source
geometry, nonuniform descales, shuffled physical pages, expired SWA
slots filled with NaN, and page sizes 1, 16, 64, 128 and 160. The planner
and workspace ownership passed 890 long-context cases up to 131,072
tokens. Mocked dispatch passed 40 compatible and 12 fallback shapes and
confirmed separate live scratch allocations.

Run `python3 cpu_check.py /path/to/patched/sglang` with NumPy installed.
Independent review checked capture metadata, K/V descales, causal
visibility and pinned application. CUDA compilation, real graph replay,
token agreement, memory peak and performance remain unverified.

The temporary worktree, Ruff installation and validation copy occupied
196,668 KiB. The saved patch reversed cleanly before non-forced removal.
Shared-volume free space increased from 35,687,664 KiB to 35,891,428 KiB,
an observed increase of 203,764 KiB. The reproducible CPU harness remains
with the frozen patch.
