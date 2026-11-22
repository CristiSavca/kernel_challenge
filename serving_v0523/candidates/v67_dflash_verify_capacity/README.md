# V67 DFLASH verification capacity

V67 adds request-slot and target-verification graph headroom to the exact
[V65 patch](../v65_grouped_sp_verify_attention/patch.diff). It retains V65's
block width, launch flags and attention dispatch. CUDA execution, numerical
agreement, peak memory and throughput remain unmeasured.

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- V65 SHA-256 `80e41ae32e748857053f553359f4d555cbe983534cff465f4910551723c30337`
- V67 SHA-256 `fbdcc9ff342a6bb0066b373ba7a659de4ba3a6bda3791d0f2cf57839e7a7ea57`
- Patch 68,665 bytes across 13 allowed Python files, with a longest added
  line of 98 bytes
- Launch flags `moe-runner-backend=humming`,
  `speculative-moe-runner-backend=marlin`, `enable-layernorm-sp`,
  `swa-full-tokens-ratio=0.25`

## Capacity bound

Let `C` be the maximum model context, `B` the configured DFLASH block
width, `P_alloc` the allocation-page width and `P` the attention-page width.
The overlap scheduler can launch the next decode before consuming the preceding
step's host result. If the host prefix length is `C0`, the preceding step may
have published as many as `B` more tokens on device. The next block can touch
slots through exclusive limit `L = C0 + 2B`. The existing per-request allocator
reserves two blocks and rounds up to the allocation page.

```text
A = ceil(L / P_alloc) * P_alloc <= C + 2B + P_alloc - 1
last allocator slot = A - 1
last attention page = floor((L - 1) / P)
```

The shared headroom and both unified Mamba pool builders now reserve space
for this bound at page size one as well as larger pages. For fixed-width
DFLASH, `C + max(4 + B, 2B + P_alloc - 1)` covers both the legacy margin
and the new bound. The headroom helper uses the configured speculative
capacity when provided. At context 131,072 and page size one, the target
verify table grows to 131,088 columns at width 8 and 131,104 columns at
width 16. The bound conservatively uses the entire configured context.
The existing scheduler admission and physical KV allocator still enforce
their own limits.

FA3 and FA4 size full and SWA target verify graph tables and strided indices
from the allocated request row width. Unified in-graph capture and ordinary
replay use the bound table's actual width. Both paths pass live sequence
lengths and leave padding in the existing sink. Ordinary decode, draft
extend and the EAGLE tree retain their prior table sizes.

The FA3 precomputed scheduler buffer is indexed by batch rows and only
attached to ordinary decode. The target verify call has no precomputed
scheduler metadata and uses native table dimensions and cache lengths for
partitioning. The FA4 target verify call treats the table width as storage
and uses actual cache lengths for masking. Its fallback split heuristic
uses the physical cache span when the wrapper leaves `max_seqlen_k` unset.
The V62 custom path uses the current live lengths and checks its table
dimensions separately. For width 16 and 32 slots at page size one, the
two target tables add 8 KiB of int32 storage per backend instance.

## Verification

- Applying V67 to the pinned index succeeds with strict whitespace checks.
  Reversing V65 leaves only the changes in `allocation_sizing.py`,
  `kv_cache_configurator.py` and `flashattention_backend.py`.
- All 13 patched Python files compile.
- [Capacity checks](cpu_check.py) execute extracted sizing and graph metadata
  source with CPU shape doubles. They passed 2,286,109 allocator tail checks
  and 33,792 graph shape cases. Coverage includes widths 1–32, 11 attention
  page sizes from 1 through 640 including 160, DCP factors 1, 2 and 4, SWA
  on and off, unified Mamba builders, and graph buckets with both one live
  request and 31 added padding rows.
- The shape tests verify the tensors and writer bounds. They do not run
  CUDA allocators, FA3, FA4 or physical KV reads.

Apply the patch in an isolated pinned SGLang checkout, then run
`python3 cpu_check.py /absolute/path/to/sglang` from this directory.
An H200 check remains necessary for the extended context path, including the
last supported context positions and replayed graph padding. The checkpoint
was trained at width 8. A width 16 trial would be a separate experiment with
its own acceptance, precision and memory checks.

## Decode architecture follow-up

[Tree speculation feasibility](tree_design.md) documents the existing model
selector, exact verify and commit semantics, the blocking integration work
and a [bounded CPU proof](tree_cpu_check.py). The frozen patch does not
contain tree speculation.

## Scratch cleanup

Removed only the detached `okc-v67.PDtEjA/sglang` worktree and its empty
parent after checking the fixed ancestor, open files, stash state and
byte-for-byte patch reproduction. The worktree had 173,984 KiB of allocated
files before reverting the patch. Available volume space immediately before
and after removal was 55,781,964 KiB and 55,962,952 KiB, an observed gain of
180,988 KiB across concurrent workspace activity.
