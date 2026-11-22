# V70 fixed-budget DFLASH selector tree

V70 extends [V68](../v68_grouped_decode_verify/README.md) with an opt-in
greedy DFLASH tree. It derives a fixed number of proposals from the existing
draft selector lattice and verifies all of them with the same target model,
FP8 KV cache and target row count as linear verification. Each proposed
token attends to its committed prefix and its own ancestors, within the
layer's ordinary causal or sliding-window bounds.

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- V68 SHA-256 `6248a58842e1a22c1cda5a957d5ab464a6ec824efcdf8308bdcc9c6a1046e505`
- V70 SHA-256 `b58dc56b9ac4c4c7422834d00a5c84be1a1e8d35c6939a5e28c03df0bfcc261f`
- Patch 115,912 bytes across 19 permitted source files, longest added line
  104 bytes
- V68 flags and additional opt-in `--flag speculative-dflash-tree`

## Target verification and commit

The draft graph uses the existing selector scores to expand the most probable
available prefix at each step. It retains the original linear sampler on
the same graph for fallback. The target still uses its own greedy predictions
to accept a matching child or emit the target bonus token. The selector's
scores change which proposals are checked. They do not change what the target
accepts. One TP broadcast shares the selected token and ancestry metadata.

The packed V62 attention kernel reuses each request's existing page table.
The per-node depth sets its RoPE position and exact sliding-window lower
bound. An ancestor bit mask excludes sibling keys while preserving all
visible committed and ancestor keys. The same graph uses a linear mask for
sampled, grammar-constrained or otherwise unsupported requests. Metadata for
padding rows is reset to the linear mask before each replay.

The commit kernel gathers the accepted path in target order. The legacy
page-one request map permutes the existing fixed-width speculative suffix.
Full and SWA KV retain the same physical IDs and their pairing. The draft
materializer then receives auxiliary hidden rows and slot IDs in committed
order. A forward-stream event after both writes replaces the target graph's
earlier read-done event so scheduler and HiCache work can see the new map.
The CPU scheduler continues to enforce requested tokens and EOS.

The opt-in checks for a compatible Muse causal decoder, Hopper FP8 packed
attention, static verification, page size one, matching legacy request
maps, a selector draft, and a block and capture size supported by V62.
Translated pools, per-request SWA ring storage, attention DP or context parallelism,
compact draft windows and incompatible backend variants retain the linear
path. Only unmodified greedy requests on the draft CUDA graph use the tree.

## Validation

- Strict cached apply and reverse checks against the pinned base passed.
  Staging exact V68 in the isolated index leaves six additive source files.
  All 18 patched Python files compile. The official CLI's local diff command
  reproduced the frozen bytes with Git object IDs set to full length.
- [CPU execution of the candidate bodies](cpu_check.py) passed 1,400 proposal
  cases, 9,800 greedy-accept/slot permutations, 707 pointer-stable replay
  resets and 32 backend capability cases. The masked array implementation
  rejects any active access outside its buffer.
- The same CPU test executed the candidate partition and merge code for
  72 full and SWA attention cases. Maximum relative L2 against a dense
  visible-key oracle was 0.001504. CPU simulation does not establish
  Hopper FP8 MMA arithmetic or runtime correctness.
- The inherited V67 suite passed 2,286,109 allocator-tail cases and
  33,792 graph-shape cases.
- An independent extracted-kernel test passed 4,410 proposal/accept cases,
  including 681 full-width accepts, 965 nonidentity permutations, invalid
  selector scores, 64-bit IDs and the 131,071-token boundary. A separate
  source review traced full/SWA slot ownership, the SWA ring rejection,
  and post-commit scheduler/HiCache stream ordering.

Run `python3 cpu_check.py /absolute/path/to/patched/sglang` with NumPy
installed. It executes source AST bodies using array-backed Triton operation
models. Actual Triton compilation, graph correctness, output agreement,
peak memory and speed remain unverified on H200.

## Throughput hypothesis

At eight target rows, the tree can use spare depth to test alternatives while
keeping target batch geometry fixed. Its gains depend on the selector's
conditional confidence and how often the alternative branch matches target
greedy. The [V44 one-user check](../../runs/d40c45e5ca484591.json) records
four committed tokens per step. That mean reveals no branch distribution.
The [fixed-budget analysis](../v67_dflash_verify_capacity/tree_design.md)
shows why equal mean acceptance can imply very different tree outcomes.
This patch has no measured acceptance or speedup.

Costs include the selector frontier, one small TP broadcast, tree metadata,
and a reordered auxiliary tensor before the draft context projection.
At 32 requests, block eight and five 8,192-wide BF16 taps, that tensor is
20 MiB. The check must establish the total peak alongside existing graph
arenas. A meaningful private comparison would use the same load, run
settings and parent build, with the opt-in as the independent variable.

## Shared-prefix tile follow-up

The separate [prefix tile patch](prefix_mask_delta.diff) applies after exact
V70. SHA-256 `96bebb48f4acbff8b17826394e55530f99578201747d5396f6ad87a2c9a9e4c5`.
Only tiles intersecting the proposal rows apply the ancestor bit mask.
Earlier tiles keep each query's depth-dependent sliding-window lower
bound and use the original positions of committed keys. An eight-row suffix
can intersect at most two tiles with the existing tile sizes.

The updated CPU suite produced identical attention output bits before and
after the delta across 72 full and SWA cases, including tile-aligned and
cross-boundary prefixes. Maximum relative L2 against the dense oracle was
0.001504. To repeat the comparison, pass the pre-delta attention Python file
to `--attention-baseline`. The completed V70 patch bytes remain archived
separately.

## Scratch cleanup

The detached `okc-v70.4inrhjDC/sglang` worktree and two verified duplicate
exports were removed after checking the pinned ancestor, open files, stash
state, and exact reproduction of both retained patches. The scratch held
174,160 KiB. Volume availability immediately before and after removal grew
from 44,115,904 KiB to 44,297,148 KiB, an observed gain of 181,244 KiB across
concurrent workspace activity.
