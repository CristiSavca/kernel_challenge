# V72: fixed-budget tree verification with decode transport improvements

V72 implements the next decode experiment on V68. The main algorithmic change
uses the existing DFLASH selector lattice to propose a tree within the same
eight target rows. This can accept an alternative continuation after the most
likely draft branch diverges. The target model still supplies every accepted
greedy token and the bonus token. V72 also shares activation loads between
paired Hopper CTAs and removes eligible all-reduce staging copies.

- Pinned SGLang: `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Final patch SHA256: `e90c47ab73e66158b678893bb6fc7950c3b52f38cc3213ac6d7b7716d1b19705`
- Payload: 128,042 bytes, 23 source files, maximum added line 106 characters
- State: CPU and independent source review passed; no H200 test or speed result
- Best measured full serving score remains V44's **1.1847**

## Composition

[V68](../v68_grouped_decode_verify/README.md) supplies grouped experts,
sequence-parallel prefill, packed verification attention and cache-capacity
repairs. [V69](../v69_humming_decode_a_multicast/README.md) adds two-CTA
activation multicast for compatible grouped expert GEMMs. The same expert,
M and K coordinates are shared across adjacent N tiles. All weight bytes,
experts and BF16/FP32 arithmetic boundaries remain unchanged.

[V70](../v70_dflash_selector_tree/README.md) supplies the selector tree,
ancestor-aware attention, target-greedy accept walk, and an accepted-path
permutation of the existing page-one KV slot IDs. Full and SWA physical IDs
stay paired. Auxiliary hidden rows are gathered in accepted order, and a
post-commit stream event protects the shared request map from scheduler and
HiCache reuse. Unsupported requests retain linear verification.

[V71](../v71_direct_multicast_workspace/README.md) writes attention projection
and grouped expert output into the existing symmetric pull buffer when the
incumbent communicator selects multicast. The unchanged collective reduces
that buffer and its immediate consumer finishes before reuse. Guards reject
overlap, incompatible collective policies and unsupported execution modes.

The final attention refinement applies ancestry masks only to tiles containing
speculative nodes. All earlier prefix tiles still obey each query's exact
sliding-window lower bound. At width eight, at most two tiles need the extra
ancestry operations. The frozen V70 plus this separate delta remains available
for an isolated tree experiment. `patch.pre_prefix_fastpath.diff` preserves
the integrated source before this refinement; it is not the active candidate.

## Launch settings

Use the same four V68 flags plus the tree opt-in:

```text
--flag swa-full-tokens-ratio=0.25
--flag moe-runner-backend=humming
--flag speculative-moe-runner-backend=marlin
--flag enable-layernorm-sp
--flag speculative-dflash-tree
```

The live judge profile is expected to satisfy the guards: Hopper, TP8/EP8,
FA3, FP8 target KV, page one, D64/GQA8, legacy shared target/draft request map,
static target verification and a captured selector draft. The draft model's
native 2,048-token attention window is distinct from the optional compact
`speculative_draft_window_size`, which remains unset. Live preparation should
emit `DFLASH selector tree prepared for greedy CUDA graph batches.` Actual
flag acceptance and activation still need official evidence.

## Validation and rules

The official local `okc serve diff` export exactly matches the frozen patch.
Forward/reverse applicability, paths, whitespace, and compilation of all 21
Python sources passed. Only text source under `python/sglang/` is modified;
no model loader, precision reduction, hidden workload condition, output cache
or off-timed input computation is introduced. The independent reviewer
re-read the live rules and found no source-level blocker. Runtime limits and
organizer acceptance remain separate gates.

The [validation receipt](validation.json) records exact component parity and
combined checks. Final V72 passed 1,400 proposal cases, 9,800 accept/slot
permutations, 707 replay resets, 32 capability cases and 72 full/SWA attention
cases. The maximum relative L2 against the dense CPU oracle was 0.001504.
Before/after prefix optimization outputs matched bit for bit in all 72 cases.
The independent review checked 1,680,000 visibility and position cases and
executed the exact eligibility code against the judge-profile mock.

The combined workspace, grouped decode and prefill suites also passed;
their files are unchanged by the final attention-only refinement. V69's
4,617 configuration assertions, 168 guards and 880 schedule simulations
apply through exact source parity. These are AST/array and source checks on
an Apple Silicon Mac. They do not execute CUDA, Triton, Humming or NCCL.
[Independent review](independent_review.md) and [final tree test output](cpu_tree_result.txt).

## Experiment and decision gates

The scheduled Nov17 00:03 UTC private checks remain V64, V62 and V68. No V72
submission, paid compute, scored promotion, commit or push was performed.
This candidate is ready for a later private experiment after inspecting those
results and available quota; it must not silently replace the scheduled control.

The tree hypothesis is more committed target tokens per verification step
without increasing target row count. The V44 one-user check reports mean
acceptance four, which reveals no selector confidence or branch distribution.
Eight target rows also cap the useful accepted chain; breadth can reduce depth.
The three improvements' gains must not be multiplied into a serving forecast.
No 2x or 3x result is established.

Require H200 compilation and live tree activation, changing-batch graph replay,
correctness and later-turn agreement, exact requested output counts, peak
memory no more than stock +2 GB per rank and TTFT no more than 2x stock at
each load. The tree additionally allocates a reordered auxiliary tensor of
up to 20 MiB at 32 requests; graph pools and the inherited paths determine
the actual peak. V44's latest +1.9 GB check leaves little measured margin.

A useful result should improve committed tokens per step enough to cover the
selector frontier, metadata broadcast and auxiliary gather, then beat the
passing V68 parent at the same load by at least 5% as a provisional usefulness
gate. Source or activation failure blocks promotion. A slowdown calls for a
specific diagnosis or an isolated tree/transport comparison, not an unchanged
retry. The full serving frontier must establish any eventual 2x/3x claim.
