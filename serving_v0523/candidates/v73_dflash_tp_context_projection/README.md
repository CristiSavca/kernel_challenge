# V73 tensor-parallel DFLASH context projection

V73 extends [V72](../v72_tree_multicast_decode/README.md) with output-channel
sharding of the unquantized DFLASH target-context projection. The checkpoint's
five 8,192-wide target taps feed a 40,960-to-8,192 BF16 linear layer.
V72 replicates this 640 MiB weight on every draft TP rank. V73 partitions
complete output channels among those ranks and gathers the projected
channels before the existing hidden RMSNorm and draft KV materializer.

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- V72 SHA-256 `e90c47ab73e66158b678893bb6fc7950c3b52f38cc3213ac6d7b7716d1b19705`
- V73 SHA-256 `9d82d7f0eabe1635316daf72cde73130a7c37629af237bd39d9eb6d6e6983cf7`
- Patch 133,765 bytes across 25 permitted source files, with a longest
  added line of 106 bytes
- Three-file [additive delta](delta.diff), SHA-256
  `03c374bb350efc00330d4d15f0938341cc1e889043d39cbc63cd0b965c60d4cf`
- V72 launch flags plus `--flag speculative-dflash-tp-context-projection`

## Projection ownership and arithmetic

`DFlashContextProjection` subclasses `nn.Linear` and runs the same
`F.linear` primitive on each rank's contiguous output-channel range.
Every output retains the entire input-feature reduction and its original
output dtype. The ordinary draft group's existing last-dimension all-gather
then restores channels in checkpoint order before normalization. This
introduces a collective with no numerical reduction.

The existing `sharded_weight_loader(0, parallel_group="tp")` freezes
ownership at construction. The model's checkpoint validation recognizes
the full projection dimensions before that loader selects local rows.
Native `fc.weight`, `model.fc.weight` and `encoder.fc.weight` names
keep their existing meanings. The option retains the replicated projection
for quantizable drafts, single-rank groups, non-CUDA platforms and output
widths that do not divide the group size.

The draft model's SINGLE_GROUP policy and `draft_tp_context` place the
input rows on the same owned group. Target prefill SP gathers auxiliary
rows before this projection. TP groups within attention DP process their
own replicated input rows. The group object and weight-loader rank are
bound during model construction and remain independent of the ambient
target scope during a later call.

The downstream draft hidden tensor is complete and has its original shape.
The KV projection, normalization, RoPE, accepted-row mask and physical
write locations follow their existing paths. V72 tree proposals and target
verification also retain their existing source.

## Work and memory

At TP8, each rank stores 80 MiB of context-projection weights, a reduction
of 560 MiB. For 256 target rows, this projection changes from roughly
171.8 GFLOP per rank to 21.5 GFLOP per rank. A complete weight sweep can
read one eighth of the prior weight bytes. These are source-derived work
counts.

The gathered 256-by-8,192 BF16 result is 4 MiB. The existing all-gather
first allocates a rank-major result, then its dimension transpose and
reshape can allocate a second complete result. Including the local output,
the conservative extra transient versus the replicated projection is
4.5 MiB at 256 rows. At a 16,384-row prefill it becomes 288 MiB. The
560 MiB weight reduction exceeds that source-derived transient, although
post-capture KV sizing can consume the released memory.

This candidate has no measured projection or serving speedup. The phase
fraction of this projection is unknown, and its communication overhead
depends on the group and input size. An H200 comparison needs to establish
both decode benefit and TTFT, including the changed GEMM shape and
post-capture cache capacity.

## Validation

- The exact V72 patch was staged in an isolated index. Only
  `arg_groups/fields/spec.py`, `models/dflash.py` and the new
  `layers/dflash_context_projection.py` form the additive delta.
- Strict cached apply and reverse checks passed. All 23 patched Python
  files compile. The official local `okc serve diff` export reproduces
  the frozen bytes with command-scoped full Git object IDs.
- [CPU checks](cpu_check.py) execute the new projection class, the existing
  sharded loader and the actual DFLASH load/project methods with array
  doubles. They passed 120 constructor cases, 336 projection/gather cases
  and 924 checkpoint/shape cases. Coverage includes TP2/4/8, empty and
  noncontiguous input, aliases, rejected checkpoint shapes and a changed
  ambient group after construction.
- An independent reviewer reran all those cases and checked the draft
  input replication, frozen rank ownership, full-K arithmetic boundary,
  gather-before-RMSNorm order and unchanged draft KV positions.
- Against the combined source, the inherited tree suite passed 1,400
  proposals, 9,800 acceptance and slot permutations, 707 replay resets,
  32 capability cases and 72 attention cases. The inherited capacity
  suite passed 2,286,109 allocator boundaries and 33,792 graph shapes.

Run `python3 cpu_check.py /absolute/path/to/patched/sglang` with NumPy
installed. The array checks model BF16 output rounding and frozen ownership.
They do not run PyTorch CUDA, cuBLAS or NCCL, and cannot establish bitwise
GPU equivalence between the original and sharded GEMM shapes.

H200 startup and collective execution, first/later-turn agreement, exact
request lengths, peak memory within stock plus 2 GB per GPU, TTFT within
twice stock and actual throughput remain unverified. This work performed
no submission, paid compute, model download or runner change.

## Scratch cleanup

Removed the detached `okc-v73.9R6RzZ6R/sglang` worktree and verified
duplicate export after checking the pinned ancestor, stash state, open
files, and byte-for-byte reproduction of the retained patch. The inspected
scratch held 174,172 KiB. Available volume space immediately before and
after removal was 31,013,792 KiB and 31,195,072 KiB, an observed gain of
181,280 KiB across concurrent workspace activity.
