# V64 grouped Humming with sequence parallelism

V64 combines V63's compact grouped expert execution with V59's Hopper
sequence-parallel projections. The patch is a private-check candidate.
There is no GPU correctness or performance result for this combination.

- SGLang base: `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Patch SHA256: `f6aaf93ce06ccfce9461e06bc3f71dea05dd901c01497c8be31bd2c515da689a`
- Size: 43,799 bytes across nine Python files
- Flags: `moe-runner-backend=humming`,
  `speculative-moe-runner-backend=marlin`, `enable-layernorm-sp`,
  `swa-full-tokens-ratio=0.25`
- Current scored incumbent: exact V44, `c3b7bd142c264853`, score 1.1847,
  with +1.8 GB peak memory per GPU and passing first/later-turn checks.

## Why combine these changes

Sequence parallelism removes the replicated large-prefill pooler and
unpooler work. The 62-layer architecture at 74,160 cold input tokens
requires about 617 TFLOP per GPU for these two projections when repeated
on every rank, versus about 77 TFLOP when distributed across TP8. These
are operation counts, not measured time shares.

The expert path sorts only the live rank-local routes into contiguous
rows and calls the installed Humming grouped GEMMs over the same packed
NVFP4 bank used for indexed decode. Its Hopper path can use TMA for
activation and weight loads. It avoids V44's per-expert dense
dequantization and GEMM loop. Both stages keep BF16 activation/output
boundaries and FP32 full-K accumulation; Stream-K is disabled.

At a balanced 16,384-token TP8/EP8 prefill, the two aliased temporary
arenas occupy 384 MiB instead of V53's 3,072 MiB full-route buffers.
The caller-owned combined output is another 128 MiB. Routing metadata,
allocator reservations, graph pools and KV sizing still affect the
measured peak. A dynamic arena-size guard retains indexed execution
when the compact layout would need more space than that path.

Each stage has a distinct purpose: projection sharding reduces repeated
matrix work; grouped execution improves expert scheduling and memory
movement. Neither implies an end-to-end multiplier by itself. V55/V56
and V59 projection savings overlap and must not be counted twice.

## Integration and validation

V63 changes three files relative to V53. V59's sequence-parallel delta
changes three different files. V64 applies those disjoint deltas and
verifies every resulting file against its exact parent. No manual merge
or numerical change was needed. The [validation receipt](validation.json)
records the source hashes, patch limits, forward/reverse application,
mode preservation and Python compilation checks.

The preserved `cpu_integration.py PATCHED_SGLANG_ROOT` harness executed
the actual SP entry methods and compact grouped adapter for 54 cases,
including TP4/TP8, padded tails, balanced routes and empty ranks. It
checked 324 pre-expert norm calls, one per rank invocation, and matched
the original-slot reference exactly with modeled BF16 arithmetic.
Independent review traced the pinned Humming API, tensor-map formats,
expert tails, compilation cache and BCG/collective ordering. No CUDA
kernel or distributed collective was executed by these CPU checks.

The SP path gathers pooled BF16 rows and routing metadata in original
token order before calling the experts. Grouped execution therefore sees
the global token count used by its dispatch threshold. The mapped
finalizer combines original top-k slots before the existing collective
returns each rank's token shard. Decode retains V53's indexed path.
The grouped eager graph break writes a fixed output allocation; route
counts, compact tensors and their aliases belong to that invocation.

GPU gates remain Humming import and compilation on the exact judge
image, graph capture/replay with live and empty expert routes, TP/EP
collective behavior, all numerical and later-turn checks, memory at or
below +2 GB, and useful throughput at the stock interactivity point.
CPU tests cannot establish those results.

## Relation to the 2x and 3x targets

Starting from measured score 1.1847, reaching 2 requires another 1.688x
speedup (40.765% less normalized total time); reaching 3 requires 2.532x
(60.51% less time). V64 is the highest-upside prefill candidate in this
iteration. V62 separately addresses target verification attention.

For illustration only, if prefill were 70% of V44's total time, a 2x
prefill improvement plus a 1.3x improvement in the remaining time would
produce `1.1847 / (0.7/2 + 0.3/1.3) = 2.04`. Those phase shares and
speedups are assumptions. A 3x score needs substantially more work saved;
component benchmark ratios cannot be multiplied into a serving score.

The inspected composition scratch held eight files totaling 6,169,154
bytes, mostly isolated Git indexes. It was removed after checking open
files and preserving the reproducible CPU harness and validation receipt.
Observed physical free space increased by 6,144,000 bytes; the volume is
shared with other activity.
