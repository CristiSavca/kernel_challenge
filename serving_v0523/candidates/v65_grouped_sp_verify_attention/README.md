# V65: grouped sequence-parallel prefill and paged verification

V65 combines exact V64 with the attention-only delta from V62. It is a
frozen private-check candidate, with no GPU correctness or throughput result.

- Pinned SGLang: `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Patch SHA256: `80e41ae32e748857053f553359f4d555cbe983534cff465f4910551723c30337`
- Patch size: 62,373 bytes, eleven permitted Python files, longest added line 98 characters
- Flags: `moe-runner-backend=humming`, `speculative-moe-runner-backend=marlin`,
  `enable-layernorm-sp`, `swa-full-tokens-ratio=0.25`
- Measured serving incumbent: V44, score 1.1847, job `c3b7bd142c264853`

## Composition

V64 supplies one packed NVFP4 Humming weight bank, compact grouped execution
for large prefills, and sequence-parallel prefill projections. V62 supplies
only `paged_verify_attention.py` and its FA3 backend dispatch. Those two files
are disjoint from V64's nine files. The V44 dense expert fallback is not
imported with V62. Every resulting file was compared byte-for-byte with its
intended parent, and the patch applies cleanly to the pinned base.

The prefill changes reduce replicated projection work and improve grouped
expert execution. The attention change partitions the live causal or sliding
window KV tiles across the GPU during compatible DFLASH target verification.
It retains all visible keys, FP8 Q/K/V, FP32 scores and reductions, and uses
BF16 probability/value multiplication. Unsupported attention cases use FA3.
The two dispatches operate in different forward modes. Their speedups cannot
be multiplied to predict the serving score.

## Validation

Both parent CPU suites passed against the actual combined source:

- V64 integration: 54 cases and 324 prenorm calls, including TP4/TP8,
  padded tails, skewed routing, empty experts, and BF16 collective models.
- V62 attention: 110 numerical cases, maximum relative L2 0.001629 against
  the full-FP32 oracle, and 890 ownership plans through 131,072 context tokens.
- All eleven Python files compile. Patch bounds, paths, source parity,
  and disjoint composition pass.
- An independent [rules and source audit](rules_audit.md) found no source
  blocker against the live official rules and mutable launch flags.

Reproduce the CPU checks by applying the patch to an isolated checkout and
passing its root to `../v64_humming_grouped_sequence_parallel/cpu_integration.py`
and `../v62_paged_verify_attention/cpu_check.py` using `.venv-modal/bin/python`.
These model checks do not execute CUDA, Triton, Humming cubins or NCCL.

The H200 check must establish compilation and capture/replay, first/later-turn
agreement, token counts, peak memory no more than 2 GB above stock per GPU,
TTFT within twice stock at each load, and useful throughput. V64's balanced
384 MiB arenas and V62's approximately 5.2 MiB attention scratch do not prove
the combined allocator/graph peak. Changes in GEMM and reduction rounding
still require the judge's output checks.

## Queue

V65 has not been submitted alone or selected for a scored run. The initial
plan retained V64, V62 and V60. After the integrated decode work passed review,
the Nov 17 00:03 UTC private-check plan became V64, V62 and V68. V68 contains
V65 plus grouped decode and the DFLASH capacity repair. Existing queued jobs
and both component controls are unchanged. Neither 2x nor 3x has been measured.
