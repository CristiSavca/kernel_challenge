# Hugging Face and CUTLASS source scout — November 4, 2026

The Hub is useful for discovering actual code and hardware requirements. A recent
upload does not establish a newly invented kernel or an improvement over our
selected implementation. No dependencies, external binaries or weights were
installed. Compact source hashes and selected lines are retained in
`experiments/research/20261104-hf-cutlass-source.json`.

## Hugging Face: source and precision filters

Read the November 4 head of
[huggingface/kernels-community](https://github.com/huggingface/kernels-community/tree/f923c618f29b019080f4b02ca2be5bc35fd10c51).
That head's commit concerns the TPU build workflow; it is not the date of every
kernel implementation. Four relevant packages were inspected:

| Package | Actual implementation and license | Decision for Muse on H200 |
|---|---|---|
| [nvfp4-gemm](https://github.com/huggingface/kernels-community/blob/f923c618f29b019080f4b02ca2be5bc35fd10c51/nvfp4-gemm/build.toml) | Apache-2.0; build targets SM100a/SM120a. `functional.py` quantizes activations to FP4 before scaled GEMM. | Incompatible with H200 and required BF16 activations. The NVFP4 name alone is insufficient. |
| [finegrained-fp8](https://github.com/huggingface/kernels-community/blob/f923c618f29b019080f4b02ca2be5bc35fd10c51/finegrained-fp8/README.md) | Apache-2.0; some SM90 kernels accept BF16 tensors but quantize them internally to FP8. FP4 path uses MXFP4 group32/UE8M0 and FP8 activations. | Input API dtype is not compute precision. Cannot replace our BF16×NVFP4 experts. Grouped routing ideas need a separately identified missing operation. |
| [flash-attn3](https://github.com/huggingface/kernels-community/blob/f923c618f29b019080f4b02ca2be5bc35fd10c51/flash-attn3/README.md) | BSD-3-Clause; packaged Dao-AILab FlashAttention, including SM90 builds. | Useful source/benchmark reference. A package upload is not a new algorithm. Must compare exact paged FP8/GQA/cast support and selected serving backend before any change. |
| [deep-gemm](https://github.com/huggingface/kernels-community/blob/f923c618f29b019080f4b02ca2be5bc35fd10c51/deep-gemm/README.md) | Package of DeepSeek's library; consulted alongside the fresh upstream scout. | Packaging does not resolve the upstream hardware/precision mismatches. Prefer upstream exact source for algorithm research. |

The official [Kernels update](https://huggingface.co/blog/revamped-kernels), July6,
explains architecture/backend metadata and the dedicated kernel repository type.
The old model-repository card for FlashAttention3 carries a September13 migration
notice. Use the current [kernel catalog](https://huggingface.co/kernels), then
inspect source at an exact revision. No binary import is needed for discovery.

Hugging Face's [attention profiling article](https://huggingface.co/blog/torch-attention-profile),
July10, demonstrates why the same high-level operation can select very different
GPU paths. Its examples use A100; its reported timings do not predict our H200
performance. The useful method is to verify the selected backend and actually
removed copies/launches. Our v22 profiles now confirm both new helper kernels.

## CUTLASS: newest release versus a fresh Hopper proposal

[CUTLASS4.8.1](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.1), October21,
is the latest release observed. Much recent extension work targets Blackwell or
Rubin. Release notes include compiler/descriptor correctness fixes, not a measured
H200 Muse speedup. Updating the entire toolchain is not justified by this scout.
The GitHub license classifier returns `NOASSERTION`; any future source reuse
must read the specific file and full applicable license instead of relying on
that classifier. No CUTLASS source was imported here.

The November3 [Hopper variable-length FMHA PR3819](https://github.com/NVIDIA/cutlass/pull/3819),
head `4b623b7567eac2ebd24274675d91a782521649f3`, is **open and unmerged**.
It adds independently ragged Q/K sequence handling, predicated output stores and
tail masks. The author reports passing GH200 tests and nearly unchanged regular
shapes; skewed lengths expose poor static persistent scheduling. FP8 key lengths
must be divisible by16, and its packed-sequence interface is not our paged cache.
These are contributor results, not our measurements or an accepted release.

Portable hypothesis: length-aware assignment can reduce imbalance where a
persistent attention grid currently spends work on maximum-length padding.
Before implementing, prove this exact waste exists in the selected serving
backend, preserve all keys, and include mixed sequence lengths and speculative
verification. **Kill** if current scheduling already avoids the waste, conversion
to packed KV adds a full cache copy, arbitrary FP8 lengths fail, or the proposal
requires changing precision. Estimated ceiling is at most the measured exposed
padding/imbalance fraction; that fraction is currently unknown, so no numerical
gain is forecast. This is lower priority than validating the measured finalizer.

## Concrete next work

1. Retain v22's measured stock-Marlin improvement and validate generic shapes,
   auxiliary dtypes/strides, exceptional masks, independent streams and graph
   replay. Official kernel coverage alone does not prove full serving safety.
2. On the official serving system, measure selected attention scheduling and the
   finalizer-to-all-reduce boundary at the required concurrency levels. Use these
   measurements to choose an isolated fusion or scheduling change.
3. Do not adopt new activation/cache quantization, a wholesale runtime upgrade,
   or a fresh search platform without a compatible, falsifiable experiment.
