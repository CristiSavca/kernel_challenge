# V63 compact grouped Humming prefill

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- V53 parent `341c37c4d6e8cd7011218597de42870023fe2a33a046c6e2c7249d123107cb48`
- Patch SHA-256 `f5c5a2d1c18c522d584ec5b19dc39e607bbc47f4d10225f9fa4213407002c5f5`
- Patch size 33,248 bytes across seven Python files
- Check flags `--flag moe-runner-backend=humming --flag speculative-moe-runner-backend=marlin --flag swa-full-tokens-ratio=0.25`

V63 adds a grouped Hopper prefill path to V53. It prepares one packed NVFP4
bank and reuses the original two Humming layer descriptors. At 256 expected
routes per expert, eligible eager or breakable-graph forwards sort local IDs,
read the first local row index once, then gather only the live BF16 rows. Two
Humming grouped-contiguous WGMMA calls consume device-side expert offsets.
TMA reads activations and the packed weights. Existing vector stores mask out
rows beyond each expert's true end. The BF16 boundaries remain between GEMM,
global scaling, SiLU and per-expert RMSNorm. The mapped finalizer from V50
accumulates routes in original slot order. Small forwards and full CUDA graph
captures use V53's indexed path.

Two reusable arenas alternate input and activated rows, and gate/up and down
outputs. Balanced EP8 with 16,384 tokens, top-k 8 and 4,096 expert width uses
384 MiB of those arenas per rank. V53's full-slot arenas use 3 GiB. The weight
bank is unchanged. Unusually dense local routing falls back to indexed if the
grouped arenas would exceed indexed capacity. It uses the same packed bank
and writes the same preallocated graph output. The memory bound applies to
arenas. Tensor allocator reserve, the route metadata, sorting scratch and
graph storage still contribute to measured peak. An all-remote batch writes
zeros without launching grouped GEMMs.

The graph-owned output is allocated before the eager break. The break reads
live route counts and fills that output before later graph segments consume
it. Both grouped cubins are prepared as part of ordinary model weight
initialization irrespective of warmup routing. Preparation and dispatch use
the same serialized Humming config and CUDA context. The configuration
explicitly disables Stream-K BF16 partials and selects FP32 accumulation.
The default SM90 large tier is 176×256×64 for both target projections. The
pinned package's layout estimator returns 128,000 bytes of shared memory per
CTA for this configuration. The normal TP scheduler executes weight
postprocessing on its process main thread, where Humming compiles and
registers the cubin before returning its kernel ID.

## Local verification

Run `python cpu_check.py PATCHED_SGLANG_ROOT` with NumPy installed to repeat
the adapter, entry gate and scheduler tile probes.

- The patch applied and reversed on the exact pinned base in a separate Git
  index. All seven Python files compiled and retained their source modes.
  The incremental diff against V53 touches three files. Maximum added line
  length is 89 bytes.
- An exact-source CPU harness compared the adapter and eager finalizer call
  against a per-route BF16 oracle. It passed 116 grouped, 57 indexed fallback
  and 67 empty cases, including ragged expert counts, interior empty experts,
  differing hidden and intermediate widths, original slot order and both
  arena aliases. Forty-eight extracted dispatcher cases covered the size
  gate, compile, dtype and eager/full/breakable graph conditions.
- Modeled 42,899 grouped scheduler tile coordinates from the pinned Humming
  scheduler with expert gaps, partial tiles and several persistent grid
  widths. Each expert's last write stops at its exact end. Executing the
  pinned Python launcher wrappers with a mocked cubin builder confirmed the
  exact precompile key is reused by forward. CPU probes model the GPU kernels.
- Reviewed the SHA-256-verified Humming 0.1.12 weight transform, launcher,
  scheduler and TMA loader. Packing depends on the unchanged layer descriptor.
  The grouped input begins at a full BF16 row, so arbitrary expert offsets
  retain the required address alignment. The grouped output tile masks rows
  after the actual expert boundary. [Humming source release](https://pypi.org/project/humming-kernels/0.1.12/)

## H200 gate

Private-check candidate. Compare against V53 to isolate grouped experts and
against V44 for the full serving impact. Require H200 compilation, all-rank
startup, prefill and later-turn correctness, peak memory, TTFT and throughput.
Check both ragged and skewed routes under graph replay. The startup check
also covers the writable Humming cache and cubin registration. Route sync,
gather, scaling, tile utilization and graph break cost can outweigh the TMA
and workspace benefits.

At balanced 16K, the two expert GEMMs perform 1.65 TFLOP per layer per GPU.
H200's dense BF16 peak gives a 1.67 ms arithmetic floor. V44 measured about
5.8 ms for the corresponding dense expert path. A hypothetical grouped
2.5 ms would remove about 205 ms across 62 layers, roughly 19% of a prior
1.1 s full prefill chunk. Removing the whole measured expert stage would
save about 33% of such a chunk. The actual grouped kernel time, TTFT and
end-to-end score are unmeasured.

## Scratch cleanup

After independent review, the exact V63 scratch directory
`okc-v63-p9bh_8n5` was removed. Its source matched the frozen patch, its
index and open-file checks passed, and the CPU harness was preserved
alongside the patch. Available space increased by 4,988,928 bytes.
