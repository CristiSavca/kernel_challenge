# V59 Hopper Humming sequence parallel prefill

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- V53 parent `341c37c4d6e8cd7011218597de42870023fe2a33a046c6e2c7249d123107cb48`
- V56 sequence-parallel source `bf9f5590adf2117483ef0a4fcb86e434ff779de334c232dbc78fd2d19bce8126`
- Patch SHA-256 `ea60c518cb47f98a8a740757f055398c816d1c30aff4573252f53a57c13f0773`
- Patch size 32,952 bytes across eight Python files
- Check flags `--flag moe-runner-backend=humming --flag speculative-moe-runner-backend=marlin --flag enable-layernorm-sp --flag swa-full-tokens-ratio=0.25`

V59 preserves the V53 indexed Humming expert, ModelOpt loading, packed
expert bank and per-expert finalizer byte for byte. It adds V56's sequence
parallel prefill path for explicit Hopper Humming. The launch gate also
retains V56's explicit Hopper Marlin mode with the V53 parent’s Marlin
packing path. Each launch prepares one expert layout. Marlin can be selected
at startup and uses its own packed weights. A Humming launch has no runtime
fallback to Marlin and carries no compact dense Marlin implementation.

The existing SP pipeline shards the FP32 residual and the pooled projection,
unpool projection, norm and routing rows. It gathers pooled BF16 activations
and original top-k IDs and scores in token order before executing every local
expert. It returns equal token shards via the same V56 NCCL BF16
reduce-scatter helper. That helper uses the original unpadded group all-reduce
whenever the incumbent would select an accelerated sum that might accumulate
in FP32. The attention heads and KV precision are unchanged. Humming retains
BF16 inputs, FP32 accumulation without Stream-K, and BF16 scale rounding before
each nonlinear consumer. Decode and target verify use V53's non-SP layout.

The configured draft MoE flag stays `marlin`. The pinned DFLASH2 draft
constructs dense MLP layers and does not execute either routed backend.
Existing target DFLASH auxiliary taps are packed and gathered in original
token order. No change was made to the draft model, draft decode, or the
target's speculative-verification computation.

## Expected effect and overlap

At 74,160 input tokens in TP8, V53's replicated pooler and unpooler perform
about 617 TFLOP per GPU. V59 reduces that to about 77 TFLOP per GPU while
sharding residual, norm and router work. At full 16K chunks with normal NCCL,
the main activation collectives move about the same ring-equivalent volume as
V53, plus around 0.9 MiB per rank per layer for routing tensors. The ~335 MiB
traffic saving quoted for V56 used V55's extra projection gathers as its
comparison. It does not apply against V53.

V59 keeps V53's roughly 3 GiB of full-slot indexed GEMM buffers per rank at
16K. V56 uses compact rows in that regime, so V59 gives up that memory
benefit. A standalone Humming gain, a projection-sharding gain and V56's
communication gain cannot all be counted independently. Peak allocation, graph
storage and score for this composition have no GPU measurement.

## Verification

The official CLI exported the patch. Exact pinned-base cached application and
reverse check, original file modes, allowed paths, whitespace and all eight
Python compiles passed. Maximum added line length is 98 bytes. Only the three
SP modules differ from the staged V53 source.

A CPU test ran the exact V59 SP wrappers, V53 expert forward dispatcher and
Humming indexed adapter using modeled device kernels and collective operations.
It matched the explicit per-expert oracle and rank-order SUM in 84 TP4/TP8
cases across both initialized banks, covering all-dropped, uneven and
single-owner routes, odd tails and 16K rows. Both legal Hopper configurations
were accepted and 30 invalid ones were rejected. The tests exercised both
ordinary and accelerated reduction dispatch, excluded cross-backend bank
access, and confirmed SP only activates for ordinary target extend. AST
comparisons confirmed the target model entry, logits and DFLASH exit, and
non-SP TP reduction/unpool method bodies matched V53.

Use a private H200 check after the parent evidence establishes a useful
comparison. Inspect the exact Humming package and writable JIT paths, all-rank
initialization, graph replay with live and all-remote routes, correctness,
overall memory peak and both TTFT and throughput. Compare with V53 to isolate
SP and V56 to compare expert choices under SP.

## Independent source review

The scheduler initializes sequence parallelism from the target configuration
before draft construction. The target's exact `EXTEND` predicate gates SP.
The existing eager logits boundary gathers DFLASH auxiliary rows, and the
prefill graph runner trims the bucket to the raw token count before draft KV
materialization. For an odd eager tail, the SP gathers trim activation and
route padding before the Humming call. Expert normalization runs once and
global IDs map to local IDs only after the gather.

The checksum-verified Humming 0.1.12 launcher uses `valid_shape_m` to select a
kernel and launches on the current PyTorch CUDA stream. The indexed device
scheduler reads `num_tokens_padded` on each execution, so a skewed or empty
local route count is independent of the host tuning estimate. The indexed
launcher allocates no tensor-map workspace. Graph capture and replay with
actual route distributions still require the H200 check.

Humming 0.1.12 defaults to `~/.humming/cache/` and `~/.humming/tmp/`, with
`HUMMING_CACHE_DIR` and `HUMMING_TMP_DIR` as overrides. Judge-image write
access, compilation and cubin registration at those paths are still
unverified. [Humming 0.1.12 source release](https://pypi.org/project/humming-kernels/0.1.12/)

After checking the frozen patch, temporary Git state and open files, the
V59 scratch directory `okc-v59.QCa2ncza` was retired. It occupied 3564 KiB.
Available space on the shared volume changed from 33698832 KiB to 33702384 KiB
(an observed difference of 3552 KiB).
