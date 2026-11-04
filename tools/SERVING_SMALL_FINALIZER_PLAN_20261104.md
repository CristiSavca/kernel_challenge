# Serving small-batch expert finalization — bounded implementation plan

2026-11-04. **A concrete portable serving opportunity is present.** This is a
read-only proposal; no SGLang source, existing patch, model weights or GPU state
was changed. Root approval and independent source/numerical review are required
before implementation/testing. It does not claim an end-to-end speedup.

## Actual selected path

Pinned checkout: `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`.
Judge configuration selects `--moe-runner-backend marlin` on8×H200. The real
call chain is `MuseSpark1xMoE._forward_experts` in
`python/sglang/srt/models/muse_spark.py:899` →
`MuseSpark1xFp4Experts.forward` in
`python/sglang/srt/layers/muse_spark_fp4_moe.py:697` → `_forward_marlin:576` →
`fused_post_expert_norm_weighted_sum:603`. Marlin exits before the separate
TRT-LLM/Helion finalizer. Its outputs are per-route BF16 tensors, not the native
standalone experiment's private layout. The fused all-reduce workspace requires
Blackwell at `models/muse_spark.py:1178`, so it does not replace this H200 path.
The existing outer all-reduce and unpool at750 remain unchanged.

The finalizer file is
`python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py`,
SHA `9b816e971817f96a22d2d09424dc8dfa6f6020e18de73fb895674c76494fe591`.
Lines77–102 take an eager Torch path for every nonempty batch below32 tokens:
mask remote slots, cast toFP32, multiply the norm weight, square, mean, add
epsilon, rsqrt, multiply normalization, multiply route score, sum slots and
cast to output dtype. CUDA graph capture removes host launch overhead but does
not fuse those separate tensor operations. Actual device-node count remains
to be profiled. The existing one-CTA-per-token fused kernel at10–60 is only
selected for32 or more tokens; merely enabling it below32 is a useful local
comparison, not evidence it wins with very few CTAs.

This scope occurs at low-concurrency decode and verification, depending on the
actual batch/capture padding. It must not dispatch on request concurrency,
benchmark names, draft assumptions or an enumerated kernel-test shape list.
The BF16 expert implementation also calls this helper, so tests must cover the
helper contract rather than assume every caller is NVFP4 Marlin.

## Fixed proposed replacement

Add two small Triton kernels in the same finalizer file, keeping its public API.
Use the existing small-batch branch for CUDA BF16, `0<T<32`, `1<=TOPK<=32`, and
`0<H<=16384`; preserve the old path outside this bounded generic kernel domain.
These are implementation resource bounds, not routes or request shortcuts.
T0 returns the existing empty output before any kernel. The current T>=32
kernel and the existing1549-byte serving patch are preserved.

1. **Per-route normalization scalar.** Grid `(T,TOPK)`, oneCTA per route;
   `BLOCK_H=next_power_of_2(H)`, existing warp policy8 atH>=4096 else4. Inspect
   `slot_expert_ids` when supplied; for negative IDs mask the expert-output
   load to positive zero, matching the existing Torch `where`, and do not read
   uninitialized expert output. Convert BF16 values toFP32. If weight-first,
   form `f=x*w` inFP32 before squared-sum;
   otherwise reduce `x*x`. Store `rsqrt(sum/H+eps)` in FP32 scratch `[T,TOPK]`.
   For `APPLY_NORM=False`, write1 without reading the expert vector.
2. **Tiled weighted combination.** Grid `(T,ceil(H/256))`, BLOCK256/four warps.
   Iterate all TOPK slots in the original order, replacing negative-ID input
   elements with positive zero before arithmetic. Keep every valid route.
   Recreate the
   original elementwise operations: weight-first `f=x*w; f=f*inverse`,
   weight-after `f=x*inverse; f=f*w`, no-weight `f=x*inverse`, or plainx when
   normalization is disabled. Then `f=f*route_score`, accumulate inFP32, and
   perform exactly one final BF16 cast. Do not pre-multiply inverse and score,
   round normalized route outputs toBF16, or move normalization after slot sum.
   Do not skip the zero-slot arithmetic: zero times a nonfinite weight/score,
   or zero normalized with zero epsilon, can produce NaN in the original small
   Torch branch. Masking only the input load preserves that contract without
   reading invalid memory. This differs deliberately from the old large-token
   kernel's skip-slot implementation.

Use ordinary per-call Torch scratch/output allocation on the caller stream,
with no persistent module scratch, counters, cross-stream events or host reads
of values. Scratch is `4*T*TOPK` bytes (256B atT8/TOPK8; at most3968B within the
proposed domain). Kernel stream ordering publishes the scalar buffer to the
combiner. Separate invocations and graph captures own independent buffers;
no new rank or communication state is introduced. Support absent weights,
both norm forms, optional IDs, optional normalization, noncontiguous caller
inputs via existing/explicit contiguous copies, and arbitrary TOPK within the
resource bound. The effective norm weight remains the existing cache at
`muse_spark_fp4_moe.py:231–243`, which includes its gain and BF16 conversion.

## Work removed and countercost

For weight-first BF16 input with remote masking, the existing small branch has
approximately `50*T*TOPK*H` bytes of main activation tensor reads/writes from
its explicit operations, excluding weight/scalar accesses and final output.
That is12.5MiB atT8/TOPK8/H4096. This is source-level requested traffic assuming
separate eager operations, not observed DRAM bytes or a latency estimate.

The proposed two passes read each live BF16 route vector twice, approximately
`4*T*TOPK*H` bytes plus output and small scalar traffic, and avoid full-sized
FP32 normalized/weighted intermediates. EP masking may reduce actual live reads
further; no distribution is assumed for correctness. It replaces the tensor
operation chain with two launches. The norm-weight multiply and BF16→FP32
conversion occur in both passes, and norm weights are read twice. There is an
extra launch relative to the existing large-batch fused kernel. These costs,
small-grid utilization and graph launch latency need an actual comparison.
No packed expert bank, model-loader change, new collective or model download
is required to validate this component.

## Numerical, concurrency and validation gates

All norm/reduction/route arithmetic staysFP32 and the BF16 boundaries stay in
their current positions. Disable FP fusion for the new kernels initially so
separate Torch multiply and addition operations are not silently replaced by
FMA. Triton and Torch reduction trees/rsqrt lowering can still differ: do not
claim bitwise output identity from matching formulas. Report absolute/relative
error and BF16 differences against the current small Torch path and the
existing fused path; use the repository's applicable correctness tolerances
and the official serving output check as the final gate.

Focused GPU tests can use synthetic expert outputs without loading the model:
T0/1/2/7/8/15/16/31/32/33, TOPK1/2/3/8/16/32 plus fallback33, hidden sizes
including ragged511/1025 and common4096/8192, both norm forms, absent weight,
APPLY_NORMFalse, several epsilons, all-local/all-remote/mixed/one-live IDs,
noncontiguous inputs, cancellation and BF16 boundary values. Poison remote
slots withNaN to prove they are not consumed; include nonfinite weights/scores
and zero epsilon to verify the original zero-slot arithmetic rather than
silently assuming finite metadata. Exercise two streams and independently
captured/replayed graphs with changing tensor contents and IDs. EP1/EP8 and
rank-offset integration fixtures must preserve local mapping and the outer
all-reduce; the new finalizer itself only tests IDs>=0 and makes no rank guess.

Root's selected first GPU validation uses a separate diagnostic candidate
derived from the unchanged official `muse_moe_w4a16/baseline.py`: embed the new
helper and redefine only that candidate's imported global finalizer binding,
without monkeypatching a loaded module. Exact restoration must show all other
Marlin math/packing/routing unchanged. One official private check then exercises
the actual five tests/six benchmarks against the stock baseline. It covers
BF16/H4096/TOPK8/Marlin and the selected small batches, not the generic cases or
concurrency above; it is not a competitive source promotion. A later separately
authorized synthetic paired test can compare stock Torch, existing one-kernel
finalization and this two-stage method under graphs with generic inputs.
Record actual kernel names/count, allocation behavior and per-shape results.
Reject on any numerical/race failure; choose dispatch only from a stable broad
small-batch advantage and preserve fallback elsewhere. This component gate
does not replace a full serving check after November10, including TTFT,
220-token output agreement, concurrency1/4/16/32 and prefill controls.

## Rules and upstream scope

The latest locally saved official serving response is the14:11UTC record in
`20261104-serving-current-readonly.json`: source patch under`python/sglang/`,
excluding model_loader, <=409600 bytes, no lower precision, all routes/norms
retained. This proposal fits that boundary. The public rules webpage opened
but exposed no readable text in this review; API/llms web retrieval failed,
so no newer schedule/limit is asserted here.

The freshly read [SGLang roadmap](https://github.com/sgl-project/sglang/issues/44678)
and [vLLM tracker](https://github.com/vllm-project/vllm/issues/64632) discuss
low-concurrency kernels, routing and Blackwell MegaMoE, but do not establish
that this exact pinned H200 small-batch finalizer has been replaced. The
[day-0 article](https://www.lmsys.org/blog/2026-10-15-meta-muse-spark-open/)
confirms per-expert normalization precedes weighted combination; its published
Blackwell gains are not evidence for this proposal. The attempted main-branch
raw finalizer URL returned404, so no current upstream implementation is claimed.
