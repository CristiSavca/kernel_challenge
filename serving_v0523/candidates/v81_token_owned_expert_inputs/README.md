# V81 token-owned routed expert inputs

V81 lets each token's GPU thread block normalize its input once and write it to every local routed expert. This fuses V80's standalone pre-expert RMSNorm and expert-centric staging into one launch. It applies to eligible grouped Humming BF16 decode and verification. The input and output projections and both packed expert GEMMs retain V80's implementations.

For a token, two histograms give the total number of slots for each expert and the number in previous tokens. Including earlier equal slots in the token assigns a unique destination even with duplicate routes. All blocks independently compute the same prefix offsets. Only block zero publishes them. Destination order is unchanged across requests, and a token that has no local experts contributes exclusively through its owning ranks. The input has already been pooled before this step.

Every participating token uses its complete input, norm epsilon and effective gain. The sum of squares, reciprocal square root and products use FP32, with one BF16 cast before the grouped GEMM. PyTorch's GPU reduction can order the additions differently. Unsupported norms, dtypes, shapes, disabled features and forward hooks retain the module call. [PyTorch 2.14.1 CUDA RMSNorm](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu)

## Cost

The removed full normalized intermediate is 2 MiB/rank at 256 target rows of width 4096. Under uniform top-eight routing without replacement across 256 experts and TP8/EP8, the expected share of tokens with any local route is 66.18%. At 256 rows, the expected reduction in logical hidden-state traffic is 4.676 MiB/layer/rank, or about 290 MiB across 62 eligible layers. This counts source reads, normalized writes and normalized reads. It excludes gamma, routing metadata and cache effects. Packed weight traffic stays the same.

The new launch uses one block per token. At 256 rows V77/V80 staging had 512 blocks scanning 2,048 route slots each; V81 has 256 blocks scanning those slots. Each block now runs two histograms and a complete RMS reduction on live tokens. At eight rows, only eight blocks can occupy the GPU during this phase. Histogram pressure, skew, and occupancy can outweigh the saved launch and memory traffic. No H200 performance result exists for V81.

## Reproduction

Use V80's seven launch options and the native eight-row verification width. The new environment switch `SGLANG_ENABLE_MUSE_SPARK_HUMMING_ROUTED_NORM=0` selects the incumbent staging and norm under this patch. The pinned SGLang revision is `297e54fd105e974cdf401e5b4d20253a22c16f90`. The frozen [patch](patch.diff) SHA-256 is `b3662cea54b86f0f4b463adcf430473184be50f82e2ff2b9467b28e717671966`. Three source sections differ from V80 in the [additive delta](delta.diff).

On an isolated patched checkout with NumPy installed, run `python3 cpu_check.py /path/to/sglang`. The actual kernel body passed 588 adversarial cases and 18,442,925 BF16 output-value comparisons, including the full width-4096 input and 2,048-slot limit. The suite also checked 16 dispatch decisions. V77's fallback suite again passed its 192 staging and 48 finalize cases. The official local export, application checks, path validation and compilation of all Python sources passed. [CPU result](cpu_result.json) and [export receipt](validation.json).

H200 compilation, agreement, graph replay, peak memory and speed remain open. Compare V81 against V80 at the same load and flags before promotion.
