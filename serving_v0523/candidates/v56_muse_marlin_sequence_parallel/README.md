# V56 Hopper Marlin sequence parallel prefill

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- V50 parent patch `7ba1708e049dbfae89eac613efe0bdeb9b7c8ccbcf0c11859ccad49f7f4441ae`
- Patch SHA-256 `bf9f5590adf2117483ef0a4fcb86e434ff779de334c232dbc78fd2d19bce8126`
- Patch size 38,529 bytes across eight allowed source files
- Check flags `--flag enable-layernorm-sp --flag swa-full-tokens-ratio=0.25`
  with the judge's explicit Marlin runner and default `none` MoE A2A

The existing LayerNorm SP pipeline keeps residual and normalization rows
sequence-sharded between layers. V56 adds a Hopper Marlin participant to it.
Query, key and value projections gather normalized rows on the original
attention TP group. Attention still computes its original per-rank heads over
the full visible context using the current FP8 KV backend. The attention output
projects and reduce-scatters back to its rank's token rows.

Each rank projects, normalizes and routes its local MoE source rows. The
sequence-order all-gathers restore the BF16 pooled activations, routing scores
and original IDs before the existing V50 local expert path. It still computes
every routed expert and applies the per-expert norm and weighted sum locally.
The ordinary NCCL BF16 TP SUM changes from all-reduce to reduce-scatter.
When the incumbent would use a custom or accelerated sum, V56 calls the
original group all-reduce on the unpadded tensor before taking its row shard.
This preserves its FP32 accumulation where used. Symmetric memory, NVLS and
compiled graphs also retain the original dispatcher. The attention output
projection GEMM uses the original unpadded shape.

The unpool projection consumes the result on its token shard. The existing
output boundary gathers the normalized rows and packed DFLASH aux states.
Padding is trimmed before per-expert execution or logits and restored with
zeros where the collective needs equal counts. Existing graph-aware SGLang
collectives own ordering, stream selection and capture. Small prefill chunks
also use SP when enabled. Decode and target verification retain their regular
full-row paths.

The opt-in requires Hopper, TP4/EP4 or TP8/EP8, ModelOpt NVFP4, explicit Marlin,
no external MoE A2A, no DP/CP/PP, and no mixed-chunk, already-scattered attention
or deterministic inference mode. The existing FlashInfer/MegaMoE and other
LayerNorm SP setup paths retain their original collectives.

## Reason for the change

The target has 128 query heads, 16 KV heads and head dimension 64. TP8 already
assigns 16 distinct query heads and two distinct KV heads per rank. At 74,160
tokens, the 16 full-attention layers account for 180 TFLOP per GPU and the 46
sliding-window layers for another 28 TFLOP, counting QK and PV. The Hopper
pooler and unpooler perform about 617 TFLOP per GPU because both run once per
rank on the full token rows. V55 and V56 both shard those two projections.
[Model geometry](https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/meta-models/Muse-Spark-open-816B-A42B)

V56's incremental change over V55 replaces full-row sums and separate pooler
and unpooler gathers with paired all-gather and reduce-scatter. Estimated
ring-equivalent bytes sent per rank per layer at 16,384 tokens with the
default all-reduce cap and ordinary NCCL:

| Activation collective | V55 | V56 |
| --- | ---: | ---: |
| Attention output/input transition | 448 MiB | 448 MiB |
| Expert output/input transition | 224 MiB | 224 MiB |
| Additional projection output all-gathers | 336 MiB | 0 |
| Routing IDs and FP32 scores | 0 | ~0.9 MiB |
| Total | 1,008 MiB | ~673 MiB |

The same design shards seven-eighths of the replicated residual, norm and
router work. The 335 MiB per-layer traffic difference corresponds to about
48 ms per full chunk at ideal unidirectional link rate. Actual timing depends
on NCCL algorithms, graph segment overhead, activation traffic and the smaller
GEMM shapes. Early engineering estimate is 5–12% first-prefill improvement
over V55 and 3–7% whole-score improvement if collective and graph behavior
are favorable. These are estimates pending matched judge results. [H200 bandwidth](https://www.nvidia.com/en-us/data-center/h200/)

## Verification

The official CLI exported the patch. Exact pinned-base cached application,
whitespace, file modes, allowed paths and seven Python compiles passed. Maximum
added line length is 110 bytes.

A CPU differential using the exact expert-shard helper and existing gather
helper matched the original route order and the modeled BF16 EP SUM in 48
TP4/TP8 cases. Cases covered 1 through 16,384 rows, ragged tails, all-local,
distributed and all-dropped routing. An extracted launch validator admitted
Hopper Marlin with DFLASH and rejected 16 incompatible configurations. AST
comparisons confirmed the existing attention, TP reduction/unpool, model entry
and DFLASH/logits exit methods were unchanged. A separate 56-case backend and
rank probe covered selection of the original unpadded sum, including BF16
cancellation that would differ with lower-precision accumulation. Sixteen
attention projection and rank cases verified the original GEMM shape across
custom all-reduce and standard NCCL. Source review traced the static capture
bucket and stable V50 eager-break buffers. CUDA collectives, graph startup,
real numerical agreement, memory and speed require H200 validation.

Use a private 16-user check before considering a scored run. Compare peak
GPU memory, startup, TTFT and first/later-turn correctness carefully with V50
and V55. V56 uses a different BF16 reduction tree and GEMM shape. The CPU
differential models the logical SUM and does not establish GPU equality.

## Independent source review

Reviewed the frozen patch for a private check on the judge's homogeneous
8×H200 configuration. The shared launch configuration selects prefill SP on
every rank. Sequence entry gives all ranks the same padded row count. The
pooled activations, FP32 routing scores and expert IDs gather in the same TP
rank order. On an odd eager tail, gathers remove SP padding before local
expert execution, then the reduction returns equal-size shards. A rank whose
shard contains only padding still participates in every collective.

Breakable capture reads the SP token count when it builds the graph at its
static bucket size. V50's dense eager break receives tensors with those
captured shapes and uses their metadata directly. The existing eager logits
tail gathers the DFLASH auxiliary states in sequence order using the input
extent and the prefill runner trims the graph bucket to the raw token count
before draft KV materialization. Decode and verify use the existing non-SP
paths.

The precision guard checks the incumbent collective on the original unpadded
GEMM output. It preserves that group's all-reduce when the original path can
accumulate in FP32, and keeps the compiler's opaque dispatcher intact. The
ordinary NCCL BF16 sum uses reduce-scatter, where a different reduction order
can still change rounding. The ~335 MiB estimate is the large-chunk
communication difference from V55, which already shards the projections.
Real CUDA graph startup and replay, cross-rank output agreement, peak memory,
and realized throughput remain unmeasured.

After checking the frozen patch, temporary Git state and open files, the
V56 scratch directory `okc-v56.LhstomCz` was retired. It occupied 3324 KiB.
Available space on the shared volume changed from 35102944 KiB to 35106260 KiB
(an observed difference of 3316 KiB).
