# V78 tensor-parallel DFLASH convolution filters

V78 shards the coefficient generator inside the draft's attention and MLP grouped convolutions. Every rank uses the full input width to compute a disjoint set of output channels. One draft-group all-gather restores the original coefficient order before the existing depthwise convolution consumes them. The filter coefficients depend on the current hidden state and are computed live on every call.

The reference DFlash2 checkpoint has five layers, two filters per layer, hidden width 8,192, two taps and 512 groups. Each bias-free coefficient generator therefore has a BF16 matrix of shape 2,048 × 8,192. At TP8 its persistent weight drops from 32 to 4 MiB per rank. Across all ten generators this saves **280 MiB per rank**. The estimated change in FLOPs and bytes per draft pass follows from those dimensions. [Draft model configuration](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/blob/main/config.json)

| Draft rows | Fewer projection FLOPs per rank | Received coefficient payload per rank |
| ---: | ---: | ---: |
| 8 | 2.35 billion | 0.273 MiB |
| 128 | 37.58 billion | 4.375 MiB |
| 256 | 75.16 billion | 8.75 MiB |

Every eligible pass incurs ten collectives. Actual communication, buffer layout copies, transient memory and GEMM efficiency determine the net effect. The saved weight bytes can be reallocated to KV at startup. A case where the draft accounts for only a small share of total time will show a correspondingly smaller serving gain.

## Frozen source and dispatch

- [Full patch](patch.diff) SHA-256 `f042bb2d29454098bf8290bb0567949edb32303c59e05f4a8543142cfc5192d5`, 145,959 bytes across 28 files.
- Parent [V76](../v76_sharded_draft_target_decode/README.md) SHA-256 `199c3bd308a30addd45b798208a4f1573c28d0b5deb470df0de63b171b3a05ac`. The [three-file delta](delta.diff) has SHA-256 `144b1c7560ee0fb042db6df7a24a474660d4e4234b3951e84a6630d7e47db450`.
- Use the six V76 launch flags plus `--flag speculative-dflash-tp-conv-projection` to isolate this change at the same load.
- The option applies to unquantized DFLASH drafts on CUDA when the output width divides the draft TP group and the group has multiple ranks. Other configurations use the original replicated projection. Weight loading checks the full checkpoint shape before selecting rank-owned rows.
- Inputs and outputs within a draft group share token order under DFLASH's single-group execution. Each coefficient retains its original full-K dot product and output dtype. The same hidden layer, base kernel, tap ordering, block boundaries and normalization run after gather.

## Local validation

The official CLI export reproduced the frozen patch. Applying the delta atop V76 reproduced the same bytes. All 25 changed Python files compiled, reverse applicability and whitespace passed, and the maximum added line has 106 characters. The file surface and seven launch flags were compared with the serving template.

The [CPU check](cpu_check.py) extracts the production constructors, loader and prepare/finish methods and executes them with array doubles. It checks 8 factory settings, 144 constructor cases, 726 checkpoint loads or rejections and 1,008 prepare/finish cases, including empty rows, noncontiguous input, disabled sharding, non-divisible filter widths and multiple TP sizes. The inherited context-projection suite passed 120 constructor, 336 projection/gather and 924 checkpoint/shape cases after binding the generalized class name. Run with NumPy installed and the V78 source applied.

The CPU cases prove rank ownership and coefficient assembly in the simulated environment. Changes in GPU GEMM tiling can affect rounding. H200 graph capture, collective progress, output agreement, exact lengths, TTFT, peak memory and the four-load speed remain unmeasured. Compare against V76 under matching settings before promotion.
