# V77 fused expert staging

V77 replaces the dependent plan and input-gather kernels in V76's grouped NVFP4 decode path with one Triton kernel. Each block owns one local expert and up to 128 original route slots. It computes where its routes belong in Humming's contiguous, expert-ordered suffix, then copies the corresponding live BF16 input rows directly into that segment. The two grouped GEMMs, post-scale rounding, SiLU, per-expert RMSNorm and original-order weighted sum use the V76 code.

For each tile, the first output row is the slot capacity minus the number of routes assigned to later local experts and the number of this expert's routes at or after the tile start. A prefix sum of equal-expert slots within the tile preserves original slot order. Invalid or remote IDs receive inverse-map value −1. The first tile writes its expert's start, and expert zero also writes the final end. Every live row has a unique producer. The remote prefix of the row-key and activation arenas may retain old values on replay. The grouped GEMM offsets and scaled SiLU both exclude this prefix.

## Work and costs

The grouped expert path drops from six GPU dispatches to five per eligible layer. Across 62 eligible routed layers this removes 62 serial dispatch boundaries per target verification pass and the sorting network over up to 2,048 route slots per layer. Activation staging and packed-weight traffic follow the same layout. The fused kernel uses the existing buffers. Its register usage, local-memory spills and CUDA graph pool usage still require device inspection.

The independent blocks read route metadata redundantly. At 2,048 slots and 32 local experts, 512 blocks examine 1,048,576 IDs, or 4 MiB of logical loads for int32 IDs before mapped-expert lookups. The unique ID array is 8 KiB. Cache behavior determines the physical traffic. Concentrated routing can leave a small number of blocks copying most of the inputs. Either cost may exceed the saved dispatch latency.

## Reproduction and validation

Use the six [V76 launch flags](../v76_sharded_draft_target_decode/README.md). The fused path defaults on within the existing grouped-decode eligibility guard with hidden width at most 8,192. The larger-width path uses the prior implementation. Local A/B testing can set `SGLANG_ENABLE_MUSE_SPARK_HUMMING_FUSED_STAGE=0` to select the old planner and gather. The candidate introduces no judge-flag changes.

- Base SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Parent V76 `199c3bd308a30addd45b798208a4f1573c28d0b5deb470df0de63b171b3a05ac`
- [Official patch](patch.diff) SHA-256 `05426d8be9e013fa73f24f10a72958a3ca4b74b942a770756f4b52126931651a`, 147,143 bytes, 28 files
- [Two-file delta](delta.diff) SHA-256 `050eb600db7de9f0915a47f812e90004c7b67ab27c2192384f0af77e8b175ef4`, 6,807 bytes

Run `python cpu_check.py PATH_TO_PATCHED_SGLANG` with NumPy. It executes the source through the established V66 tensor and Humming emulators. It passed 192 staging cases, 48 grouped-and-finalizer comparisons and 55,733 rounded route values. Cases cover arbitrary mapped IDs, out-of-range int64 IDs, remote-only batches, concentrated and mixed routes, partial tiles and capped sizes, shuffled block execution, NaN-poisoned unused floating rows and untouched integer sentinels. Official export and forward/reverse apply checks passed.

Compare V77 with V76 at matched load. Confirm target verification throughput, exact output length and agreement, first and later TTFT, peak GPU memory and CUDA graph replay before selecting this path. V77 has no H200 measurement or full-serving speed estimate.
