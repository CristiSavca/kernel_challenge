# V145 low-load committed-row projection

At one user, projecting only the accepted DFLASH rows reduces the mean matrix height from eight to four. **It eliminates zero weight bytes and zero 16-row matrix tiles.** Every verify pass commits at least the bonus, and one live row still needs every channel of the 640 MiB target-to-draft context matrix and 20 MiB stacked K/V matrix. V120b already reduces the former to an 80 MiB output shard per rank. We kept the incumbent one-user path. V145 produced a feasibility test and used no challenge quota. [Scored V64 rungs](../../runs/abd217baf0ed49fd.json) · [Source checks and calculations](check.json) · [V120b](../v120b_native_muse_prompt/README.md)

## What a shorter projection can save

The pinned native commit function was exercised on all 4,096 combinations of one through eight committed rows for four requests. On 73,728 live rows, packing the confirmed features, applying a full-K BF16 projection, row normalization and the stacked K/V projection produced the same outputs as applying those operations to the dense synthetic blocks and selecting the live results. A separate test advanced 400 permuted batches through 12,366 distinct committed positions. The next anchor begins at the next unprocessed position. A bonus predicted on the previous step has no materialized target hidden row to reuse. This CPU test establishes the row mapping. It leaves GPU GEMM rounding and kernel timing unmeasured. [Runnable test](check.py)

| Users | Dense → mean live rows | Dense 16-row tiles | Expected compact tiles from mean alone | Modeled compact tiles with independent geometric matches | V64 cold bandwidth/compute floor, dense → compact |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 → 4 | 1 | 1 | 1.00 | 144.2 → 144.2 µs |
| 4 | 32 → 16 | 2 | 1.00–1.92 | 1.45 | 144.2 → 144.2 µs |
| 16 | 128 → 64 | 8 | 4.00–4.94 | 4.47 | 144.2 → 144.2 µs |
| 32 | 256 → 134.4 | 16 | 8.40–9.34 | 8.87 | 179.1 → 144.2 µs |

The interval treats the published rounded mean as exact and allows any within-batch correlation. The fitted model assumes identically distributed independent matches. At four users its probability of saving the second 16-row tile is 55.1%. Neither method measures the actual acceptance histogram. The roofline uses one cold read of both banks, 4.8 TB/s and 989 TFLOP/s per H200. It omits current hidden reads, scratch, launches and communication. Actual cuBLAS kernels choose their own tiles; their latency can exceed this ideal floor. [Calculations and both weight layouts](check.json) · [Published hardware assumptions](../../../muse_bf16_proj/TASK.md)

V82 already packs committed tree rows on device and skips empty 16-row projection tiles. Its four-tile minimum enables 16 and 32 requests at native width eight. A straight-line adapter could also use that geometry with identity row order, but one user always issues the first tile. At four users the planner and scatter would have to beat a native GEMM that can already reuse each weight across all 32 inputs. V104 writes only the confirmed prefixes into draft KV; V130 and V133 address prompt materialization. A dynamic compact plan cannot claim those existing savings again. [V82](../v82_committed_draft_context/README.md) · [V104](../v104_v64_direct_draft_kv/README.md) · [V130](../v130_streamed_prompt_draft_kv/README.md) · [V133](../v133_direct_prompt_draft_kv/README.md)

## Why the final anchor still needs the MLP

The draft vocabulary head reads positions one through seven. The final layer's trained two-tap post-MLP convolution forms output position one using the MLP outputs at positions zero **and** one. Running the pinned convolution with representable BF16 weights and activations, then removing MLP row zero, changed the first hidden row’s winning token from 0 to 1 and removed token 0 from the top 16. Consequently an implementation that skips the anchor before the last MLP changes the ordinary proposal lattice. Earlier bidirectional attention also mixes the block rows. [Native-body counterexample](check.json) · [Earlier pre-RoPE mask study](../v114_draft_mask_feasibility/README.md)

## Larger measured opportunities

The separately scored H200 attention benchmark measured a weighted eight-row, one-user QKV-through-gate time of **1.605 ms stock and 0.529 ms with the standalone kernel** over 16 full and 46 sliding layers. The corresponding differences were 2.496 ms at 16 users and 4.204 ms at 32 users. These are isolated component measurements on a different path. V62 already builds a serving-specific split-KV verifier, while V139 fuses Q/K normalization, RoPE and physical FP8 KV stores on the target. V142 also reduces cold expert projection weights by up to 56 MiB per sparse layer at eight rows versus the multicast-capable parent. The retained candidates address model-wide repeated work that a one-tile draft-row adjustment cannot reach. Hardware tests must establish their serving gains. [Official attention result](../../../experiments/runs/8f9e5971f95841a0.json) · [V62](../v62_paged_verify_attention/README.md) · [V139](../v139_integrated_experts_verify/README.md) · [V142 geometry](../v142_sharded_expert_projections/geometry.json)
