# V76 sharded draft and target decode

V76 combines three independently testable changes on V72. It targets repeated work during target verification and draft-context preparation. The candidate retains V72's eight-row tree and target-greedy verification.

| Change | Expected effect at TP8 | Cost to measure on H200 |
| --- | --- | --- |
| [V75 target projection sharding](../v75_sharded_decode_projections/independent_review.md) | Each GPU computes its output channels for the two BF16 projections surrounding each routed expert block. Estimated weight reads avoided reach 6.78 GiB per rank per eligible pass across 62 layers, or 3.39 GiB at eight rows. | At 256 rows the two gathers receive 325.5 MiB per rank. The larger push workspace adds 14 MiB per rank. |
| [V73 draft projection sharding](../v73_dflash_tp_context_projection/README.md) | The 640 MiB draft context projection occupies 80 MiB per rank. Each output channel retains the full reduction dimension. | A gather restores checkpoint channel order before RMSNorm. Gather temporaries and KV allocation affect peak usage. |
| [V74 grouped expert scheduling](../v74_humming_decode_two_ctas/README.md) | Eligible Hopper kernels can schedule two CTAs per SM, up from one. | The lower math-thread register budget can cause spills. Actual occupancy and throughput require compilation and profiling. |

## Reproduction

Base SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`. Frozen [official export](patch.diff) SHA-256 `199c3bd308a30addd45b798208a4f1573c28d0b5deb470df0de63b171b3a05ac`, 143,800 bytes across 28 source files. From the challenge repository root, use the following once eligible for the planned private check.

```sh
okc serve check serving_v0523/candidates/v76_sharded_draft_target_decode/patch.diff \
  --users 16 --flag swa-full-tokens-ratio=0.25 \
  --flag moe-runner-backend=humming --flag speculative-moe-runner-backend=marlin \
  --flag enable-layernorm-sp --flag speculative-dflash-tree \
  --flag speculative-dflash-tp-context-projection --no-wait
```

The [independent review](independent_review.json) traced every patch section to its tested parent, compiled all changed Python sources and checked the permitted submission surface. The [flag audit](flag_audit.json) compared the six launch options with all 76 fixed flag patterns from the judge template and found zero matches. The combined tree CPU tests passed 1,400 proposal cases, 9,800 accept/slot permutations, 707 replay resets, 72 attention cases and 32 capability cases. Maximum relative L2 error versus the dense attention oracle was 0.001504. These tests exercise dispatch, indexing and arithmetic models on CPU. H200 compilation, collective progress across ranks and CUDA graph replay still require an official check.

## Decision

Compare V72 and V76 at the same 16-user load after V68 passes. Require the official pass, exact output lengths, acceptable first- and later-turn agreement, peak memory within stock +2 GB per GPU and TTFT within 2× stock at each scored load. A provisional 5% throughput improvement over V72 warrants further evaluation. If V76 falls behind, the isolated V75 patch separates the larger target gathers from the draft gather and grouped-kernel scheduling. Only the scored frontier establishes an end-to-end 2× or 3× result.
