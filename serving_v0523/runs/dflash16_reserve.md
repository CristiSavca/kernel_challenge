# DFlash2 block 16 reserve

The proposed private check uses the exact V48b patch, SHA-256
`e59d8efcd1a544567686708cf4dd48361103e7640af95ff4d75ee556c4121a8f`,
with `--flag swa-full-tokens-ratio=0.25 --flag speculative-num-draft-tokens=16`.
V48b scored 1.1037 with block 8. The block 16 check is unscheduled.
[V48b judge result](8ebed2bced8942ff.json)

The pinned DFlash worker accepts the override, logs the checkpoint mismatch, and calls
`set_block_size` before graph capture. The call updates convolution row boundaries.
The two-tap weights and selector dimensions are independent of the block size. The BF16
draft has five sliding-attention layers, a 2,048-token window and 1,048,576-position RoPE
metadata. A 16-row block exceeds the checkpoint's configured width of 8, so measured
acceptance at the longer positions is unknown.
[Draft config](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/blob/46c90f412549c472941a68382d91a99e5e6372a0/config.json)

Both decode graph runners size the flattened width from the resolved draft count,
giving 512 rows at 32 requests. The target verifies with ordinary causal attention
and the existing precision. Greedy verification emits the consecutive matching
drafts and one target token. The scheduler uses the result's padded stride and
accepted lengths, then truncates at the ordinary EOS or requested output-token boundary.
Width 16 still needs the official numerical, every-token and output-count checks.

With a constant conditional match probability `p`, expected tokens per verify
are `A(B) = sum(p**i for i in range(B))`, including the target token. Keeping that
probability at the extra positions gives `A(16) / A(8) = 1 + p**8`.
The last column is the total step-time increase that breaks even for decode throughput.
The end-to-end score also depends on prefill and the scored throughput frontier.

| Users | V48b tokens per step | Fitted p | Block 16 tokens per step | Step-time break-even |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.1 | 0.795 | 4.754 | +16.0% |
| 4 | 4.0 | 0.787 | 4.587 | +14.7% |
| 16 | 3.9 | 0.778 | 4.424 | +13.4% |
| 32 | 4.1 | 0.795 | 4.754 | +16.0% |

- At page size 1, the request-to-token map allocates `context_len + 4 + B`
  columns and overlap decode plans through `committed_len + 2*B`.
  At width 16 this can overrun a row once the committed prefix exceeds
  `context_len - 12`, inside the admitted context range. Allocating 512 extra
  logical slots across 32 users is small compared with the observed 3,308,220
  full KV slots. The per-request row bound is the concern.
- The judged startup rehearses dense non-greedy verification at batch
  32. V48b logs 8.17 GiB peak plus its FP32 inputs and a 9,030.3 MiB borrowable
  graph pool. Doubling rows could bring the rehearsal peak to about 16.34 GiB
  and trigger the existing unborrowed fallback. The estimated 16.73 GiB headroom
  is within the incumbent static reserve of roughly 20.7 GiB, but final KV
  capacity, allocator peaks and the judge's +2 GB limit need measurement.
  [V48b startup log](8ebed2bced8942ff.json)

Source anchors in the pinned `297e54fd105e974cdf401e5b4d20253a22c16f90` checkout are
`arg_groups/speculative_hook.py:389`, `models/dflash.py:806`,
`speculative/dflash_worker_v2.py:507,799,2824,3002`, `speculative/dflash_info_v2.py:142`,
`mem_cache/allocation_sizing.py:96`, `model_executor/runner/decode_cuda_graph_runner.py:349`,
and `managers/scheduler_components/batch_result_processor.py:770`, all under
`python/sglang/srt`. Audit used the public checkpoint config, source and existing judge
logs. No GPU test or submission was made.
