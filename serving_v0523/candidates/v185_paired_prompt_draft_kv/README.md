# V185 projects draft keys and values before exchanging prompt rows

V171 assigns each prompt row to its tensor-parallel owner for the draft context FC and RMSNorm. It then gathers every normalized BF16 row onto every GPU. V185 divides the next operation, a projection into all five draft layers’ local keys and values, among groups of GPUs. Each rank returns the smaller projected rows directly to its existing draft cache writer. The change applies to eligible long prefills and preserves V171’s six launch flags. [Submission patch](patch.diff) · [Changes against V171](delta.diff)

## Per-rank cost

A group of two or four neighboring ranks shares its input rows. Ranks at the same position in different groups exchange their original packed, transposed BF16 KV matrices. Each rank projects its group's rows with each matrix using a full-width native BF16 GEMM. It sends the destination its projected rows in canonical token order. The first round of grouped PyNCCL sends carries the input rows and weights. The second round carries projected outputs. No weight copies persist between calls. [Production implementation](delta.diff)

| Live tokens at TP8 | Ranks per group | V171 gather payload sent | V185 input + weights + projected outputs sent | Returned KV | V185 named staging tensors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 4 | 112 MiB | 48 + 20 + 10 = 78 MiB | 20 MiB | 114 MiB |
| 16,384 | 2 | 224 MiB | 32 + 60 + 30 = 122 MiB | 40 MiB | 194 MiB |

At 16,384 rows, this cuts per-rank logical send volume by 45.5%. The prior gathered feature tensor occupies 256 MiB. The new 194 MiB includes the 40 MiB returned KV tensor. Both paths also hold the local input and the permanent matrix. The new path runs four smaller GEMMs instead of one large GEMM at this shape. cuBLAS workspace, its selected kernels, allocator retention, CPU votes and NVLink scheduling determine actual timing and peak memory. [Byte accounting and archive counts](geometry.json)

In the older eight-H200 V41 trace, 75 of 241 prompt forwards at 16 users qualified by size. Those calls accounted for 1,228,800 of 1,415,564 live prompt rows. At 32 users, 153 of 351 forwards met the geometry gate. These are opportunities in the archived shapes. Selection also depends on live module, cache, graph and communicator state. [Trace reproducer](geometry.py)

## Eligibility and recovery

V185 requires an ordinary uninstrumented V171 prompt projection, matching TP4 or TP8 PyNCCL, an active fused draft helper and contiguous BF16 matrices. The dimensions must save at least one eighth of the old payload. A rank-wide shape and allocation vote precedes the exchange. The ranks vote again after projection so a GEMM workspace failure on any rank leaves the ordinary V171 gather available to all peers. The CPU consensus buffers are allocated outside PyTorch inference mode to support Gloo’s background copies. During CUDA capture the parent’s ordinary gather remains available. [Projection and fallback](delta.diff)

The worker keeps original prefix positions, physical locations, cache scales, normalization, RoPE and write order. The pooled writer accepts the raw per-layer projections. Warm calls reuse the existing K/V workspace and cold calls allocate the outputs on demand. A failed fused write can overwrite the same slots from the raw projections using the original per-layer fallback, including already written layers. [Worker replay and limits](worker_review.md)

## Verification

- A TP4/TP8 Gloo replay exercised 168 eligible rank calls and 84 asymmetric declines. All 367,840 projected BF16 values matched the full-gather oracle exactly. [Transport results](transport.json) · [Reproducer](check_transport.py)
- A second replay used the pinned PyNCCL send, receive, stream selection, grouping and state-restoration methods. A Gloo transport at the raw pointer boundary reproduced every output across 372 rank-local groups and 1,992 send or receive calls. [Pointer replay](pynccl.json) · [Reproducer](pynccl_check.py)
- The real Muse outer forward and its projection ran in inference mode at TP4/TP8. The replay covered 288 rank-forwards, 576 Full or Breakable output trims and 48 projected outputs. It compared 412,080 projected BF16 components against a dense CPU oracle with relative tolerance 0.008 and absolute tolerance 0.015. CPU implementations supplied final logits observation and RMSNorm. [Model results](model.json) · [Reproducer](check_model.py)
- The worker fixture matched all 2,042,880 physical BF16 cache values across 112 paired cases, including reserved capacity, remapped slots and a failure after the third layer wrote. The CUDA norm/RoPE and final scatter were emulated. [Cache results](worker.json) · [Reproducer](worker_check.py)
- The official CLI exported 184,451 bytes across 28 source paths. Full and incremental patches apply and reverse on the published SGLang pin. Static checks cover the 400 KiB size limit, source and flag restrictions, and all 76 denied flag patterns. H200 output agreement, TTFT, peak memory and throughput remain unmeasured. [Validation](validation.json) · [Published serving rules](https://openkernelchallenge.dev/api/v1/serving)

V187 combines this endpoint with V182’s paged verification and fused query preparation on V68’s measured lineage. [V187 export](../v187_paged_verify_paired_prompt/validation.json)
