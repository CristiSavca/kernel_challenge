# Attention experiments

## Fused QKV producer, v1

- Off-board check: `702a17ec1ad94e03`.
- Candidate: `candidates/fused_qkv_v1.py` (same code as current submission).
- Fuses Q/K RMSNorm, interleaved RoPE, BF16-to-FP8 query conversion and K/V cache scatter.
- Preserves BF16 rounding after normalization and after RoPE, and honors input cache scale tensors.
- Cache slot zero is written when requested, as required by this challenge (production reserves it).
- Uses unchanged production FA3 attention and output RMSNorm/gate.
- Private H200 check passed correctness and static checks: **1.1358528683×**,
  95% interval **[1.1188150753, 1.1528906613]** relative to SGLang.
- Scored submission November 3: `8727d7d1d2da427c`; passed at **1.1345×**. Uses the same frozen source as the passing private check.

### Static audit

- `py_compile`, AST parsing and whitespace checks passed locally; the remote H200 run also compiled and passed correctness.
- The producer handles both TP8 (16 Q / 2 KV heads) and DP (128 Q / 16 KV heads), and the batch and token dimensions come from configuration.
- Cache offsets use the supplied flat `cache_loc`; flattening `[pages, page_size, Hkv, 64]` this way supports both page size 1 and 64.
- FA3 receives the original page table, causal block offsets, lengths, window and scheduler metadata. Every key in the required context remains visible.
- Query normalization and RoPE keep the production BF16 rounding steps before the permitted FP8 query cast. K/V scaling loads the actual input scale tensors; zero slots are not skipped.
- RoPE table caching depends only on the model theta and dimension, contains no input-derived values, and matches the official baseline's constant table.
- No host sync, data-dependent allocation/control flow, output caching, subprocess or network use appears in the candidate.
- The initial queue delay has cleared for this private check. Scored submission performance and rank remain separate from the measured private result.

## Grouped sliding-window split-KV, v2 (November 3)

- Private H200 check: `d046a6beb6634134`; last observed status `received`, GPU results pending.
- Frozen candidate: `candidates/window_splitkv_v2.py`. SHA256: `f6f5b76c19bf3057d736b34f27aea9657487759ff86c0514b8f9288230b9c5e1`.
- The v1 profiles spend 8–15 microseconds in sliding FA3 attention, plus separate split-combine and metadata-preparation launches.
- v2 replaces sliding FA3 with paged grouped attention: all eight query heads sharing one KV head, and all queries in a verify block, reuse each loaded K/V tile.
- Splits use an online stable softmax and FP32 accumulation/partials; a second kernel combines partials, rounds attention to BF16, and applies RMSNorm and sigmoid gating. The full-attention FA3 path is unchanged.
- Query activations stay BF16 on the sliding path (more precise than the permitted stock FP8 query rounding); cache tensors remain FP8. Keys/values convert exactly from FP8 to BF16 for tensor-core operands, and actual input scale tensors are applied.
- The split range covers `window + q_len - 1` positions: 2,055 for an 8-token block. Each query independently masks its causal end and exact 2,048-token window. No required key is dropped.
- Flat cache addressing respects supplied page sizes, including 64, and writes new cache entries before attention. Empty splits produce zero output and negative-infinity log-sum-exp, and merge with zero weight.
- Local Python syntax and whitespace checks passed; root independently reviewed the causal/window/page/scale/merge math. CUDA compilation, correctness and timing remain unverified until the private result arrives.
- No second private check or scored submission was made for this version. Next action: read this job once complete, then promote only if correctness passes and measured gain is useful.
# November 3 FP8 masked-load repair

Private check `d046a6beb6634134` completed with `result.passed: false` and no
speedup. Static source checks passed, but sliding-window tests failed compilation:
Triton cannot cast the integer masked-load fill value directly to FP8 E4M3.
Full-attention behavior was unchanged. Both FP8 cache loads now use `other=0.0`.
The repaired source is `candidates/window_splitkv_v3_fp8_fill.py`, also active in
`submission.py`, and is pending check `05f77382a0c84254`.

The scored V1 remains the validated attention incumbent. Do not interpret the
outer result `status: ok` as correctness; inspect `passed` and the test records.

## Goal continuation: native CUDA candidate

Active `submission.py` was restored to passing/scored fused-QKV V1 while the
Triton V3 experiment runs from its immutable snapshot. The independently
reviewed native CUDA adaptation `candidates/public_cuda_v4.py` is in private
check `c21d391dbf604bb5`. Its CUDA/CPP strings match the imported source; Python
changes give each call fresh output and route full-attention batches above 64
to the complete fallback. The published timing report repeats identical pairs
and is not treated as performance evidence. Our private check is pending.

## V3 passed and submitted

Check `05f77382a0c84254` passed all 16 numerical tests and 12 timed shapes at
1.2050178708×, CI [1.1869426027, 1.2230931388]. The exact immutable source
`103624d509012ab92fd2f3a2237827bc739a542a0eae5a333802d3cf3712aa60` is now active
and was submitted for scoring as `52117bcc7137435e` (pending).
