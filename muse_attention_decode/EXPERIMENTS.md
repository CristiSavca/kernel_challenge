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
- Scored submission sent November 3: `8727d7d1d2da427c`; confirmed queued in
  mode `run`, result pending. Uses the same frozen source as the passing check.

### Static audit

- `py_compile`, AST parsing and whitespace checks passed locally; the remote H200 run also compiled and passed correctness.
- The producer handles both TP8 (16 Q / 2 KV heads) and DP (128 Q / 16 KV heads), and the batch and token dimensions come from configuration.
- Cache offsets use the supplied flat `cache_loc`; flattening `[pages, page_size, Hkv, 64]` this way supports both page size 1 and 64.
- FA3 receives the original page table, causal block offsets, lengths, window and scheduler metadata. Every key in the required context remains visible.
- Query normalization and RoPE keep the production BF16 rounding steps before the permitted FP8 query cast. K/V scaling loads the actual input scale tensors; zero slots are not skipped.
- RoPE table caching depends only on the model theta and dimension, contains no input-derived values, and matches the official baseline's constant table.
- No host sync, data-dependent allocation/control flow, output caching, subprocess or network use appears in the candidate.
- The initial queue delay has cleared for this private check. Scored submission performance and rank remain separate from the measured private result.
