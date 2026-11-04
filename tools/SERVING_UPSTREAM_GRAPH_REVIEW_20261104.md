# Upstream draft-extend graph fix: different path from this challenge

2026-11-04. **Do not port PR40762 as a Muse DFlash optimization.** The change
is absent from the pinned checkout, but the challenge does not execute the
EAGLE path that it changes. No serving source, configuration, patch, candidate,
GPU state or provider account was modified by this review.

The user-linked [AICoder article](https://aicoder.com/news/news-1792384474854-fl9ix)
summarizes broad SGLang release gains across other models and hardware. Those
headlines are not evidence of a speedup for the pinned Muse configuration.
This review therefore checked one concrete primary-source change rather than
assuming a package upgrade would transfer the reported gains.

[SGLang PR40762](https://github.com/sgl-project/sglang/pull/40762), merged
October20, fixes FlashAttention eligibility for EAGLE draft-extend graph capture.
Its [final diff](https://github.com/sgl-project/sglang/pull/40762/files), commit
`f3f66ecc12d64eb4806aea28d16ed491e2c2ed23`, adds seven lines in
`eagle_worker_v2.py`: a CUDA-only lazy import and an append of
`FlashAttentionBackend` to `graph_supported_backend_types`. The PR reports an
EAGLE-chain example; it neither changes DFlash nor establishes Muse throughput.
The actual diff is narrower than the PR description's earlier two-line wording.

In checkout `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`, the EAGLE list at
`python/sglang/srt/speculative/eagle_worker_v2.py:529` lacks that append. This
means the fix is not already present in that EAGLE implementation. It does not
mean the selected DFlash draft is running eagerly.

The independently read local call chain is decisive:

1. `serving/RULES.md:26` selects `--speculative-algorithm DFLASH`, the released
   Muse DFlash2 drafter and eight draft tokens. These are the judge's fixed flags.
2. `speculative/spec_info.py:311` selects `DFlashWorkerV2` for this algorithm.
   `dflash_worker_v2.py:632` explicitly describes its plain `TpModelWorker`;
   DFlash has no EAGLE draft/draft-extend worker split.
3. `dflash_worker_v2.py:679` initializes its own graph path, controlled by the
   decode-graph setting, platform support, DP-attention participation and a
   memory-availability guard. At730 it calls the plain worker's
   `init_cuda_graphs(capture_decode_cuda_graph=...)`. It never consults EAGLE's
   `graph_supported_backend_types`.
4. `managers/tp_worker.py:423` forwards graph initialization to its model runners.
   DFlash constructs its draft block with `ForwardMode.TARGET_VERIFY` at
   `dflash_worker_v2.py:2604`, then calls the draft model runner at2651.
5. `model_executor/forward_batch_info.py:272` includes TARGET_VERIFY in
   `is_cuda_graph`. `model_runner.py:1901` tests graph eligibility and at1917
   executes the decode graph runner when eligible. FlashAttention's current
   `target_verify_metadata` allocation is also present at approximately2438.

Thus the precise conclusion is **different path**, not “already applied.” An
EAGLE user could benefit from the upstream change, but adding those seven lines
would not change this configured DFlash branch. General graph-capture principles
are transferable; this particular allowlist patch is not an actionable Muse
performance lead.

This is source-chain evidence, not a claim that every real Muse batch replays a
graph. Capture can be disabled, memory can be insufficient, DP attention can
require eager execution, and a batch can miss a captured bucket. Any future
runtime evidence of those conditions should be investigated in DFlash's own
guards and graph eligibility. It would not justify bypassing them by copying
an unrelated EAGLE allowlist. No runtime log or graph trace establishing such a
failure was available here, so no new graph experiment is proposed.

The inspected source files are unchanged relative to the pinned checkout:

| File under `serving/sglang/python/sglang/srt/` | SHA256 |
| --- | --- |
| `speculative/eagle_worker_v2.py` | `05a269f9f3e4ea42b58714bab7beb8612b10a640fa9733bc5ba9b7f643e923fb` |
| `speculative/dflash_worker_v2.py` | `7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c` |
| `speculative/spec_info.py` | `43e59f9d15b711753a9170807452f5643085cc9d9811345761559b0e908d2fa9` |
| `model_executor/model_runner.py` | `191062bf3607d5f87367ceb9c2f773653241a7e65bcba1a9d985b957fc80f92d` |
| `layers/attention/flashattention_backend.py` | `53489175aace7b97498efb10ae8b01c489d35e91058842eee02b19aec18f1a3c` |

The GitHub files view supplied the actual final patch. Raw-diff and API retrieval
failed in the web tool; no conclusion relies on an unobserved raw response.
No release-wide upgrade or other PRs were assessed in this bounded review.
