# Public MUSE sources reviewed November 3

The requested [Auto-research-agent repository](https://github.com/rohanarun/Auto-research-agent)
is a macOS research coordinator and also publishes actual MUSE candidate source
and local benchmark results. We imported selected text artifacts at commit
`018b7a102348ea3b73701e8be52d24f82d8c91ee`, with URLs, SHA-256 hashes and MIT
license, under `third_party/auto_research/`. No app build or full checkout was
needed; all imported artifacts total about 1.5 MiB.

The compact GPT-6.1 Sol MoE candidate uses per-expert Triton route alignment,
stock packed-NVFP4 Marlin GEMMs, and fused final normalization/weighted sum.
It is 8.7 KiB and has been read end to end. Its local published evaluation
reports approximately 1.4243×. That measurement is not an official leaderboard
result and does not establish our correctness or performance.

`muse_moe_w4a16/candidates/public_local_align_v4.py` preserves its kernel behavior
and adds provenance comments. Our private check `7d7d9292e9054337` passed all
five numerical tests and six timed shapes at 1.3348× (CI 1.3148–1.3548).
The exact snapshot passed scoring as `d674d6a1ada44d16` at 1.3383×
(CI 1.3182–1.3583).
The raw original, evaluation and license are retained.

The original compact adaptation points to the retained license. The separate
`public_local_align_v7_licensed.py` now embeds the complete upstream MIT notice
inside the uploaded file; its AST including docstrings is identical to V4.
Its exact source is in private check `cac648f3b3904842`. The analogous
`public_alignment_v6_licensed.py` has AST equality against decoder V3. V3
passed at 1.1582×; the complete-notice source is now in private check
`83d9b48acd01462e` before scored promotion.

The larger Gemini cohort reports approximately 2.65× attention, 2.38× MoE and
2.17× decoder speedups in that repository's local protocol. These are research
leads only. The MoE and decoder headers identify Apache-2.0-derived code; the
decoder attributes team vjeux. Preserve both the benchmark license and upstream
Apache attribution in any adaptation. Readable inline CUDA still needs review
of preparation, cache handling, precision and all required keys/experts.

Further inspection found the published attention evaluation repeats identical
paired timings and the same speedup ratio across distinct cases (for example,
all five first-case baseline times are exactly 100 microseconds). We have not
established why. Treat these files as unverified reports, not independent timing
evidence. Our actual official measurements govern all promotion decisions.

Reviewed adaptations now in official private checks:

- Attention `public_cuda_v4.py`, `c21d391dbf604bb5`: original CUDA/CPP strings
  unchanged, full MIT/Apache notices embedded, fresh output per invocation,
  and full-attention batches above 64 use the complete fallback because the
  native flat traversal has a 64-segment array. Cache ownership/window coverage
  were independently reviewed. Global scratch needs separate ownership before
  overlapping-stream serving integration. Our private run passed all 16
  numerical tests at 2.6425×, CI 2.5935–2.6822; exact-source scored submission
  `8f9e5971f95841a0` is pending.
- Experts `public_cuda_v6.py`, `143f8587286e4384`: production CUDA/CPP strings
  and dispatch constants unchanged, unused diagnostics/timers removed, licenses
  embedded. CPU packing/decode checks covered 10,129 words per packed layout
  and all 119 positive-normal scale codes. GPU behavior remains unverified.
- Decoder `public_alignment_v3.py`, `41920b5cf4bf423f`: integrates only the
  validated compact expert route alignment and finalizer into our passing
  decoder. All previous decoder functions except `_experts` are AST-identical;
  its existing expert-ID rotation remains intact.

The Gemini dense projection contains compressed/encoded source and `exec`, so
we did not import or submit it. The current challenge forbids encoded or
compressed source. The readable upstream projection was instead retained from
[vjeux/okc-2026](https://github.com/vjeux/okc-2026) at commit
`f35a41ee1c13e316dabbf772830d47b23b453393`, with its Apache-2.0 license. Its
lossless weight compression, split-K kernels and preparation-time tuning were
subsequently audited and adapted for private check `8c992affff0e45b8`, detailed
below. GPU correctness and speed remain unverified. No public candidate's
reported score is treated as one of ours.

## Our research loop

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) describes an
edit, bounded experiment, evaluate, retain-or-discard loop. That pattern fits
this task; we did not establish who borrowed it from whom.

1. Pick one measured bottleneck and write a hypothesis.
2. Freeze readable source with its origin and hash; leave the evaluator fixed.
3. Run a bounded H200 public-reference check when Modal is activated. Save
   failures as well as per-shape timings; use fresh inputs and CUDA graphs.
4. Compare against the unchanged baseline and our passing incumbent. Keep
   numerical accuracy and test coverage visible beside timings.
5. Run an official private check on useful candidates. A response's outer
   `status: ok` or static `checks.ok` does not mean numerical tests passed;
   require `result.passed: true` and a useful measured speedup.
6. Score the exact passing snapshot and transfer measured wins to serving.

`tools/experiments.py` tracks official runs; `tools/modal_h200.py` tracks bounded
Modal runs. This is a reproducible experiment workflow, not an unattended
model-calling daemon or a promise of background work after the chat stops.

Dense projection adaptation `public_compressed_v4.py` is now in private check
`8c992affff0e45b8`. It retains all 14 upstream native source strings and embeds
Apache-2.0/CUTLASS notices. CPU codec review covered 495,616 decode cases and
7,403,520 finite escape/base cases. Current compressed SK2 routes must retain
CL=1; a dormant CL>1 compressed branch drops peer escape corrections. Actual
CUDA correctness, timing and prepare-time budget remain unverified.

Native decoder adaptation `public_cuda_v4.py` is in private check
`f8e991f1a2c2469e`. It embeds the original licenses and retains native source
strings. Bias cloning and unconditional restoration prevent synthetic tuning
from modifying checkpoint bias on failure. It also applies the full-attention
64-request guard. All eight routes and weight precision were reviewed. The
prepared state assumes serial calls; serving requires a separate concurrency
review. CUDA synchronization, UV reassociation accuracy and timing are pending.

## Upstream freshness and attention comparison

GitHub's live main refs checked at 2026-11-04 00:29 UTC still match both pinned
commits above. There was no newer source to import. A separate read-only
comparison against upstream
[muse_attention_decode.py](https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_attention_decode.py)
(SHA-256 `5811d22bb985092ba5a29a8f4c8e266de189399ee708d7b14efcb9c9bf5a510e`)
found no substantive active H200 math difference from the queued native
adaptation. Planning matched across 98,304 H200 shape combinations, tuning
tables matched, and 17 active math/packing helpers matched after selecting
the SM90 preprocessing branches. The smaller upstream source mainly removes
dead experiments, old-architecture paths and unused ABI fields. This is not
evidence for another performance check. Our adaptation already fixes the
reusable output alias and full-attention batch bound; scratch ownership still
needs separate review before concurrent serving use.

Scratch cleanup ledger: the sole temporary upstream comparison file was
removed after hash, Git and open-handle inspection. Removed 154,604 logical
bytes / 155,648 allocated bytes. Observed physical free space increased
139,264 bytes, from 41,892,605,952 to 41,892,745,216 bytes; concurrent filesystem
activity can make that differ from the allocation. No retained source or
candidate was deleted.
