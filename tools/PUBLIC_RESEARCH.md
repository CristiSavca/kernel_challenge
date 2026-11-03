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
and adds provenance comments. It is in our own private check `7d7d9292e9054337`.
The raw original, evaluation and license are retained. It is not scored yet.

The larger Gemini cohort reports approximately 2.65× attention, 2.38× MoE and
2.17× decoder speedups in that repository's local protocol. These are research
leads only. The MoE and decoder headers identify Apache-2.0-derived code; the
decoder attributes team vjeux. Preserve both the benchmark license and upstream
Apache attribution in any adaptation. Readable inline CUDA still needs review
of preparation, cache handling, precision and all required keys/experts.

The Gemini dense projection contains compressed/encoded source and `exec`, so
we did not import or submit it. The current challenge forbids encoded or
compressed source. The readable upstream projection was instead retained from
[vjeux/okc-2026](https://github.com/vjeux/okc-2026) at commit
`f35a41ee1c13e316dabbf772830d47b23b453393`, with its Apache-2.0 license. Its
lossless weight compression, split-K kernels and preparation-time tuning are
potential ideas; that entire implementation has not been approved or submitted
here. No public candidate's reported score is treated as one of ours.

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
