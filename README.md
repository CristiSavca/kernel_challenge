# Open Kernel Challenge

Competition workspace for **CristiSavca**, initialized October 24, 2026.
Status updated November 3, 2026.
The target is a competitive SGLang serving entry. The $100,000 prize pool is
awarded on that track; first place is $50,000 plus a signed H200. Kernel results
earn serving-queue priority for the overall top 50.

## Current state

- All five official problems are initialized; the first Git commit preserves
  their original starters, references and SGLang baselines.
- All three private H200 checks passed correctness. Attention measured
  **1.1359×**, routed experts **1.0526×**, and dense projections **0.9505×**
  relative to SGLang. Attention and experts have gains outside their intervals.
- Two scored submissions were sent November 3: attention
  `8727d7d1d2da427c` and experts `8f1f69d433364789`. The service returned
  receipt IDs and confirmed both as queued scored runs; results and ranks are pending.
- SGLang is checked out at the official revision
  `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f` in `serving/sglang`.
- `serving/patch.diff` is an unvalidated draft of the expert routing changes.
  The serving track opens November 5. The checkout contains source, not weights.

| Problem | Private H200 check | Measured speedup | Scored submission |
| --- | --- | --- | --- |
| Dense projections | `43cb543b57c64b29` passed | 0.9505× | Candidate needs improvement |
| Attention decode | `702a17ec1ad94e03` passed | 1.1359× | `8727d7d1d2da427c`, queued |
| Routed experts | `1cc96c4a472b470a` passed | 1.0526× | `8f1f69d433364789`, queued |
| Whole decoder layer | Official starter | — | — |
| Prefill layer | Official starter | — | — |

The official remote checks provide compilation, numerical tests, paired SGLang
timing and profiles. A queued job should not be submitted again. Scored results
may differ from the earlier private measurements; retain both results.

## Continue the experiments

From this directory:

```bash
okc problems
python3 tools/experiments.py status
okc status 8727d7d1d2da427c --json
okc status 8f1f69d433364789 --json
```

The experiment helper freezes the exact source before uploading and saves a
receipt, local SHA-256 and full status under `experiments/`. The server's
content identifier is retained separately. It does not print or handle tokens.

```bash
python3 tools/experiments.py check muse_bf16_proj --label next-idea
python3 tools/experiments.py status
# Only after a passing check and a useful measured gain:
python3 tools/experiments.py submit muse_bf16_proj --label validated-candidate
```

Read each problem's `EXPERIMENTS.md`, `TASK.md`, and the returned profile before
tuning. Preserve candidate snapshots and failed-run diagnostics. Use the daily
checks to isolate one change at a time, and score candidates after they pass.
Gains inside the reported confidence interval may be noise.

## Route to the serving entry

1. Resolve the two scored results and inspect per-shape timings.
2. Improve the largest measured costs, especially the expert path at larger
   speculative batches, and combine passing kernels in the whole decoder.
3. Transfer measured improvements into the pinned serving source. Export a
   legal patch with `okc serve diff serving`.
4. From November 5, validate the full 8×H200 service on the hidden conversation
   workload, including output agreement and time to first token. Optimize the
   throughput/interactivity curve at 1, 4, 16 and 32 concurrent users.

Kernel submissions close **November 5 at 2:59 am Eastern** (November 4 at
11:59 pm Pacific). Serving opens **November 5 at 3 pm Eastern**. The live site
says serving closes November 19 Pacific, while the legal terms say November 18;
target a finished entry by **November 18** until the organizer resolves this.

[NOTES.md](NOTES.md) records rules, scoring, source links, limits and the date
discrepancy. The top 50 on the overall kernel board get serving-queue priority;
eligible entrants may participate in the serving track independently, without
a qualifying kernel rank (official terms section 6).
