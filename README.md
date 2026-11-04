# Open Kernel Challenge

Competition workspace for **CristiSavca**, initialized October 24, 2026.
Status updated November 3, 2026.
The target is a competitive SGLang serving entry. The $100,000 prize pool is
awarded on that track; first place is $50,000 plus a signed H200. Kernel results
earn serving-queue priority for the overall top 50.

## Current state

- All five official problems are initialized; the first Git commit preserves
  their original starters, references and SGLang baselines.
- Four kernel boards have passing scored entries: projections **1.1303×**,
  attention **2.6120×**, experts **1.3383×**, and the whole decoder **1.0813×**.
- Native CUDA attention, experts, projections and whole decoder passed all
  official private correctness tests at **2.6425×**, **2.3605×**, **1.4774×**
  and **2.1541×** respectively. Their exact sources are active and submitted
  for scoring. Attention has now passed scoring at **2.6120×**; the other
  three scored results are pending.
- The direct packed-GEMV expert experiment passed correctness but regressed
  to **0.7385×**. It was rejected as an optimization and was not scored.
- Modal 1.6.3 is installed and authenticated. H200 access awaits the user's
  payment verification, which Modal says unlocks the $30 free monthly allowance.
  No GPU started, no usage was charged, and the failed app is stopped.
- Selected public sources are pinned with licenses/hashes under `third_party/`.
  The fully attributed compact expert integration passed at **1.1612×** and
  its combined-attention control passed at **1.1750×**. Both are retained as
  fallbacks; the native whole decoder is substantially faster.
  Published third-party timing reports are not our performance evidence.
- SGLang is checked out at the official revision
  `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f` in `serving/sglang`.
- `serving/patch.diff` is an unvalidated draft of the expert routing changes.
  The serving track opens November 5. The checkout contains source, not weights.

| Problem | Private H200 check | Measured speedup | Scored submission |
| --- | --- | --- | --- |
| Dense projections | `8c992affff0e45b8` passed | 1.4774× | `3caa02bd72da4c49` queued; hybrid V3 `b5538a0dd25d46be` passed 1.1303× |
| Attention decode | `c21d391dbf604bb5` passed | 2.6425× | `8f9e5971f95841a0` passed 2.6120× |
| Routed experts | `143f8587286e4384` passed | 2.3605× | `3c15b408fdb04c44` queued; compact V4 `d674d6a1ada44d16` passed 1.3383× |
| Whole decoder layer | `f8e991f1a2c2469e` passed | 2.1541× | `4b216098993e4112` queued; V2 `846bf3fd98c54703` passed 1.0813× |
| Prefill layer | Official starter | — | — |

The official remote checks provide compilation, numerical tests, paired SGLang
timing and profiles. A queued job should not be submitted again. Scored results
may differ from the earlier private measurements; retain both results.

## November 3 kernel focus

Prioritize expert and attention costs, then integrate measured wins in the
whole decoder. Use the projection profiles for a quick hybrid improvement.
Prefill has a separate board and does not count toward the overall board used
for serving priority, so it is lower priority during this pass.

| Problem | New private check | Candidate |
| --- | --- | --- |
| Projections | `ce713efc5e9b44d8` | Profile-guided hybrid and larger-router split-K |
| Attention | `d046a6beb6634134` failed compile; corrected `05f77382a0c84254` passed 1.2050× | Grouped split-KV sliding-window attention and fused output norm/gate |
| Experts | `8443a6493f7c4234` | Fused finalizer for small batches |
| Experts | `cdec36c2da4b410c` failed compile; corrected `8af27f0313b44d84` passed at 0.7385×, not scored | Direct packed NVFP4 GEMVs for up to 8 tokens |
| Whole decoder | `9e55826b550f414b` | Previously validated component changes combined |
| Whole decoder | `a28df47ba0584bed` | Combined components plus fused small-batch finalizer |
| Experts | `7d7d9292e9054337` passed 1.3348× | Public local alignment and fused finalizer, original source/license preserved |
| Attention | `c21d391dbf604bb5` passed 2.6425× | Reviewed public CUDA attention with fresh output and full-batch traversal guard |
| Experts | `143f8587286e4384` passed 2.3605× | Reviewed public CUDA packed-weight experts; diagnostics removed |
| Experts | `cac648f3b3904842` passed 1.3494×; not scored | Compact V4 with the full upstream MIT notice embedded; identical executable AST, slower than native V6 |
| Whole decoder | `41920b5cf4bf423f` passed 1.1582× | Passing V2 decoder with the public V4 expert alignment/finalization |
| Whole decoder | `83d9b48acd01462e` passed 1.1612×; not scored | Same decoder integration with full upstream MIT notice embedded; slower than native V4 |
| Projections | `8c992affff0e45b8` passed 1.4774× | Reviewed native CUDA with lossless BF16 weight compression |
| Whole decoder | `f8e991f1a2c2469e` passed 2.1541× | Reviewed public CUDA decoder with bias restoration and batch-bound fixes |
| Whole decoder | `8e13cd6e80194b4f` passed 1.1060×; not scored | Passing decoder V2 with grouped window attention; slower than native V4 |
| Whole decoder | `b255f6b17f1b4758` passed 1.1750×; not scored | Compact expert integration plus grouped window attention; slower than native V4 |

**Active submission.py files hold passing candidates:** native attention V4,
native experts V6, native projections V4 and native decoder V4. Pending alternatives are
preserved separately under `candidates/` and immutable `experiments/` snapshots.
Score the exact source associated with a passing check; do not substitute an
unvalidated active candidate. The kernel phase has both November 3 and November
4 available before the cutoff; preserve time to prepare the separate serving
entry, which does not require a qualifying kernel rank.

The bounded Modal runner is documented in [tools/MODAL.md](tools/MODAL.md).
It freezes exact inputs and checks public-reference correctness plus paired
CUDA-graph timing; results are explicitly unofficial. Its lifecycle tests pass,
but actual GPU execution remains blocked by Modal account verification.
[tools/PUBLIC_RESEARCH.md](tools/PUBLIC_RESEARCH.md) records reusable sources,
licenses, the encoded-source exclusion, and the experiment workflow.

## Continue the experiments

From this directory:

```bash
okc problems
python3 tools/experiments.py status
okc status 8f9e5971f95841a0 --json  # native attention
okc status 3c15b408fdb04c44 --json  # native experts
okc status 3caa02bd72da4c49 --json  # native projections
okc status 4b216098993e4112 --json  # native whole decoder
```

The experiment helper freezes the exact source before uploading and saves a
receipt, local SHA-256 and full status under `experiments/`. The server's
content identifier is retained separately. It does not print or handle tokens.

```bash
python3 tools/experiments.py check muse_bf16_proj --label next-idea
python3 tools/experiments.py status
# Only after a passing check and a useful measured gain; use its exact snapshot:
python3 tools/experiments.py submit muse_bf16_proj \
  --file 'experiments/candidates/muse_bf16_proj/<passing-source-sha256>.py' \
  --label validated-candidate
```

Read each problem's `EXPERIMENTS.md`, `TASK.md`, and the returned profile before
tuning. Preserve candidate snapshots and failed-run diagnostics. Use the daily
checks to isolate one change at a time, and score candidates after they pass.
Gains inside the reported confidence interval may be noise.

## Route to the serving entry

1. Resolve the new private checks and inspect per-shape timings. Publish
   candidates that pass and establish useful gains.
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
