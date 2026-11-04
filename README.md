# Open Kernel Challenge

Competition workspace for **CristiSavca**, initialized October 24, 2026.
Status updated November 4, 2026, 21:27 UTC (board), after the V28 private check.
The target is a competitive SGLang serving entry. The $100,000 prize pool is
awarded on that track; first place is $50,000 plus a signed H200. Kernel results
earn serving-queue priority for the overall top 50.

## Latest measured state

CristiSavca is 408th overall. Public scores are dense projections 1.4868×,
attention 2.6120×, experts 2.3737× and whole decoder 2.1812×. The decoder leader
is 2.5762×. Passing public sources remain preserved; private experimental
regressions have not replaced them.

Decoder V28 passed all 13 official numerical tests but failed its frozen speed
gate: **2.1146x**, versus the preserved public decoder's **2.1812x**. The three
affected weighted loads have 5.15% higher normalized latency than the public
parent; all six affected cases are slower than both exact-source anchors.
Root and independent critic agree: close the attempt without scoring or retry.
Replacing the public kernel's vector atomics with private owner sums did not
carry over the gain seen against the serving prototype's scalar CAS path.
See [V28 results](experiments/iterations/20261104-decoder-v28-closeout.json).

Serving V27 now has an isolated, reproducible 213,160-byte patch connecting the
passing V26 core to actual SGLang weight loading, forward dispatch and serial
model-owned workspaces. Independent CPU review passed rank mapping, atomic
installation/rollback, norm handling and chunk boundaries. The stock norm
setter's device lookup was adapted for released raw weights. Device validation
is being prepared against the exact stock methods; this is not yet a GPU,
full-engine or official serving pass. The dirty serving checkout was preserved.

Serving expert prototype V26 passed official private job `274c69cbc0954d76`
with all five tests, six benchmarks and clean checks. Its same-run stock score
is **1.6851x**, CI [1.6598, 1.7104], and every tested load is faster: latency
falls 13.03% at M1 and 64.41% at M256. Root and independent critic reproduced
all frozen gates. Private partials and ordered FP32 reduction replace scalar
CAS accumulation while retaining the reviewed serving BF16 boundaries and
original finalizer. Keep this component for integration validation. It remains
slower than our public native kernel, so no scored source was replaced. Full
serving memory, generic ranks, concurrency, graphs, prefill and throughput remain
unverified. See [V26 results](experiments/iterations/20261104-native-serving-v26-closeout.json).

V25 remains closed: it passed correctness at 1.2538x but made M1 13.63% slower
than stock. V26's cross-run normalized aggregate is 1.3440x V25's; this is not
same-input causal attribution. The next experiment tests the installed V27
adapter's numerical and workspace behavior on one bounded H200.

The research loop now adapts EAT's diagnosis-driven role changes using our
existing tools. Its [current handoff](tools/RESEARCH_LOOP.md) is separated from
the preserved historical log. Primary-source reviews cover
[EAT](tools/EAT_REVIEW_20261104.md),
[ParallelKernelBench](tools/PARALLEL_KERNEL_BENCH_REVIEW_20261104.md) and
[Together Compiler](tools/TOGETHER_COMPILER_REVIEW_20261104.md).
No external paid agent framework was installed; published gains are not our scores.

Serving-finalizer v22 passed official private job `cffa322caeff4aa2`, with all
five correctness tests, six timed cases and clean checks. Compared with the
same-run stock Marlin component, latency fell 32.6%/21.4%/18.0% at 1/8/16 tokens;
larger controls stayed within 0.05%. Both independent analyses passed the fixed
gate. This is a serving-component improvement, not a faster native kernel entry
or a measured full-model gain. Broader free-credit H200 validation subsequently
failed the generic H511 cancellation case; the first runner lost its detailed
checkpoint after application teardown. The repaired runner recovered a second
run: our candidate equals the independent FP32 oracle exactly on that case,
while stock differs by at most 2.3842e-7. The strict relative check still fails;
the later coverage run passed all 15 remaining cases, eight stream calls and
eight fresh graph replays. Across the two runs, 89 of 90 numerical cases pass;
the cancellation failure remains explicit. A completed stage diagnostic
reproduced all three helper outputs bitwise. It localized
the residual to unequal original means/inverses on identical opposing-route
squares. Keeping logical H511 while changing physical row layout removed that
inequality. This supports reduction-layout dependence; it does not establish
general correctness or clear the component for serving promotion. No public
kernel score/source was replaced. See
[the failure evidence](tools/SERVING_FINALIZER_VALIDATION_FAILURE_20261104.md)
and [coverage evidence](experiments/iterations/20261104-finalizer-coverage-root-result.json).

Fresh arXiv, X/release, DeepSeek, Hugging Face and CUTLASS research is linked in
[the active handoff](tools/RESEARCH_LOOP.md). Current official receipts and
research retain exact sources, licenses, evaluator identity and measured limits.
Serving router V24 is implemented in an isolated patch. Its bounded H200 run
passed 432 exact-output cases, eight padding cases and eight legacy API cases,
then stopped because the benchmark expected two parent launches and observed
three. The candidate used one launch at T16. Stream/graph checks and paired
timings were not reached, so validation is incomplete and no speed gain is
claimed. The fixed attempt is closed without retry or promotion; see
[the result](experiments/iterations/20261104-router-local-ids-root-result.json).
Unchanged-finalizer diagnostics are closed; full serving compatibility remains
unverified. Separately, exact-one
attention rescale candidate V23 passed official private correctness check
`0193a093baa446f3` but failed its fixed speed gate: 2.5911× versus the public
parent's 2.6120× and private parent's 2.6425×. It is closed without a scored
submission or replacement of the passing attention entry. Research leads need
an actual compatible code path before a new optimization.

The user-requested Lean methodology now has a working first implementation:
[source-bound packed-decoder proof](formal/README.md). Lean checks all four
MULDEC integer outputs for every 32-bit word and the packing-position bijection.
The real source passes; three deliberately broken constant variants fail Lean.
Seven source/axiom tests and three upload-gate tests pass. `--verify-ptx` runs
the proof before an official upload and keeps its exact report hash in the
receipt. This is our independent fragment implementation, not an imported SF
Tensor model or a full PTX/CUDA correctness proof. No kernel timing changed.

## Earlier setup and November 3 results

- All five official problems are initialized; the first Git commit preserves
  their original starters, references and SGLang baselines.
- Four kernel boards have passing scored entries: projections **1.4868×**,
  attention **2.6120×**, experts **2.3561×**, and the whole decoder **2.1502×**.
- Native CUDA attention, experts, projections and whole decoder passed all
  official private correctness tests at **2.6425×**, **2.3605×**, **1.4774×**
  and **2.1541×** respectively. Their exact sources are active and submitted
  for scoring. All four scored reruns passed, with 43 numerical tests and
  60 timed cases in total, no flags or rejections, and exact source matches.
  All 32 recorded experiments are resolved; the result watcher has exited.
- At November 3, 9:20 pm Eastern (November 4, 02:20 UTC), the official overall
  board placed CristiSavca **386th of 1,946** with four problems solved.
  This is a live snapshot; top-50 serving-queue priority has not been earned.
- The direct packed-GEMV expert experiment passed correctness but regressed
  to **0.7385×**. It was rejected as an optimization and was not scored.
- Modal payment verification is complete and the $30 allowance is active.
  A single-H200 smoke passed, then terminated its sandbox and stopped the app.
  Its metered compute was about $0.00277, covered by credits.
- A 15-minute research heartbeat with an independent critic is active. The
  first-place push targets the decoder's dominant expert work; see
  [tools/RESEARCH_LOOP.md](tools/RESEARCH_LOOP.md) for the hypothesis queue,
  acceptance criteria, spending guard and handoff.
- Selected public sources are pinned with licenses/hashes under `third_party/`.
  The fully attributed compact expert integration passed at **1.1612×** and
  its combined-attention control passed at **1.1750×**. Both are retained as
  fallbacks; the native whole decoder is substantially faster.
  Published third-party timing reports are not our performance evidence.
- SGLang is checked out at the official revision
  `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f` in `serving/sglang`.
- `serving/patch.diff` is an unvalidated draft of the expert routing changes.
  The serving track now opens November 10. The checkout contains source, not weights.

| Problem | Private H200 check | Measured speedup | Scored submission |
| --- | --- | --- | --- |
| Dense projections | `8c992affff0e45b8` passed | 1.4774× | `3caa02bd72da4c49` passed 1.4868× |
| Attention decode | `c21d391dbf604bb5` passed | 2.6425× | `8f9e5971f95841a0` passed 2.6120× |
| Routed experts | `143f8587286e4384` passed | 2.3605× | `3c15b408fdb04c44` passed 2.3561× |
| Whole decoder layer | `f8e991f1a2c2469e` passed | 2.1541× | `4b216098993e4112` passed 2.1502× |
| Prefill layer | Official starter | — | — |

The official remote checks provide compilation, numerical tests, paired SGLang
timing and profiles. A queued job should not be submitted again. Scored results
may differ from the earlier private measurements; retain both results.
The completed scored audit is saved in
[experiments/native-scored-audit.json](experiments/native-scored-audit.json),
with current leaderboard snapshots under `experiments/leaderboards/`.

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
native experts V6, native projections V4 and native decoder V4. Earlier alternatives are
preserved separately under `candidates/` and immutable `experiments/` snapshots.
Score the exact source associated with a passing check; do not substitute an
unvalidated active candidate. The kernel phase has both November 3 and November
4 available before the cutoff; preserve time to prepare the separate serving
entry, which does not require a qualifying kernel rank.

The bounded Modal runner is documented in [tools/MODAL.md](tools/MODAL.md).
It freezes exact inputs and checks public-reference correctness plus paired
CUDA-graph timing; results are explicitly unofficial. Its lifecycle tests pass,
and the H200 smoke has now passed. Kernel benchmark validation follows separately.
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

1. Use the completed native kernel checks, scored results and profiles as the
   starting evidence. All four tested sources are active and published.
2. Adapt useful improvements to serving's BF16 boundaries and concurrent
   execution. Kernel success does not establish full-service correctness.
3. Transfer validated improvements into the pinned serving source. Export a
   legal patch with `okc serve diff serving`.
4. From November 10, validate the full 8×H200 service on the hidden conversation
   workload, including output agreement and time to first token. Optimize the
   throughput/interactivity curve at 1, 4, 16 and 32 concurrent users.

Kernel submissions close **November 5 at 2:59 am Eastern** (November 4 at
11:59 pm Pacific). Serving now opens **November 10 at noon Eastern**. The live site
says serving closes November 19 Pacific, while the legal terms say November 18;
target a finished entry by **November 18** until the organizer resolves this.

[NOTES.md](NOTES.md) records rules, scoring, source links, limits and the date
discrepancy. The top 50 on the overall kernel board get serving-queue priority;
eligible entrants may participate in the serving track independently, without
a qualifying kernel rank (official terms section 6).
