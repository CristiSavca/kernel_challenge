# Serving finalizer validation failure — 2026-11-04

Run `experiments/modal/20261104T182526Z-3d547341` exited1 at `finite_cancellation`. The archived helper, original and validation-worker hashes independently match `91e6173b…`, `9b816e97…`, and `c94c3134…`. The sandbox is recorded terminated. **No result/checkpoint JSON was recovered.** The manifest records `validation_checkpoint_recovered=false` and `validation_retrieval_error_type=NotFoundError`; only source files, request/manifest and stderr survive. Do not claim measured error magnitudes or identify a failed comparison pair from this evidence.

The exact trace reaches the worker's assertion after `_evaluate` appends a failed case. `_evaluate` independently checks candidate/original, candidate/FP32-oracle and original/FP32-oracle, and any failing pair triggers this same assertion. Candidate-only failure is therefore unestablished. The frozen case is zero-based index74 of90, so fixed control flow implies74 previous numerical cases and all six activation checks completed successfully before the failure. Six profiler start/stop pairs survive in stderr; detailed CUDA names/counts and individual metrics do not. The remaining15 numerical cases, two-stream checks and two-graph checks were never reached. This is incomplete generic validation, despite the prior passing official stock-Marlin component result.

## Exact cancellation construction and legality

Seed20268578; T3/K3/H511; CUDA BF16 x; FP32 scores; BF16 H-vector norm weight; all-local int32 IDs; pre-norm weight multiplication; epsilon1e-8; contiguous inputs. Before BF16 conversion, slot1 is assigned the negative of slot0, slot2 is zero, and all scores are exactly1. No poison or nonfinite value is introduced. The normalized positive/negative pair cancels in real arithmetic. Scores are not normalized to sum1, but the helper API imposes no such requirement and every involved value is finite; no fast-path metadata restriction excludes the case. It is inside the newly claimed generic small-helper domain. It is not itself the deployed Muse H4096/K8 distribution, and cannot establish full-model output failure. Conversely, excluding it after failure merely because it is adversarial would invalidate the declared component test.

## Strongest source-supported hypothesis, not a measured diagnosis

The candidate computes each route's variance in a separate Triton program, then accumulates slots sequentially in FP32. The original uses a Torch mean over the contiguous `[T,K,H]` tensor and a Torch slot reduction. The independent oracle computes each slot through a separate Torch `[T,H]` sum-of-squares and sequential slot accumulation. These are different floating-point reduction layouts, even though all preserve the prescribed BF16 input and FP32 arithmetic boundaries.

An almost-zero cancellation output magnifies tiny normalization differences under the fixed relative-L2 criterion. For example, a zero output against a nonzero reference residual has relative error1 regardless of that residual's small absolute size. Against an exactly zero oracle, the1e-12 denominator floor applies. This is an explanation of sensitivity, **not permission to change the tolerance**.

A particularly plausible cause is that Torch's ragged-H mean reduction treats row addresses/alignments differently: opposing rows of length511 need not execute an identical accumulation order, while independent route programs or separately allocated slot tensors may. Equal squared values alone do not prove equal FP32 reductions across these implementations. The kernel source suggests candidate opposite rows should produce matching variances, but no compiled instruction trace or stage outputs were saved. Alignment-dependent Torch reduction behavior remains a hypothesis, not a demonstrated defect or a reason to blame the oracle.

Falsifiable attribution after the exact checkpoint is obtained:

| Failed pairs | Interpretation requiring further numerical inspection |
| --- | --- |
| candidate/original and original/oracle; candidate/oracle passes | Original differs from the route-wise zero-cancellation result; candidate-only defect is not established. |
| candidate/original and candidate/oracle; original/oracle passes | Candidate arithmetic/layout is the primary suspect. |
| original/oracle fails even though candidate/original passes | The independent oracle disagrees with the original; preserving the original alone did not satisfy the predeclared cross-check. |
| All pairs fail | More than one reduction difference or another defect; no single attribution. |

The next diagnostic must keep helpers, input seed/case and thresholds frozen and recover the existing three comparison records. That alone distinguishes these possibilities and reports absolute versus relative error. Only then should a separately reviewed diagnostic inspect per-route squared sums/inverses and compare H511 with an aligned control to test the alignment hypothesis. Such additional GPU work is not performed or authorized by this note. A production correction must preserve the original numerical contract or select the original fallback through a defensible supported-domain boundary; do not inspect input values on the host, cache outputs or weaken the acceptance metric. No particular helper fix is yet substantiated by the lost evidence.

## Confirmed observability bug and narrow fix

The runner's exception handler attempted checkpoint retrieval **after** leaving `app.run`. That context can stop the ephemeral application and remove the sandbox before recovery. The surviving `NotFoundError` is consistent with this lifecycle ordering. The previous mocks incorrectly left the sandbox available after context exit, so they did not model the failure.

A new CPU test invalidates the mocked sandbox during app-context teardown. It failed on runner339a85ef with the event order `app_exit, checkpoint_read`. The narrow repair enters a finalizer-only recovery scope after `app.run`, so reverse-order context exit recovers first, then permits application teardown. It preserves the original worker exit/error, the fixed1MiB/10s read, source/coverage validation, separate checkpoint naming and final termination. It removes the finalizer's post-context recovery; no worker, helper, tolerance, allocation or financial-guard change is included.

After repair the regression test observes `checkpoint_read, app_exit` and retains the failed checkpoint. Focused runner13 tests, existing lifecycle18 tests and frozen-worker12 CPU tests pass. Frozen repaired runner SHA256 `6f3f61a480d0e48b298eea23b0f1b63b36bd2ccc0850ce775f890321a13b0f16`; focused tests `c97fc56960be97a0c36ecd845893790830be38ebd5ded66bb3371be7396038dd`. These are mocked lifecycle/source checks, not a new GPU result. Root owns any separately approved fixed diagnostic rerun and its full cost reservation.

## Recovered comparison evidence from the separate fixed rerun

The separately authorized run `experiments/modal/20261104T183437Z-9826c968` recovered `result-checkpoint.json` before application teardown. Checkpoint SHA256: `e5ba29486796ba4d01c92388b3437c680c73358bb0e3bbdeee98d42e9052be95`. It binds the same candidate `91e6173b…`, original `9b816e97…`, worker `c94c3134…`, and case manifest `a57d656ba2635c8e3c6d95c93351de0aa4508e2c3b292510108c73abc535ccb2`. Recorded environment: H200, Torch2.14.1+cu130, CUDA13.0, Triton3.8.0, Python3.12.3. The worker exited1 after12.953s; the manifest preserves that failure, records successful checkpoint recovery and sandbox termination. This evidence does not retroactively recover the first run's missing metrics.

For the identical `finite_cancellation` descriptor/seed above:

| Comparison | Exact output-value equality | Maximum absolute difference | Whole relative L2 | Worst row relative L2 | Fixed test |
| --- | --- | --- | --- | --- | --- |
| Candidate / FP32 oracle | yes | 0 | 0 | 0 | pass |
| Candidate / original | no | 2.384185791015625e-7 | 1 | 1 | fail |
| Original / FP32 oracle | no | 2.384185791015625e-7 | 1582823.078587789 | 1582823.078587789 | fail |

All three comparisons have zero NaN/+Inf/-Inf output and reference counts. Equality here is the worker's numerical output-value comparison; it does not distinguish signed-zero bits. The relative metric/floor and equality are consistent with zero candidate/oracle output and a tiny original residual; raw output bits/norms were not retained, so do not infer exact zero solely from aggregated metrics. The checkpoint contains no per-route intermediate values or output bit dump, and the exact location and signed values of the difference are not saved.

This resolves the pair attribution: **the original disagrees with both the candidate and the independent route-wise oracle. A candidate-only arithmetic defect is not established.** It does not establish that the original is defective, that the candidate preserves the original finite-precision result, or that a generic oracle is authoritative over the existing implementation contract. The required candidate/original comparison still fails at the unchanged0.015/0.075 thresholds. The earlier alignment/reduction hypothesis remains unproved.

The checkpoint records75 numerical cases, of which74 pass and this one fails. All six activation checks pass with positive-duration CUDA events: T31 has exactly one new inverse and one new combine kernel; T32/T33 each have one original large kernel; the three eager fallback cases have no helper kernel. The remaining15 numerical cases and both stream/graph suites were not reached. Generic validation is still failed/incomplete, and this result supplies no promotion or new performance evidence.

## Next-stage diagnostic plan only — localize the numerical divergence

Keep both helpers, the90-case worker and thresholds frozen. The independent reviewer has the same bounded proposal in [SERVING_FINALIZER_VALIDATION_RESULT_CRITIC_20261104.md](SERVING_FINALIZER_VALIDATION_RESULT_CRITIC_20261104.md). This is a plan, not authorization for another allocation or a generic-validation PASS.

1. Reconstruct the exact failed case with unchanged `make_inputs`, seed20268578 and RNG draw/conversion order. Hash the final BF16 x/weight, FP32 scores and IDs; reuse these rounded tensors throughout. First reproduce the existing comparison outcomes. A changed-shape random draw with the same seed is not an exact-input control.
2. Inspect the nine original and oracle per-route variances/inverses, using their exact Torch expressions, and invoke the existing candidate inverse kernel unchanged. Record opposite-route weighted-input/square equality, scalar FP32 bits, normalized route pairs and a bounded output residual dump. Distinguish the original `mean` from a diagnostic `sum/H`: `mean*H` is not a captured internal squared sum. Require staged recomposition to reproduce each unchanged helper output before attribution; staging can alter allocations. If original opposite-route means differ, asymmetry precedes rsqrt; if means/inverses match, investigate normalization/route combination instead. Counterfactual combination with another implementation's inverses must be labeled separately.
3. Test H511 reduction-layout sensitivity with the same logical squared values in original contiguous rows versus a512-element physical row pitch or aligned row copies. Record shape/stride/row-address residues; Torch may also change its reduction algorithm. An optional actualH512 helper control zero-appends to the same rounded inputs (new weight1), but changes the normalization denominator511→512, so it is a cancellation/control experiment, not isolation proof or evidence that all aligned shapes are safe.

FP64 on these same BF16-rounded inputs—and separately on already-roundedFP32 products—may explain the squared-sum/inverse residual. It cannot replace the FP32 contract or weaken acceptance. The alignment hypothesis is weakened if original inverse pairs match or no layout control changes the earliest discrepancy. No particular production fix is substantiated yet. A justified metadata fallback or numerical implementation change would need separate review and the unchanged complete validation, including the15 unrun cases, streams and graphs. No helper/worker edit, GPU run, tolerance adjustment or promotion was performed for this note.
