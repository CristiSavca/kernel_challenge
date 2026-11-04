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
