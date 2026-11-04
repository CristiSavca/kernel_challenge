# Independent finalizer stage result — November 4, 2026

**The diagnostic completed and supports a layout-dependent difference in the original normalization reduction. Generic validation remains FAILED.** This is not evidence of a candidate-only kernel bug, a new speedup or production acceptance.

Run `20261104T185116Z-a8743472` used exact helper91e6173b, original9b816e97, reference worker c94c3134, stage worker368af79c and runner690d4a46. I verified all five diagnostic source hashes, the request/result identities, all16 scalar records (144 values) by decoding their raw IEEE bits and hashes, the fixed result-reader requirements and comparison equality with the prior checkpoint. All three decomposed calculations reproduce their unchanged helper outputs **bitwise**. Inputs remain unchanged. The manifest records exit0, retrieved result, diagnostic_complete and sandbox termination. Worker elapsed time is5.4235seconds; this is diagnostic runtime, not a kernel benchmark.

## What changed between the opposing routes

The failed case is T3/K3/H511, seed20268578. Opposing inputs and weighted values are exact sign reversals; their squared values are bit-identical. For the third token, the original vectorized mean nevertheless returns adjacent FP32 values:

| Calculation, third token slots0/1 | Variance bits | Inverse bits |
| --- | --- | --- |
| Original contiguous H511 | `3f9217cb / 3f9217ca` | `3f6f9fbf / 3f6f9fc0` |
| Same H511 values, physical pitch512 | `3f9217cb / 3f9217cb` | `3f6f9fbf / 3f6f9fbf` |
| Same H511 values, separate aligned rows | `3f9217cb / 3f9217cb` | `3f6f9fbf / 3f6f9fbf` |
| FP32 slot oracle | `3f9217cb / 3f9217cb` | `3f6f9fbf / 3f6f9fbf` |
| Candidate | Unobserved | `3f6f9fbf / 3f6f9fbf` |

Original variances are1.1413511037826538 and1.1413509845733643; inverses are0.9360312819480896 and0.9360313415527344. Each pair differs by one ULP. Epsilon addition does not remove the variance difference.

This conclusion does **not** rely on changing logical H to512. The contiguous and padded-pitch controls both have shape3×3×511 and exactly the same squared-input byte hash. Their physical strides are1533/511/1 and1536/512/1. Contiguous row addresses vary modulo16; padded and separately cloned rows are all256-byte aligned. Both same-H controls remove the unequal inverse pair. This demonstrates dependence on the reduction operand's physical layout in the tested environment. It does not isolate an exact machine instruction, prove alignment is the sole cause, or exclude a different reduction kernel/dispatch choice under those layouts.

## Where the residual appears

The original normalized opposing-route pair already has381 nonzero values, all in token2, maximum absolute value2.384185791015625e-7 and L2 norm1.582823078587789e-6. Its FP32 byte hash equals the final FP32 route-sum hash. Thus the final sum alone did not introduce this observed residual. The staged-original output also matches the unmodified original helper bitwise, which rules out the tested instrumentation changing its final result.

Candidate and FP32-oracle outputs are all positive-zero BF16 bytes: their hashes independently match the corresponding zero buffer. Both explanatory FP64 paths likewise yield zero and equal scalar hashes. The original produces the same residual as the prior run. The fixed candidate/original relative-L2 error remains1; original/oracle remains1582823.078587789 with the denominator floor. No tolerance has changed.

A crucial limit: the candidate's token0 inverse is `3f746984`, while the oracle's is `3f746985`. Each implementation has equal inverse values within its opposing pair and therefore cancels. Equal final zero outputs do not prove equal intermediates or universal interchangeability. Candidate variance was not exposed and is not inferred by inverting rsqrt.

The zero-padded H512 helper control also yields all-zero outputs for all three implementations. It changes the normalization denominator and potentially dispatch as well as physical layout, so it is corroborating evidence only—not proof of a universally safe aligned production domain.

## Decision and next useful work

Preserve the failed generic result and the earlier official H4096/K8 stock-component speedup as separate evidence. This diagnostic adds **0% direct performance gain** and completes none of the15 remaining numerical cases or stream/graph checks. It supports a numerical explanation; it does not authorize promotion.

The most useful next serving step is a separately reviewed **coverage-only** protocol for those unexecuted checks using unchanged helpers. It must retain this exact known failure, preserve every threshold and report coverage completion separately from generic PASS. The strongest objection is that even complete remaining coverage cannot waive the failed original-compatibility criterion or establish full-model/serving agreement. Those remain explicit downstream decisions and measurements.

Two alternatives should remain conditional. A narrower production-domain specialization with ordinary metadata fallback could be reasonable if deployment scope warrants it, but this one aligned control cannot establish that domain. Reproducing the original eager reduction order could target a strict value-compatibility contract, but it would deliberately reproduce an alignment-dependent residual and may erase performance gains. The current evidence does not require forcing production to match that artifact, and it does not justify relaxing the acceptance threshold either.

Full source/evidence pins, all decoded scalars and outcome limits are retained in `experiments/iterations/20261104-finalizer-stage-result-critic.json`. This reviewer made no GPU, provider or official calls and changed no helper or worker source.

## Read-only post-run spend-floor audit

Root's later runner change raises the saved observation from4.17081442 to4.18466342, using billing-after evidence SHA `5707549ecaa30578ff1302dfea155b79332eb8c87f9d2605ac7cc2b5991b156b`. I independently ran only the saved-evidence reader and verified that reversing its three path/hash/amount literals reconstructs the exact runner in commit97e1af5 (`690d4a46…`). Current runner is `60f311f2…`; the executed run's source remains690d4a46. No financial formula changed. The billing snapshot reports billed0 and all three listed apps stopped with zero tasks.

The current journal preserves the exact14-record byte prefix from97e1af5 and appends only this stage run's full1.14 reservation. There are15 records totaling29.82. Grant, guard and both correction certificates are byte-identical to that commit. Decimal recomputation gives `30 − 5 − 4.18466342 − 29.82 + 3.74 + 9.51 = 4.24533658` working estimate, with the5 headroom separate. This is an unchanged conservative estimate, not a provider cap or fresh allocation approval; every future allocation still requires the full existing guard. No journal, certificate, account or allocation was modified by this review.
