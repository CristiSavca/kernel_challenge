# Independent finalizer validation result — November 4, 2026

**The fixed generic validation failed. The recovered evidence does not establish a candidate-only arithmetic defect.** Run `20261104T183437Z-9826c968` retained75 numerical records:74 pass and the declared `finite_cancellation` case fails. All six recorded activation checks pass. Fifteen later numerical cases, the two-stream checks and the two-graph checks were never reached. There is no general-helper or serving promotion from this result.

I independently matched the candidate `91e6173b…`, original `9b816e97…`, validation worker `c94c3134…`, request and90-case manifest to the first failed run. These bytes and thresholds are unchanged. The repaired run used runner `6f3f61a4…`; the recovered checkpoint SHA is `e5ba29486796ba4d01c92388b3437c680c73358bb0e3bbdeee98d42e9052be95`. Every retained descriptor is the exact first75 entries of the frozen manifest. The saved environment is H200, Torch2.14.1+cu130, Triton3.8.0 and CUDA13.0. This is a component diagnostic, not an official scoring run.

## Recovered numerical attribution

The finite cancellation case is T3/K3/H511, seed20268578, BF16 inputs and norm weights, FP32 scores, all-local expert IDs, weight multiplication before normalization (`weight_first=true`), and epsilon1e-8. The second slot is the exact sign reversal of the first before BF16 conversion, the third slot is zero, and all route scores are1.

| Recorded comparison | Maximum absolute difference | Whole relative L2 | Worst-row relative L2 | Pass |
| --- | ---: | ---: | ---: | --- |
| Candidate vs FP32 oracle |0|0|0|Yes|
| Candidate vs original |2.384185791015625e-7|1|1|No|
| Original vs FP32 oracle |2.384185791015625e-7|1582823.078587789|1582823.078587789|No|

All outputs in these comparisons are finite. Candidate/oracle also records `equal_values=true`; that establishes numerical value equality under the recorded comparison, not signed-zero/bitwise identity. The original disagrees with both. The absolute residual is small, but it fails the predeclared relative metric and its1e-12 denominator floor. No tolerance has changed and the case is not excluded after observing failure. Agreement with the independent FP32 oracle prevents labeling this evidence a candidate-only arithmetic defect; it does not waive the original-compatibility requirement or prove a universal numerical contract.

Before this case, the maximum candidate/original whole and worst-row relative errors were4.530630184547198e-5 and0.00024744065727820764. Those74 passing cases are useful retained evidence, not a replacement for the unexecuted checks. Actual positive-duration GPU profiles establish the two new kernels on boundaryT31, the original kernel onT32/T33, and eager fallback on FP32/K33/H16385. The cancellation call was not itself separately profiled; its fast-path selection follows the pinned helper and its input metadata.

The checkpoint contains final output comparison metrics, not the intermediate variance/inverse values or raw output tensors. It therefore does not identify the stage causing the difference. Original vectorized Torch means, per-slot oracle reductions, candidate reductions and final route summation use different floating-point layouts. Alignment-dependent ragged-row reduction is a concrete hypothesis, not a demonstrated PyTorch defect.

## Recovery and accounting evidence

The manifest records failure/exit1, recovery inside the live app context, successful checkpoint recovery and sandbox termination. This validates the intended recovery path for this observed failure; it is not a claim that every network failure is recoverable. The first failed run and its lost-checkpoint limitation remain preserved.

A subsequent root edit only changes the runner's saved billing path, digest and expected observation from4.16434237 to4.17000000. I inspected this three-literal diff and independently called the saved-evidence reader without any provider request. Evidence digest `567c6247…` matches, billed cost is0, and both listed apps are stopped with zero tasks. This increases the conservative debit. It does not alter the frozen run6f3f61a4, old receipts or journal entries. Current runner digest after that separate change is `649db776…`.

## Bounded next diagnostic proposal

Do not repeat the same validation and hope for a different outcome. A separately reviewed stage diagnostic can test one cause with the exact failed tensor and one deterministic H512 zero-padded paired-route alignment control. Keep both helpers unchanged and retain the original thresholds/status.

Capture the nine per-route variance/inverse scalars from the original vectorized expressions, the independent slot expressions and the existing candidate inverse kernel. Record equality of opposing weighted inputs/squares, normalized route pairs, output norms and a bounded set of residual coordinates. Verify that each staged recomposition equals the corresponding unchanged helper output before attributing a discrepancy; record shape, stride and storage alignment because staging can alter allocation behavior. A failed reproduction makes the diagnostic inconclusive.

The falsifiable hypothesis is unequal original means/inverses for opposite H511 rows while candidate and slot-oracle inverse pairs remain equal. If the original inverse pairs are equal, investigate the final route combination instead. H512 is an alignment/control experiment, not evidence that arbitrary aligned shapes are universally safe. Two tiny tensors, no weights/model, no autotuning and no performance measurement fit a bounded diagnostic; a separately authorized outer limit of at most300 seconds is sufficient as a requested cap, not a runtime guarantee. This proposal grants no allocation and cannot produce generic-validation PASS by itself.

Full evidence pins and all comparison records are in `experiments/iterations/20261104-finalizer-validation-result-critic.json`. This reviewer made no GPU, provider or official calls and edited no helper, worker or runner source.
