# Fixed serving-finalizer stage diagnostic — 2026-11-04

Purpose: explain the retained H511 cancellation discrepancy, not validate the
whole helper or change its acceptance metric. Candidate91e6173b, original9b816e97
and reference worker c94c3134 remain unchanged. Four uploaded files only;
H200/300s outer limit/240s worker budget; no model, tuning or timings.

Reconstruct `finite_cancellation` through the pinned reference's `make_inputs`
(T3/K3/H511, seed20268578). Retain exact rounded inputs, hashes and raw prefixes.
Reproduce all three comparisons and the known absolute residual. If reproduction
differs, stop inconclusive. Capture original masked/preweighted FP32 values,
squares, actual mean, epsilon addition, inverse, normalized/scored routes and
route sum; separately capture the oracle's actual per-slot sum/H. Invoke the
existing candidate inverse and combine kernels with unchanged launch parameters.
Require bitwise output recomposition equality for all three implementations
before proceeding. Staging can change allocation/alignment, so a failed equality
is an inconclusive diagnostic, not evidence about the helper's internal stage.

Record all nine original/oracle variance and inverse FP32 bits and actual
candidate inverse bits. Candidate variance is not directly exposed and must not
be invented from its inverse. Record shapes, strides, row-address alignment,
output FP64 norms and at most16 residual coordinates. Stage hashes/symmetry
checks replace full tensor dumps.

Layout controls operate on the **actual squared FP32 reduction operand**, not
on x before `.contiguous()` and allocating Torch operations: same logical H511
values in contiguous rows,512-element physical pitch, and separate aligned row
copies. Record equality, mean/inverse bits and operand metadata. H512 is one
zero-padded helper control using the same rounded inputs, weight suffix1 and
unchanged scores/IDs; denominator512 differs from511. It cannot establish a safe
aligned production domain. FP64 references on BF16-rounded inputs and separately
on already-rounded FP32 products are explanation only, not replacement arithmetic.

Terminal `diagnostic_complete` requires exact reproduction, all three exact
recompositions, three complete same-value layout records, H512 control, FP64
references and unchanged input hashes. Numerical disagreement between production
helpers remains a reported failure of generic validation. Timeout, missing stage
or reconstitution mismatch is incomplete/failed, never PASS. JSON/checkpoints
are fixed-path `/tmp/okc-result.json`, at most1MiB, and failure stdout is compact.
Root alone integrates the runner, reviews cost and authorizes any execution.
