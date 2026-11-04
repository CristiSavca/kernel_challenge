# Remaining finalizer coverage — fixed plan, 2026-11-04

The stage result3ce10b10 supports an original Torch reduction-layout explanation
for the H511 cancellation residual. It does not waive the original/candidate
compatibility failure. This experiment has0% direct performance gain: it only
completes missing evidence for the unchanged helper91e6173b/original9b816e97.

Use the exact c94c3134 reference worker for `cases`, `make_inputs`, `_evaluate`,
all three comparisons and unchanged `_concurrency`/`_graphs`. Execute only
original indices75–89: four tiny/subnormal/large/zero cases and11 exceptional
cases. Preserve every descriptor/seed and0.015 whole/0.075 row tolerance,
0.1 row floor, FP64 metric accumulation and exact exceptional classifications.
Numerical case disagreements are recorded and the independent15-case loop
continues. Runtime/CUDA errors, input mutation or output/input storage aliasing
abort. Existing stream/graph functions retain their original failure behavior;
their exceptions abort with partial evidence rather than returning invented
coverage. No rerun of case74 or the earlier74 cases/profiles is implied.

Then run the original two streams/eight calls and two graphs/eight fresh
replays. Preserve producer dependencies, per-call scratch lifetime, retained
outputs, cross-graph allocation disjointness and fresh-input copies. No claims
of measured execution overlap, new latency results or full-model correctness.

Host build verifies retained checkpoint e5ba29486796ba4d01c92388b3437c680c73358bb0e3bbdeee98d42e9052be95
and stage result3ce10b107e0108b284be0ec7d185b6d7b3b8edfdba8d02046156e2cbcf6900cc.
Embed a compact constant of its known failed comparison values, explicitly
historical/not rerun; do not upload another receipt or import an unpinned helper.
Exactly four source files: candidate.py, baseline.py, validation_reference.py,
finalizer_coverage.py. Fixed image/H200,300s outer and240s worker budget,
bounded1MiB `/tmp/okc-result.json` checkpoint and compact failure stdout.

Falsifiable completion requires all15 exact records with independently
recomputed comparison outcomes, plus complete unchanged stream/graph records.
`coverage_complete` means only that coverage finished. A separate
`remaining_numerics_passed` and exact failed-case list expose any new failure.
`generic_validation_status` is always `failed_at_finite_cancellation`; neither
all remaining cases passing nor the explanatory stage result permits PASS or
promotion. Missing records, corrupted pins, runtime/ownership errors or a time
limit produce failed/incomplete status. No automatic retry or tolerance change.
Root alone owns runner integration, fresh credit review and GPU allocation.
