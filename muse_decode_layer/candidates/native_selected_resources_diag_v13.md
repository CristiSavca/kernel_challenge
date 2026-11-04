# Selected-resource diagnostic V13

Frozen source SHA-256:
`e444f46f945e03bb42dab8d80ed4869f8a7da6464f845ac09ac3cf8ea19473fd`.
Parent: scored decoder
`97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
Created November 4, 2026. Independent final source review approved this exact
hash for one diagnostic-only private check. Root independently reproduced the
combined CPU report (SHA `c255633e96db54e98df4717c2f51783e0a54cabf1b5853d7ab2fc681f86d1c77`).
Root submitted the exact source as private check `c0b4201c5ad345d1` at10:49 UTC.
The service received it; compilation, correctness and diagnostic coverage are
pending. Resume that receipt without another upload. The author made no GPU
allocation or official submission.

This is an unscored resource diagnostic, not a performance candidate. It adds
host-only queries on first actual post-prepare SK2 and small-MoE dispatches.
Reporting is disabled throughout preparation/tuning, including on reused
extension modules. Existing device code, packing, flags, tuning, launch arguments,
fusion and PDL computation are unchanged. Cache names and diagnostic bindings
are new. Queries do not add CUDA graph nodes; their host overhead makes the
check unsuitable as a performance improvement claim.

SK2 records the exact template, physical M/N/K, pmax, UV mode, PDL, grid,
threads, dynamic shared memory, compiled registers/local/static shared,
opt-in maximum, block capacity and each query status. For CL>1 the capacity
is block-only; cluster residency remains unknown. Small-MoE records effective
compiled macros and the first sample for one resource identity per module.
The total diagnostic cap is 10,240 bytes, with bounded storage and explicit
partial-coverage STOP records. Generation is an arming epoch, not a weight ID.

CPU validation restores **all parent bytes** by removing the explicit host
patches. Generated SM90/non-SM90 SK2 and all five embedded CUDA strings also
match the parent after host-only normalization. Exact host snippets compile
with mock CUDA APIs; tests cover disabled/tuning state, rearming, deduplication,
signature changes, failures, exhaustion, registry bounds and expert concurrent
callers. These are host/source checks, not CUDA compilation or numeric proof.

Plan: `tools/DECODER_SELECTED_RESOURCE_DIAGNOSTIC_PLAN.md`.
Combined proof: `tools/test_decoder_resource_diagnostic_cpu.py`.
Proof record: `experiments/iterations/20261104-decoder-selected-resource-cpu.json`.
Build recipe: `tools/build_decoder_resource_diagnostic.py`; separate expert
fragment: `tools/decoder_moe_resource_diag_fragment.py`.

Success means enough valid records to resolve selected resource constraints.
There is no speed threshold. Errors, missing signatures or log truncation remain
unknowns and do not authorize a retry. Attributes do not establish achieved
occupancy, SASS serialization, spill traffic, stalls or optimization payoff.
Never publish or promote this diagnostic source based on its score.
