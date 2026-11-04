# Selected-resource diagnostic V13

Frozen source SHA-256:
`e444f46f945e03bb42dab8d80ed4869f8a7da6464f845ac09ac3cf8ea19473fd`.
Parent: scored decoder
`97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
Created November 4, 2026. Independent final source review approved this exact
hash for one diagnostic-only private check. Root independently reproduced the
combined CPU report (SHA `c255633e96db54e98df4717c2f51783e0a54cabf1b5853d7ab2fc681f86d1c77`).
Root submitted the exact source as private check `c0b4201c5ad345d1` at10:49 UTC.
The check finished successfully, confirmed at11:04 UTC:13 correctness tests,
12 timed cases and12 profiles, with no flags or rejected checks. The author
made no GPU allocation or official submission. Do not upload this source again.

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

The retained log contains19 SK2 and3 small-MoE resource records, all at arming
epoch1. Every query succeeded; local-memory allocation is zero in every record.
SK2 registers range from71 to116. QKV and pooler report one block/SM capacity;
the observed output projections and T16 boundary-unpooler report two. The
cluster router's value is block-only capacity. The three small-MoE builds report
144/144/148 registers,214784 dynamic shared bytes and one block/SM capacity.
Pooler fusion uses384 static shared bytes; other retained records report zero.

No STOP, malformed record, missing schema or conflicting resource identity was
retained. The saved3455-byte log is an exact prefix of the5411-character receipt
tail; its suffix is24 profiler USDT lines. Thirty SK2 profile rows each have one
compatible retained record, but truncated names and runtime fields still prevent
claiming an exact benchmark invocation identity. MoE profile names do not identify
their compiled flag builds. The unexpected-local-allocation hypothesis is closed
negatively; no bottleneck or automatic optimization follows from these facts.

Offline parser: `tools/analyze_decoder_selected_resources.py` (SHA
`cc967b3a9d3052ea4c8aa9d65e03e78aa624f8b38d0ac189e5d0ec97f7c839f7`).
All14 parser fixtures pass, including conflicting identities and failed-run
eligibility. Root reproduced the saved analysis exactly and separately audited
raw resource lines, official correctness and source/profile/log provenance.
Evidence: `experiments/iterations/20261104-decoder-selected-resource-analysis.json`
and `20261104-decoder-selected-resource-root-result.json`.
