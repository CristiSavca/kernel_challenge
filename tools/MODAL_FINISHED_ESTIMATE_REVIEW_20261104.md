# Finished-run estimate implementation review

This implementation was frozen with `REVIEWED_ESTIMATE_SHA256=None` and then
activated by root after independent review and 44 root accounting tests.
No provider call, account change or allocation was performed during implementation. Existing grant, overlap certificate,
provider evidence, manifests, sources and the eleven-record journal are intact.

Only the fixed original nine H200 runs can receive the $9.51 correction. The
per-run revised estimates total $11.63 and use the pinned provider lifetime,
rounding upward, an additional 60 seconds, existing resource rates plus 20%,
and the full $0.50 build allowance. Original journal debits remain immutable;
the two recent reservations and every later reservation remain fully charged.

The fixed correction is applied once to the current total, alongside the
unchanged $3.74 posted-overlap correction. Fresh provider usage and previous
receipt high-waters remain counted; pinned post-H200 evidence also enforces
P ≥ $4.16. Current conditional allowance is $9.75 before a future reservation,
subject to all fresh checks. A 600-second CPU reservation would leave $8.97 if
provider usage has not increased. No certificate is generated or expanded
automatically, and this revision is explicitly not final settlement or a
provider-enforced cap.

The implementation validates the reviewed plan/evidence hashes, fixed
parameters, original run/manifest/worker bindings, one exact resource-allocation
AST per driver, authoritative sandbox/app/task/lifetime mapping, unchanged
eleven-record journal prefix, and the two excluded recent reservations.
Before reserving it retains the original provider checks and additionally
reads a full-cycle hourly report through the latest complete hour. Missing,
duplicate, downward-revised or unreconciled rows fail closed; total attributed
cost for each revised app must remain within its revised estimate, including
charges in hours beyond the old overlap report's endpoint.

Validation: 11 new end-to-end/fault tests, 23 existing overlap tests and 10 CPU
budget tests pass. The new tests exercise fixed-once accounting, later full
reservations, higher/lower provider observations, a late 15:00 charge above U,
full-report failures, byte/hash corruption, resource AST faults, altered scope
and allowances, prefix/conflicting-certificate changes, crash retention and
the same cross-resource lock. All journals were temporary fixtures and every
provider call was mocked. The old overlap fixture explicitly disables this
new policy so it continues to test the original nine-record policy after
future activation.

The 60-second teardown allowance and $0.50 build allowance are conservative
estimates, not newly proved upper bounds. This is the reviewed policy's stated
assumption. Fresh checks detect contradictions before further allocation but
cannot guarantee that delayed billing will never exceed an estimate.

Files: `tools/modal_credit_budget.py`,
`tools/modal_credit_finished_estimate_20261104.json`,
`tools/test_modal_finished_estimate.py`, and the isolated fixture adjustment
in `tools/test_modal_credit_overlap.py`. Exact hashes and immutable-evidence
verification are in
`experiments/iterations/20261104-modal-finished-estimate-cpu.json`.

Root activated the exact certificate after independent and root review on
November 4, 2026. The activation changes only its fixed pin and comment; active
guard SHA `026499524dc66d158f44393edec15f45738cc9d7c04a7d83120f96198f494aa8`.
The disabled-state description above records the implementation review stage.
