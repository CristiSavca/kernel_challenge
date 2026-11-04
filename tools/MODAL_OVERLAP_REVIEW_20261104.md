# Fixed posted-cost overlap certificate

The optional certificate in `modal_credit_overlap_20261104.json` is **inactive**:
`modal_credit_budget.REVIEWED_OVERLAP_SHA256` remains `None`. Independent review
must precede activation. No grant, reservation, manifest, lock or account setting
was changed, and this work allocates no GPU.

The certificate binds the original nine journal records, the unchanged grant,
their manifests and frozen uploaded sources/workers, original guard copies,
provider sandbox-to-App mappings, and full-cycle billing evidence. Each App's
posted cost is rounded down separately to cents: offsets sum to **$3.74**.
The original **$21.14** of reservations remains counted in full. The calculation
is always `30 - 5 - Pmax - sum(original and future reservations) + 3.74`.
It never adds offsets from previous receipts. Historical `Pmax = 3.90423640`
remains a floor; current billing and subsequent journal observations can raise it.

Before a reservation, the existing lock, current-profile, billing, stopped-app,
termination and journal checks still run. The optional path additionally queries
the certificate's exact November 1–4 14:00 UTC hourly billing interval. Missing
or downward-revised rows, duplicate intervals, broken attribution, source/hash
changes, costs above original reservations and inconsistent summary inclusion
fail closed. Larger observed costs never enlarge the fixed offset. All new
reservations retain their full original charge and are fsynced before allocation.

The 33 separately displayed report rows differ from the summary component by
`2e-8`; the validator permits at most one last-place decimal unit per row plus
one for the aggregate. Per-App cent floors leave a much larger retained margin.
No unused reservation is refunded and no final-settlement claim is made. The
original timeout/resource/build estimates remain estimates, not provider caps.

Focused CPU tests use copies of the pinned real evidence in temporary folders,
with all provider calls mocked. They cover future reservations and fixed-offset
reuse, revisions, missing mappings, corrupt evidence, duplicate intervals,
overcredit, crashes and exclusion of a second process by the unchanged lock.

```sh
.venv-modal/bin/python -m unittest discover -s tools -p test_modal_credit_overlap.py -v
.venv-modal/bin/python -m unittest discover -s tools -p test_modal_h200.py -v
```

Review evidence is in `experiments/iterations/20261104-modal-overlap-cpu.json`.
The root agent owns independent reruns, activation and any later allocation.

Root activation at 15:00 UTC: independent reviewer found no blocking accounting
issue and reproduced all 41 CPU tests. Root also reproduced them and reviewed
the implementation/certificate. The exact certificate hash is now pinned in the
guard. The inactive description above records the implementation handoff.
This activation creates no allocation; fresh provider/cleanup checks still gate
each later reservation. Root evidence: experiments/iterations/20261104-modal-overlap-root-review.json.
