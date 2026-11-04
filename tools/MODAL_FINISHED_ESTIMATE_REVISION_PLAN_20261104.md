# Fixed revision of nine completed-run estimates

November 4, 2026. Plan only: no guard, certificate, grant, journal or account
change, and no allocation. This is a manually reviewed **conservative estimate
revision**, not a provider-proved maximum or a statement of final settlement.
The original reservations were also estimates. The full $5 headroom remains.

## Fixed arithmetic and scope

For only the nine runs already covered by the posted-overlap certificate:

`U_i = min(R_i, ceil_cent(7.68 × (ceil_seconds(finished_at − created_at) + 60) / 3600 + 0.50))`

The $7.68/hour preserves the original rounded H200/4-CPU/32-GiB rates and 20%
margin. The extra 60 seconds covers termination/metadata uncertainty as an
explicit allowance, not a measured upper limit. Every run retains its full
$0.50 image-build allowance. The fixed per-run amounts, in certificate order,
are $1.93, $0.99, $0.96, $1.47, $1.78, $1.54, $1.13, $0.92 and $0.91.
Their total is **$11.63**, versus original reservations of **$21.14**.

A new immutable, append-only certificate could authorize exactly one fixed
estimate adjustment `E = Σ(R_i − U_i) = $9.51`. Leave every original journal
record verbatim. Do not add E repeatedly from subsequent receipts:

`debit = Pmax + Σ(all recorded original and future R) − existing A − E`

Keep existing posted overlap `A=$3.74` separate. Require `0 ≤ A_i ≤ U_i ≤ R_i`.
The newer H200 run `20261104T150731Z-ef5e7aac` retains its full $2.42 and the
failed CPU run `20261104T153757Z-d116f3fe` its full $0.78. Neither is eligible
for this revision, regardless of its short execution or result.

At current total R=$24.34 and **Pmax≥$4.16**, conditional allowance is
`30−5−4.16−24.34+3.74+9.51 = $9.75`. One later CPU600 reservation would leave
$8.97 before additional observed usage. These are plan figures, not currently
available allowance. Without revision, preserving $4.16 leaves $0.24. The lower
$4.10 reading must not turn that into $0.30.

## Evidence and correction of the earlier review

Provider metadata binds each original sandbox to its App ID, task ID, H200 type
and creation/start/finish times. All nine are terminal and their manifests
record termination. Each manifest pins its full original `worker.py` driver.
I independently extracted every driver's one `Sandbox.create` AST: all request
H200, CPU `(4,4)`, memory `(32768,32768)`, and blocked networking.

**Correction:** the earlier lifetime critic incorrectly described the original
host allocation code as unpinned. The saved workers include that host code.
Eight exact driver hashes also match Git history; one is a unique saved
working-tree version whose pinned full source is still available. Historical
commit timing is supplementary, not a substitute for those exact snapshots.

The evidence JSON pins metadata, all manifests/drivers, current eleven-record
journal, existing certificate/grant/provider evidence, and the actual $4.16
`20261104T150731Z-ef5e7aac/billing-after.json` (SHA `c6b1f62e…`). All saved
per-app costs fit below their proposed U; that does not prove later costs will.

## Required future implementation gates

A separately authorized implementation would default disabled until its exact
certificate hash is reviewed. Bind the grant/cycle/profile/environment, the
existing overlap certificate, the nine revised identities and original record
indices, the reviewed eleven-record prefix, all original snapshots and the
$4.16 high-water evidence. Validate the formula and exact totals; no caller
rate overrides, automatic new certificates, repeated reductions or refunds.
New and unmatched reservations keep their full recorded R under the same lock.

Before any allocation, preserve all current profile/apps/summary/overlap checks.
Also fetch a fresh full-cycle hourly resource report through the most recent
complete hour. Canonicalize object/environment/interval/resource identities;
reject duplicates, missing old attribution, downward revision, and every
revised App ID whose **total** attributed cost exceeds its U. Reconcile report
cost inclusion against the current summary with only the existing rounding
tolerance. Keep `Pmax=max(fresh observed, all historical receipt observations,
existing $3.90423640 evidence, pinned $4.16)` even when usage drops. A fresh report
cannot certify unposted costs; this gate detects contradictions before further
allocation. It does not enlarge E when observed costs are lower.

Tests should cover fixed-once E with unchanged A, both excluded recent runs,
future full debits, high-water decline, delayed cost above U, wrong metadata or
resource AST, mutated prefix/certificate/source, duplicate revised IDs,
crash-retained reservations and the existing cross-resource allocation lock.

## Concrete failure conditions and remaining uncertainty

The recorded downward billing revisions are real, so preserving the $4.16
high-water is essential. Task completion is not an explicit billing-stop
contract; cleanup exceeding the added 60 seconds could understate runtime.
Pre-creation image work could exceed $0.50. Multiple containers or changed
resource limits associated with a supposedly unique App ID would invalidate
this estimate. Full report attribution and the frozen one-allocation drivers
help detect these issues but do not make the allowance a provider cap.

No such excess is demonstrated in the saved original-nine evidence. Final
settlement is not required merely because posting is delayed: under the
reviewed estimate, residual `U−A` and headroom continue covering outstanding
exposure. The decision is whether to accept this tighter estimate with stated
margins, not whether to pretend the bill is final. I recommend root review of
this fixed plan; implementation and allocation remain separate decisions.

Exact arithmetic, source/resource audit and hashes:
`experiments/iterations/20261104-modal-finished-estimate-revision-plan.json`.
