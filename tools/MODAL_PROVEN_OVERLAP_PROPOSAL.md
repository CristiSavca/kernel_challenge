# Modal accounting correction: proved overlap only

November 4, 2026. **A conditional accounting correction is defensible; the
available evidence does not yet authorize a positive correction or allocation.**
This is a proposal only. Guard, grant, reservation journal, manifests and lock
remain unchanged. Existing free-credit authorization does not need to be
re-requested merely to review an implementation; evidence gates below are
technical requirements, not an invented user-approval requirement.

The nine reservations total $21.14. The original $30 grant and $5 headroom stay
fixed. Root's 11:08 UTC read reports $3.79 gross usage and $0 billed, versus
$3.90423640 in retained 06:30 evidence. The decline is not proof of restored
credits. Current code takes the maximum of grant/journal observations and its
fresh query, not every saved `billing-after.json`: with this fresh query it would
leave $0.07, still below its minimum $0.63 reservation. A reviewed correction
should additionally preserve the known historical $3.90423640 high-water mark;
with that value the current double-counted allowance is −$0.04423640.

## One bounded policy

Keep every original reservation `R_i` permanently. Let `A_i` be only the portion
of an already posted provider charge that is independently proved to belong to
that exact reserved run **and to be included in the same-cycle gross-usage
high-water value `P`**. Require `0 ≤ A_i ≤ R_i`, no duplicate attribution and
`ΣA_i ≤ P`. The proposed accounted debit is:

`D = P + Σ(R_i − A_i)`; `available = 30 − 5 − D`.

This changes double counting only. It never releases the unused difference
between an estimate and a final bill. Algebraically `D ≥ ΣR_i`; all original
reserved dollars remain accounted for. Unmatched usage stays in `P`, unmatched
reservations stay whole, and delayed charges remain covered up to the original
reservation estimate. This does **not** cure a run whose true cost exceeds its
reservation. It provides no provider-enforced spending cap.

For example, a 300-second reservation needs at least $1.18423640 of proved
overlap; a 600-second reservation needs $1.82423640; 900 seconds needs
$2.46423640. Even complete attribution of the observed post-grant project usage
would leave at most $3.85723137 working allowance under this policy, not the
nominal $26.21 account credit figure. These are conditional arithmetic examples,
not available credits. No offset is currently established.

## Required evidence and source binding

1. Preserve the grant and original journal verbatim. A separately reviewed,
   append-only correction certificate would bind their hashes and the exact
   nine-record prefix, original guard version, each run/manifest/source hash,
   terminated sandbox ID, and authoritative provider object mapping. Future
   reservations still debit their full amount; there is no automatic settlement
   loop or reset. The existing lock inode and crash-before-allocation reservation
   discipline remain in force.
2. Root obtains provider records without upgrading the plan or spending money.
   Match object ID, workspace, environment, billing cycle, full interval and
   resource cost basis. Saved manifests have sandbox IDs but no explicit App ID;
   shared app names and nearby timestamps are not a sufficient mapping. Establish
   inclusion in `P`, not merely the numerical inequality `ΣA_i ≤ P`. An aggregate
   increase, disappearance of an app, or repeated stable totals cannot prove it.
3. Use one canonical set of nonoverlapping object/interval/resource entries.
   Never add repeated snapshots, hourly and daily views of the same usage, or a
   row's total and its resource breakdown. Exclude credits, taxes, unrelated
   usage, the pre-grant smoke, and unbound image-build costs from offsets. Keep
   all such unmatched amounts charged conservatively. Provider revisions or
   contradictory attribution block further correction/allocation pending review;
   lower aggregate readings never reduce `P`.
4. A future implementation would validate certificate integrity and absence of
   reused line items, preserve old missing-journal/termination/profile checks,
   and fail closed on unknown evidence. CPU tests must cover duplicate/overlapping
   reports, downward provider revisions, unrelated usage, missing source bindings,
   corrupted journal prefixes, concurrent allocators and crash persistence.
   This proposal authorizes neither that implementation nor an allocation.

## Present evidence limits and next useful read

Modal says granular billing exports are available to Team/Enterprise workspaces
and warns that collection can be delayed; no fixed settlement deadline is
promised. Reports cover complete intervals, not an exact run's lifetime.
The installed client exposes object/interval/cost/resource fields but no immutable
charge ID, aggregate-inclusion token or settled-through watermark. A frozen
Python dataclass is not proof that the provider's billing is final.
Sources: [billing guide](https://modal.com/docs/guide/billing),
[report API](https://modal.com/docs/sdk/py/latest/Workspace#billingreport).

Consequently, clearing reservations solely because all nine sandboxes terminated
is **NO GO**. The useful next read is whether the current Starter account's
existing UI/API can supply authoritative object-cost mapping and inclusion
evidence without an upgrade. If it cannot, leave every `A_i=0` and stop this
route. No guessed settlement delay or aggregate-delta substitute is proposed.

Fresh `main` app listing alone also cannot exclude other environments, clients
or hosts. The local lock cannot prevent their concurrent spending. A no-paid-
charges claim therefore needs verified provider enforcement: Modal documents a
separate net spend limit from its pre-credit usage budget, and says remaining
credit-funded work may continue after the net limit is reached. Read the actual
workspace limits; do not assume the default is zero or change them in this task.
Source: [provider budgets and spend limits](https://modal.com/docs/guide/budgets).

Evidence, exact hashes and arithmetic:
`experiments/iterations/20261104-modal-proven-overlap-proposal.json`.
Independent critic accepted the conditional algebra and identified the same
missing attribution/inclusion evidence; that is not implementation approval.
