# Independent review of the additional Modal reconciliation

**GO for a separately reviewed fixed-policy implementation; no activation or allocation approval.** The frozen proposal `f8b763ddc3b7062734e11afa2f3041abac0992382ebe86236135650e769fc6f2` supports an additional $0.83 posted-charge offset and $5.48 revision of six finished H200 estimates. These amounts are disjoint from the existing $3.74/$9.51 corrections. No arithmetic or source-attribution blocker was found.

I independently checked the 18-record journal and its original 9/11-record certificate prefixes; all proposal evidence hashes; nine new manifests, workers and saved guards; 41 uploaded-file hashes; and the actual single Sandbox.create AST in each worker. The fixed CPU/memory limits are 4/32 GiB. The conditional GPU expression is controlled by `args.mode == 'compile-sass'`; saved provider metadata confirms no GPU for the two CPU runs and H200 for the others. Each new sandbox maps to a distinct app/task, has ordered created/started/finished timestamps and a terminated status, and agrees with its terminated local manifest.

All 33 previously saved report rows remain present at equal or higher cost in the fresh 58-row report. Each additional offset is the per-app posted total rounded down to cents. The six revised indices are exactly 9 and 12–16. Applying the unchanged formula `min(R, ceil_cent(7.68 * (ceil(finished-created)+60)/3600 + 0.50))` gives revised estimates $1.12, $0.68, $0.68, $0.67, $0.67 and $0.74, respectively. Every new posted charge is below its revised/full estimate. CPU indices 10/11 and timeout index 17 retain their full reservations.

The immutable journal still totals $33.88. With the historical provider floor **$4.83274092**, the $30 grant and $5 headroom, the proposed conditional allowance is:

`30 - 5 - 4.83274092 - 33.88 + 3.74 + 9.51 + 0.83 + 5.48 = 5.84725908`.

This is not current available credit. The fresh $4.65 provider summary must not lower the historical floor. Each correction must enter the total debit once; a repeated preflight must not accumulate the same adjustment from past receipts. Later reservations remain full.

Implementation acceptance must preserve the fixed scope and evidence hashes, original journal entries, lock and crash behavior, fresh full-cycle report checks, all-app stopped check, per-app charge ceilings and all historical spend observations. Focused negative tests should cover duplicate/overlapping correction membership, altered prefixes or source caps, missing/downward/new excessive billing rows, a lower fresh summary, repeated preflights and new reservations, and unresolved cleanup. The saved app list contains only the latest stopped app; it is not an exhaustive proof about workspace activity. The separate 18 owned sandbox records establish termination of the reviewed runs, while a fresh `app list --limit 0` remains necessary before allocation.

The lifetime adjustment remains an explicit conservative estimate. A provider task's finished timestamp is not a documented final billable-release watermark; the additional 60 seconds and $0.50 build allowance are not provider-enforced bounds. The report/compute-summary difference is only $0.00000002 and fits the existing display-rounding allowance. Delayed, unrelated and unallocated provider costs remain included through the full spend floor. No provider call, guard/journal/lock edit or allocation was performed during this review.

Exact evidence and verification counts are saved in `experiments/iterations/20261104-modal-additional-reconciliation-independent-review.json`.
