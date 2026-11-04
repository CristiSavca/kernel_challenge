# Fixed additional reconciliation proposal — not activated

The read-only provider snapshot identifies an additional **$0.83 posted-charge overlap** on the nine reservations after the original reviewed certificates. A separate fixed revision of **six later H200 estimates** under the already-reviewed lifetime policy would reduce their total by **$5.48**. These sets exclude the original nine runs: the existing $3.74 overlap and $9.51 estimate corrections remain counted exactly once. No guard, journal, lock, grant or existing certificate changed.

Fresh full-cycle hourly usage totals $4.64994882, within $0.00000002 of the provider compute component. The new apps account for $0.86480320 of that total; each is bound through the original manifest sandbox ID to a distinct terminated provider task/app. All original saved billing rows remain present at equal or greater cost. Actual provider resources confirm4CPU/32GiB, with H200 only for the H200 reservations. Original worker, guard and all uploaded-file hashes match their manifests.

Fresh provider spend is $4.65, below the saved $4.83274092 high-water. The proposal preserves the higher figure. Starting from the current conservative allowance of **-$0.46274092**, overlap alone gives **$0.36725908**, which still cannot fund even the smallest fixed CPU/H200 reservation. If independently reviewed and implemented, both changes would give **$5.84725908**, before any new provider usage or reservation. This is a conditional accounting result, not current authorization or a provider credit balance.

| Run | Original R | New posted overlap | Proposed R | Estimate reduction |
|---|---:|---:|---:|---:|
| `20261104T150731Z-ef5e7aac` | $2.42 | $0.31 | $1.12 | $1.30 |
| `20261104T153757Z-d116f3fe` | $0.78 | $0.00 | $0.78 | $0.00 |
| `20261104T155445Z-1ed7603c` | $0.78 | $0.02 | $0.78 | $0.00 |
| `20261104T182526Z-3d547341` | $1.78 | $0.02 | $0.68 | $1.10 |
| `20261104T183437Z-9826c968` | $1.78 | $0.02 | $0.68 | $1.10 |
| `20261104T185116Z-a8743472` | $1.14 | $0.01 | $0.67 | $0.47 |
| `20261104T191321Z-25f35b45` | $1.14 | $0.01 | $0.67 | $0.47 |
| `20261104T194033Z-c834b1b0` | $1.78 | $0.05 | $0.74 | $1.04 |
| `20261104T213729Z-f6cf2030` | $1.14 | $0.39 | $1.14 | $0.00 |

For the six H200 revisions, retain the exact previous formula: ceil elapsed provider lifetime, add60seconds, charge $7.68/hour (rounded resource rates plus20%), add the full$0.50 build allowance, round upward to cents and cap at the original reservation. Both CPU reservations remain full; the V27 timeout keeps its full$1.14 because the formula would not lower it. No charge is treated as zero merely because the worker failed.

The earlier policy explicitly calls this an estimate: task completion is not a final billable-resource watermark, and the60-second/$0.50 allowances are not provider-enforced bounds. A new additive certificate must be pinned to these exact runs and evidence, preserve all original records, require fresh nondecreasing billing rows and per-app charges no larger than the revised/full estimate, maintain the historical high-water and$5 safety headroom, and apply corrections only once. Later reservations remain full. Root and an independent reviewer must review implementation and negative tests before a fresh guard preflight.

Evidence: `experiments/iterations/20261104-modal-new-attribution-readonly.json`; exact proposal/source bindings: `experiments/iterations/20261104-modal-additional-reconciliation-proposal.json`. The raw snapshot came from read-only profile/billing/app metadata and SandboxList; it created no sandbox or app.
