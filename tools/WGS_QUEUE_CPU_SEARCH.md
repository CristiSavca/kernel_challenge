# CPU search for WGS queue lifetimes

November 4, 2026. This is a bounded search over an executable model of the
current WGS queue hypothesis. It generates no CUDA variants, runs no GPU job,
and makes no performance prediction or Lean proof claim. The separately
reviewed four-bank/wait2 experiment remains the only implemented experiment.

Run `python3 tools/search_wgs_queue_cpu.py`. The tool reads
`tools/wgs_queue_search_manifest.json` and writes
`experiments/iterations/20261104-wgs-queue-cpu-search.json`.
The manifest pins the exact scored MoE parent `247888ac…`, the existing
`audit_wgs_wait2_lifetimes.py` checker, and the NVIDIA occupancy-header evidence
by SHA-256. No public-repository code is executed.

## Search and admission rules

The grid has 320 combinations: wait depth 0–7, AF banks 1/2/4/8/16, and previous
activation-slot release after group ci0–7. GS2, eight commit groups per stage,
three weight/activation slots, 576 threads, 231,168 dynamic shared bytes, and
the original segment-end wait0 remain fixed. AF bank selection resets each
stage; NTT1's accumulator parity remains `ci & 1` independently of AF banks.

Two arithmetic lifetime conditions are checked independently of the event
checker's pending-group list:

1. The bank reuse distance is `min(af_banks, 8)`. Wait depth must be smaller
   than that distance, because those most recent groups may still read their
   register-A operands before the next decode. Sixteen declared banks reference
   only banks0–7 in this fixed eight-group stage.
2. At release, `release_ci + 1 >= wait_depth` must hold. Otherwise a group from
   the previous stage may still read the activation slot whose credit returns.

The event checker confirms each original rejection. When both hazards exist,
the output retains both reasons and an isolated witness for each: delay release
to ci7 to expose AF overwrite; use eight banks to expose premature X release.
This avoids hiding the second error behind the checker's first assertion.

Every surviving schedule is compared with the parent's complete modeled
K-order/accumulator trace over 672 cases: all NTT1–8, six starting sequences
spanning the three ring slots and two logical generations, and fourteen
empty/full/partial/adjacent segment patterns. A separate flat-K trace oracle
also checks the parent. Negative controls include the three-bank stage-reset
collision, two-bank/wait2 overwrite, early X release, and deliberately changed
K order/accumulator parity. The latter are trace-detector controls, not mutant
CUDA source validation.

The complete report retains all 320 outcomes and rejection details:

| Result | Count |
| --- | ---: |
| Unsafe schedules rejected | 181 |
| Lifetime-safe schedules, device behavior unknown | 139 |
| Safe schedules within the deeper-queue hypothesis, depth >=2 | 67 |
| AF-overwrite reasons | 136 |
| Premature-X-release reasons | 105 |
| Parent-trace comparisons | 93,408 |
| Commit groups compared | 7,472,640 |
| Automatically approved GPU experiments | 0 |

The two rejection counts overlap. Banks8 and banks16 have identical abstract
bank references here, giving 96 abstract behavior classes among the survivors;
their differently declared arrays are not proven to compile identically.
The existing four-bank/wait2/ci1 choice passes with 24 peak logical live AF
words versus the parent's16. Its32 declared words do not imply32 physical
registers, and this search does not rank it by predicted speed.

## Hardware gate and limits

The pinned evidence calls NVIDIA CUDA13.0.96's unmodified occupancy-header
register calculation on CPU with SM90 properties. For576 threads it admits
one CTA at96 registers, and zero at97 or112. This corrects the misleading
`65536/576` aggregate-register argument. It is a register-only necessary
condition, not an actual device query or a full resource-admission decision.

The report explicitly leaves actual compiled registers, local storage, static
shared memory, occupancy, compiler serialization and performance unknown.
Logical AF liveness cannot supply those values. Any future exact source still
needs both BIG specializations' compiled attributes and actual positive
occupancy at the fixed launch geometry, followed by correctness and measured
performance. A source that reports <=96 registers can still spill, serialize,
or run slower; no automatic variant generation or rescue is authorized here.

The model assumes ordered group retirement, unchanged barrier participants and
producer protocol, the original FCI3 flush of the already drained preceding
segment, and segment-end wait0 before generic accumulator use. The finite cases
do not simulate every possible route, CUDA scheduling interleaving, barrier
phase, or operand value. Abstract trace equality does not establish GPU numerical correctness.
The arithmetic conditions are transparent reasoning plus executable checks,
not a machine-checked theorem in Lean.

The reusable workflow idea from the public repository is a pinned manifest,
explicit checker, retained negative outcomes, and conservative admission of
unknowns. See `tools/ROHAN_REPOSITORY_RELEASE_AUDIT_20261104.md` for the released
tools versus the announced GPU search system. A statistical predictor needs
an adequate measured dataset; none is introduced by this search.
