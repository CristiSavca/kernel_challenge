# Independent serving-finalizer GPU validation review — November 4, 2026

This review concerns unchanged helper
`91e6173baa984fb873994cc3cd2883da16e013091f10218262785e70e613ed98`.
Its official stock-Marlin diagnostic already passed the fixed component screen;
the present task tests remaining correctness and ownership risks. It does not
select a new implementation, retune a speed gate, or establish full serving
performance. Root alone owns the runner, allocation and any later serving check.

## Plan requirements

The bounded plan is sound in principle. A single synthetic H200 run can cover
the changed helper without loading Muse weights, building the MoE extension, or
importing the entire serving stack. The following are required before a source
GO and must be explicit in the eventual result:

- Bind the exact candidate, unchanged original helper
  `9b816e971817f96a22d2d09424dc8dfa6f6020e18de73fb895674c76494fe591`,
  worker and finite manifest. References execute in separate modules. The FP32
  oracle must independently express the original small-path operations and must
  not call candidate kernels or share their scalar scratch/reduction helper.
- Compare candidate to both original and oracle; also compare original to oracle.
  Preserve and attribute disagreements rather than accepting two implementations
  merely because they agree with each other. Whole/row relative-L2 limits
  0.015/0.075 with the 0.1 average-row-norm floor are the chosen existing MoE
  component limits, not a claim of bitwise or full-serving agreement.
- Run counterexamples early: ragged hidden/output tails, the T31/32 boundary,
  masked poisoned input, all-remote and one-live routes, zero epsilon, nonfinite
  score/weight, norm-before/after-weight, and disabled normalization. For
  exceptional inputs require matching NaN/+Inf/-Inf masks and then assess finite
  positions separately. The official evaluator rejects nonfinite outputs;
  exceptional classification tests are an additional helper-contract check.
- Cover arbitrary small token counts with reused compiler signatures, selected
  H/K boundaries, auxiliary dtypes, noncontiguous inputs and auxiliary tensors,
  and original fallback branches. Large-path oracle controls should use finite,
  ordinary layouts: the original large kernel skips invalid-route arithmetic
  and has inherited auxiliary-stride limitations. A changed-path defect and an
  inherited limitation need separate attribution.
- Observe actual GPU kernel names for representative fast and fallback paths.
  Host metadata prediction alone does not prove that the intended kernels ran.
  Bound signature counts using pointer dtypes and constexpr flags as well as H/K;
  do not equate shape count with actual compiler cache entries.
- Test two nondefault streams with explicit input-production dependencies and
  independently owned outstanding outputs. Independently captured graphs must
  replay fresh x/scores/IDs on their own streams and retain distinct graph output
  storage. A graph intentionally reuses its captured output on replay; test
  independence between graphs, not a nonexistent fresh-output-per-replay promise.
- A temporary allocation observer may record pointer addresses and byte extents
  while delegating `torch.empty` unchanged. It must not retain inverse-scratch
  Tensor objects: extending their lifetime could hide the ownership defect under
  test. Restore the observer before replay. Retain only the normal graph/output
  ownership required by the production API.
- Stop on first material failure. A 540-second between-test deadline inside a
  600-second outer sandbox budget is a bound on starting work, not an interrupt
  guarantee for CUDA compilation or kernels. Timeout/incomplete coverage must
  never become PASS. Checkpoints must identify exact completed stages, failure
  case/seed/pattern and required remaining checks, with bounded output size.

No extra performance campaign or generic verification framework is needed.
The remaining source review will inspect whether the worker actually enforces
these requirements; plan acceptance alone is not GPU-validation approval.

## Conservative observed-spend floor

Root's separate minimal guard change is appropriate: an optional nonnegative,
finite `observed_spend_floor` can only increase the conservative provider debit.
It cannot revise old reservations, certificates or the grant. The inspected
delta parses the floor before filesystem/provider work, takes its maximum with
all existing observations, and persists the effective high-water observation in
the new journal row so later allocations inherit it. Four focused CPU tests
independently passed, including invalid input before side effects, higher-floor
inheritance, lower-floor non-reduction, and exhausted-budget rejection without
journal append. Frozen hashes and the runner's exact saved-evidence binding will
be checked with the worker before allocation. No provider call or budget-state
mutation was made by this reviewer.

## Frozen worker verdict

**GO for this fixed worker, conditional on the separately reviewed runner and
ordinary fresh credit/cleanup checks.** Source
`tools/serving_finalizer_validation_worker.py` is
`c94c31345575540c74a3867512aefff1480c0dbc9471409d5c1ceb39d48f5ede`;
plan `a556ea43d5c48c106c65601ee719b042523285c816a40db8cefd967c96e43d59`;
CPU test `2d5d7ea82375e7e9d82c49c7eef1928db0b5b7acce5fb1eaae802e38593a25ce`.
The fixed 90-case manifest is
`a57d656ba2635c8e3c6d95c93351de0aa4508e2c3b292510108c73abc535ccb2`.

Full source inspection found no remaining worker blocker. Independent execution
of all 12 local tests passed. They cover policy/source mutation rejection,
isolated three-file packaging without importing Torch, the bounded specialization
inventory, allocation-range arithmetic, and diagnostic metric behavior including
large finite values and exceptional classes. They do not execute CUDA or prove
the finalizer's GPU numerical, graph or stream correctness.

The source implements the requested counterexample-first sequence and preserves
separate candidate/original/oracle comparisons. It includes true BF16-subnormal
input, forced positive/negative infinity outputs without normalization, and
zero/NaN/infinite/negative epsilon cases. There are at most 62 conservatively
counted small-kernel signatures under the declared 80 cap, including pointer
dtypes; this remains distinct from measuring actual compiler cache entries.

Six representative activation checks require positive-duration CUDA profiler
events: fast T31, old-kernel T32/T33, and eager FP32/K33/H16385 fallbacks. Missing
GPU events produce incomplete coverage. Both stream queues depend on completed
input production without waiting on one another; retained outputs are compared
after synchronization. Graph capture records addresses/extents only, restores
the allocator observer, and keeps graph/static-input/output ownership intact.
Eight fresh-input replays across two graphs are checked individually, followed
by an explicit graph-B-does-not-modify-graph-A-output check. This establishes a
concrete test of independence, not measured simultaneous execution or universal
scheduler safety.

The initial runner draft used unbounded SDK `filesystem.read_text` for success
and failure recovery. Review requested the existing bounded remote-read pattern
for this new mode: fixed result path, a pre-transfer 1MiB cap and 10-second read
command, with nonzero worker exit preserved independently. This is a runner
review item, not a reason to change the finalizer or expand GPU testing.

## Final runner and execution verdict

**GO for one fixed 600-second H200 validation through the existing fresh credit
and cleanup guard.** Runner
`339a85ef3a539c4bd4d5f8fb21fb3dfb6c9227b9a870da8d6ae8507cc5397a85`
resolves the retrieval item: a fixed remote command caps the result before
transfer at 1MiB and has a 10-second command timeout. The host checks source,
worker, manifest and image identities. PASS requires all 90 case descriptors,
six activation records, eight stream calls and eight replays across two graphs.
Partial/recovered data stays in a checkpoint file; nonzero worker exit and its
original error survive retrieval failure. The existing termination path remains.

Independent checks passed: 12 new runner tests, 18 existing lifecycle tests,
12 worker contract/metric tests, four floor-accounting tests, and the actual
`serving-finalizer-validation --timeout 600 --dry-run` without provider access.
The new runner tests are
`f87f19a4768ed92e97150511dc8e75cddb87f41d87e607187500b17bf6a066cf`.
Installed Modal 1.6.3's inspected `Sandbox.exec` signature supports the binary
`text=False` read used here. The 10-second bound is on the remote read command,
not a universal bound on network failures.

Guard `9f70c5006d8afd6e4a077cec8f3fe3baada2620703479bebf87ea0a333f0c9b2`
and floor tests
`04fc44059600f7aeb76baf3d9ebfe64e34bb75c67ee25e3008c6a7a5502d0ad6`
preserve the conservative accounting semantics described above. The mode binds
the saved observation of $4.16434237 to billing-after evidence
`0daf0080100319c8fb1af515a6f831598b457cf712e6356cd34e2566075be909`.
Fresh provider costs can only increase that debit. The actual sandbox request
remains one H200, fixed four CPUs/32GiB, network blocked, 600 seconds; no model
weights are needed. This review performed no allocation or provider query.
