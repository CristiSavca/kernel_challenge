# Open Kernel Challenge — Claude handoff

Prepared November 4, 2026, after official V33 completed. Repository:
<https://github.com/CristiSavca/kernel_challenge>. Original workspace:
`/Users/cristisavca/Documents/HASIDIC_PROJECTS/open-kernel-challenge`.

## Current state: paused, no GPU jobs running

The user explicitly paused the optimization goal, then asked for this handoff
and GitHub push. Both the goal and Codex automation
`okc-research-and-harsh-critic` are **PAUSED**. All optimization agents are stopped. No
official or Modal GPU job is in flight. Do not interpret the older automation
prompt, historical plans, or this handoff as permission to resume experiments.

The eventual objective is #1 and a competitive serving entry. It has **not**
been achieved. The requested 100% score gain was a stretch target, not a measured
result or defensible prediction. Recent private decoder experiments regressed;
none replaced the public incumbent.

The last saved board snapshot reports CristiSavca **#409 overall and decoder**,
with decoder 2.1812× against a 2.5762× leader. It is a dated snapshot, not a live
rank guarantee: [board evidence](experiments/leaderboards/20261104T2300-v32-closed.json).

## Read these first

1. [AGENTS.md](AGENTS.md): project constraints and existing authorization.
2. This handoff, then [tools/RESEARCH_LOOP.md](tools/RESEARCH_LOOP.md).
3. [tools/MODAL.md](tools/MODAL.md) before any resource use.
4. The relevant problem's `AGENTS.md`, `TASK.md`, and actual source.
5. The exact receipt/profile for the hypothesis being considered. Broad older
   research notes are not substitutes for current measurements.

## Passing public incumbents — preserve these

These source files match the recorded scored receipts. Full hashes are in
[incumbents.json](handoff/20261104/incumbents.json).

| Problem | Public speedup | Scored job | Source SHA prefix |
|---|---:|---|---|
| Dense projections | 1.4868267835× | `3caa02bd72da4c49` | `27a0afe97d0a9b8c` |
| Attention | 2.6119701145× | `8f9e5971f95841a0` | `87a9bddc1b013ff3` |
| Routed experts | 2.3736910444× | `d24dd50196cb4158` | `247888acca10f50d` |
| Whole decoder | 2.1811831350× | `7a8c4941741645ee` | `97af39947d4cab45` |

The decoder also has an exact-source private anchor `b2a773e2d7194de8`,
2.1766544507×. Evaluate a new proposal against both retained decoder anchors.
Canonical receipts are `experiments/runs/<job>.json`; frozen submitted sources
are under `experiments/candidates/<problem>/`; raw profile JSON/text/logs sit in
that directory's `profiles/`. A local compile or benchmark is not an official
pass. Do not promote a source with unverified flags or incomplete case coverage.

## Latest iteration and the approaches already rejected

V33 is **closed**, with no scored submission or public source change.

| Iteration | Official private job | Score | Decision / observed reason |
|---|---|---:|---|
| V28 decoder private partials | `2976c25cc3dc41b1` | 2.1146266798× | Correct, slower than incumbent; no retry/port |
| V29 exact BF16 repair | `185fee4d182f41dd` | 1.8985960692× | Correct, no compressed timed activation |
| V30 split MoE architecture | `d0510bc0412a4537` | 2.0335370900× | Standalone MoE; slower than its 2.3737× incumbent; no decoder port |
| V31 eligibility diagnostic | `5f48e0da4dcd4595` | 1.9115472025× | Selftests pass; tuner rejects compressed choices; array-sized local allocation |
| V32 scalar repair | `6a938fff9d5d417c` | 1.9080803869× | Local allocation removed, still no compressed timed activation |
| V33 missing-nibble sidecar | `60b61041d08a4880` | 1.9740118155× | Correct, partial activation, all affected performance caps fail |

V33 exact source is
`a8a619eaa0c531aa115d39fd2db8718d7d52828feac8e6fdea005c4e61a3c090`.
It retained a 768-byte primary per 512 BF16 weights and supplied missing high
nibbles from a 256-byte sidecar for escaped units. This removed per-word repairs
and the header dependency before primary loads. CPU packing tests passed; the
official check passed 13 tests and 12 timed cases with clean flags.

Nevertheless, V33's whole-score ratios were **0.905019 / 0.906902** against the
public/private incumbent; affected four-load speed ratios were
**0.864490 / 0.866609**. All eight affected cases exceeded the fixed latency cap.
Controls passed. Only some compressed kernels activated. All 22 observed
compiled pointers had zero local allocation, including 18 compressed pointers.
Those compiler observations do not establish a performance gain or sole cause.

Read [V33 closeout](experiments/iterations/20261104-decoder-nibble-sidecar-v33-closeout.json),
[raw analysis](experiments/iterations/20261104-decoder-nibble-sidecar-v33-root-result.json),
and [independent critique](tools/DECODER_NIBBLE_SIDECAR_V33_RESULT_CRITIC_20261104.md).
**Close the V29–V33 consumer repair/completion line.** No identical retry,
parameter rescue, forced selection, relaxed threshold, or post-result pooler
subset splice. A materially different hypothesis needs its own evidence.

Other closed work includes shared-A, SC6, R16, escape directories, prefetch,
wait-distance and register sweeps. Consult the current research loop and the
specific records before proposing something that sounds new. The unique-expert
reuse audit found only a conditional ~1.465% whole-decoder scenario, not a
demonstrated large lever.

## Proposed next public-kernel experiment: V34, not implemented

The distinct evidence gap is the **original incumbent's dense-kernel hardware
counters**. Existing retained NCU captures are standalone MoE; V31–V33 compiler
attributes and tuner times do not identify the original dense bottleneck.

Proposed scope, only after the user resumes and resource/worker review passes:

- Load exact decoder97af unchanged. Use `bfp_prepare` with all five dense weight
  matrices; no MoE/cache allocations are needed for this component diagnostic.
- Inspect exactly QKV M16 via `bfp__run_sk2` and fused pooler/router M16 via
  `bfp_fused_pooler_router`, at natural `bfp_SK2_PDL=0`.
- Pin source, image, generated CUDA/CPP/build flags, weights, templates and
  PK/FUSE identities. Record actual escape counts and patch-table sizes.
- Independently check outputs and collect ordinary cold graph observations
  separately from NCU replay. This is standalone dense evidence, **not** UV,
  PDL, full-forward timing, or an official speedup.
- Declare one metric set covering DRAM/L2, issue/eligible warps, shared traffic
  and conflicts, synchronization/scoreboards and pipeline pressure. Query actual
  availability; missing required metrics or denied counters means stop
  inconclusive. No metric fishing or allocation retries.
- Aggregate counters can distinguish broad pressure classes. They cannot
  identify escape instruction PCs or be summed into recoverable-time percentages.

**No V34 worker, test, manifest, or ready-to-run payload exists.** The controller
draft was started, then archived at the pause. Proposed worker contract:
`tools/decoder_dense_hardware_v34.py`, APIs `build_payload(root,image,tag)`,
`validate_payload`, `run`, `validate_result`; files `decoder.py`,
`hardware_metrics.py`, `dense_v34.py`; 900-second sandbox, 850-second worker
budget; atomic result <=1 MiB. The metric list, implementation, identity checks,
full-symbol parser, oracle and tests remain unfinished.

The archived controller patch references this missing worker. **Do not apply
the patch and assume it is executable.** The existing controller was restored
to its last committed version after the exact draft was preserved.

## Modal money: current guard still blocks allocation

Authorization is **free-credit experiments only**, not out-of-pocket GPU/API
charges. One H200 at a time, <=900 seconds, no automatic allocation retries,
guaranteed sandbox termination. Use the guarded runners. Adding a payment
method did not authorize paid usage. Official challenge checks remain a separate
resource, subject to current rules/quotas/deadlines.

The $30 grant, $5 headroom, historical provider spend floor **$4.83274092**,
$33.88 original reservations and already-reviewed $3.74/$9.51 adjustments leave
the old-policy conservative allowance **−$0.46274092**. This is ledger
accounting, not a statement that the provider's entire grant was consumed.
Never delete/reset `experiments/modal/.allocation.lock` or the reservation
journal, reduce the spend high-water, or count corrections twice.

A new read-only reconciliation found **$0.83** additional posted-cost overlap
and **$5.48** additional conservative estimate reductions. Conditional allowance
would be **$5.84725908** after reviewed activation and fresh preflight. These are
not settled refunds or currently spendable authorization. CPU runs and the V27
timeout retain full reservations. The lifetime policy has an explicit teardown
and delayed-billing uncertainty; maintain the existing margins and headroom.

The [proposal](tools/MODAL_ADDITIONAL_RECONCILIATION_PROPOSAL_20261104.md) and
[independent proposal review](tools/MODAL_ADDITIONAL_RECONCILIATION_CRITIC_20261104.md)
are complete. **Implementation was interrupted, untested and not reviewed.** Its
activation pin was `None`. The planned test file does not exist. The exact draft
guard and certificate are archived; the active guard is restored to
`9f70c5006d8afd6e4a077cec8f3fe3baada2620703479bebf87ea0a333f0c9b2`.

Before any future activation, finish the fixed additive policy, independently
review it, and test tampering, overlap, repeated corrections, missing/downward
billing rows, excess costs, resource changes and unrelated future reservations.
Preserve old certificates and the complete original journal. Requery billing,
all app activity and available grant through the guard immediately before use.

## Draft preservation and the nested serving checkout

[handoff/20261104/manifest.json](handoff/20261104/manifest.json) records exact
base/draft/patch hashes. It preserves:

- `modal-credit-additional.DRAFT.patch` and its draft certificate. These are
  incomplete, disabled accounting changes; no implementation approval.
- `decoder-dense-v34-controller.DRAFT.patch`. Only controller scaffolding;
  its worker is absent. Eleven existing V27 controller/worker tests passed
  during development, which does not validate V34.
- `serving-working-tree.PRESERVED.patch`, the pre-existing two-file nested
  checkout diff at SGLang `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`.

Both draft patches were checked as reversible and applicable to their recorded
base. Active tool files were restored only after exact patch preservation. No
user edits or unique nested commits were discarded. The nested serving working
tree itself was left untouched. Its full checkout is intentionally ignored by
the parent repo; on another machine, recreate the pinned SGLang checkout before
applying its preserved patch. That dirty patch is V22-related work, **not** the
isolated V27 serving entry and not a validated deployment.

## Serving work worth retaining

The prize path is the separate serving track. Prototype **V26** passed official
private job `274c69cbc0954d76`, five tests/six benchmarks, score **1.6851100701×**
its same-run stock component. It is slower than the public native MoE kernel;
its value is a serving integration candidate, not a public-kernel improvement.
No full-service throughput, rank, TTFT or prize claim is established.

**V27** is the isolated pre-Marlin installer/forward adapter at
`serving/candidates/native_serving_v27/`: native SHA `7310dd76…`, patch
`c0ca4e41…`, 213,160 bytes across four allowed files. CPU reviews passed. Its
bounded Modal run `20261104T213729Z-f6cf2030` installed the rank-3 native bank,
then timed out within a coarse stock-preparation/initial-forward interval.
It completed **0/38 numerical cases and no timings**. The sandbox terminated.
This is incomplete validation, not a numerical pass/fail diagnosis.

The distinct next serving probe would isolate the stock interval with stage
checkpoints, not rerun the same full suite. See
[diagnosis](tools/SERVING_NATIVE_MOE_V27_STOCK_PREP_DIAGNOSIS_20261104.md).
A separate Python3.13 `ast.dump` empty-field fingerprint mismatch was reproduced;
`show_empty=True` resolves it, but it did not cause the timeout. Preserve the
frozen failed worker and repair only a new diagnostic version.

## Rules and deadlines last verified November 4

Recheck [live rules](https://openkernelchallenge.dev/rules) before resuming:

- Kernel cutoff: **November 5, 2026, 08:00 UTC**. Stop kernel submissions then.
- Serving opens **November 10, 17:00 UTC**, closes **November 20, 08:00 UTC**.
- Prizes are for serving; kernel top50 earns queue priority. Serving board is
  private, so #1 may be unobservable before it ends.
- Live kernel quotas were40 private checks/20 scored runs per problem/day,
  resetting00:00UTC; `okc problems --json` is authoritative.
- Serving uses8H200, 3 checks/1 scored submission daily. Preserve serving-specific
  BF16, concurrency, output and TTFT requirements; kernel permissions differ.
- Serving patch: only source/JSON under `python/sglang`, no model_loader changes,
  binaries or renames, <=409600 bytes, no added line over400 characters.
- No timing/harness tricks, cached input-derived outputs, skipped experts/keys,
  lower-than-allowed precision or seed-specific policy. Weight-only lossless
  packing is allowed. Official evaluators stay unchanged. A human accepts terms.

## Environment, provenance and resumption

Original host: Apple Silicon Mac, no CUDA GPU. Installed tooling was `okc`0.7.1;
`.venv-modal` used Python3.13.4, Modal1.6.3, NumPy2.5.4. The venv, credentials,
keychain and nested SGLang checkout are not in Git. Authenticate on the target
machine through normal human-owned flows; never commit tokens. CUDA compilation
and numerical/performance validation require the actual H200 environment.

Some immutable receipts embed the original Mac's absolute artifact paths.
Strict readers such as `analyze_decoder_v28.py:read_file` reject paths outside
their current repository root. On a differently located clone, read the retained
JSON directly or implement an explicit, reviewed old-prefix-to-new-root mapping
in a new reader version. Do not rewrite frozen receipts, weaken path checks, or
silently invalidate their recorded hashes to make a historical analysis run.

The repository includes retained diagnostic binaries/SASS as evidence. They are
not permitted submission payloads: challenge GPU code must be built from readable
source during the run. Preserve third-party LICENSE/NOTICE files and attribution.

Research intake already reviewed EAT, ParallelKernelBench, Together Compiler,
X/arXiv/GitHub/Hugging Face/DeepSeek/CUTLASS leads. Start with their linked notes
in `tools/RESEARCH_LOOP.md`; do not repeat a broad scan without a specific new
lead. No verified stronger public whole-decoder drop-in was found in the scoped
scan. The Lean work under `formal/` proves limited source-bound integer/packing
properties, not whole CUDA numerical correctness, races or performance.

When the user explicitly resumes: first check the deadline, jobs/processes,
live quotas/boards and source hashes. Finish the disabled credit-policy review
only if needed for a concrete bounded experiment. Complete/review V34 before
allocating; or follow a newly justified distinct hypothesis. Keep one submitter
and separate implementer/critic roles, freeze sources/gates before measuring,
resolve material findings, and score only an exact officially passing useful
candidate. Record raw results, costs and keep/discard decision. Do not resume
the Codex recurring monitor unless the user asks to resume that work.
