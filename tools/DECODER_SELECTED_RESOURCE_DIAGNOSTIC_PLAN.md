# Selected decoder resource diagnostic — 2026-11-04

Purpose: one private, unscored check of an isolated copy of decoder
`97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
Obtain compiled resource and launch-capacity facts for actual post-prepare SK2
launches and the decoder's selected expert builds. This is an evidence-gathering
check, not another performance candidate. Its score is not a promotion signal.
The independent critic approved bounded CPU implementation of the plan (initial
SHA `95ab414c9fc26677afaa2199ce4be49ac5673250fc87e2a43d25bfd1ea1a18a5`).
No GPU check is authorized before final frozen-source review and root action.

## Selected launch identity

Instrument the host `launch_sk2` lambda, after its existing `attr_done` opt-in
setup and after the `cap != nullptr` return (parent lines 11000–11079). The
exact `kern` pointer already includes MT, TOKR, NT, NWR, WSK, UPW, NSTAGE,
COMP, HL, NACC, CL, OutT, PK, FUSE and BND. Record those arguments plus actual
M/N/K, pmax, dynamic shared bytes, threads, grid, `ta.mode` and PDL. This
captures ordinary, fused pooler/router, UV and BND dispatch without reproducing
Python selectors or calling `_uv_on` (which can allocate state). In particular,
UV pooler records its physical `M=2*T`; output projection mode 1 and pooler
mode 2 are distinct signatures. Cap-only preparation calls do not count as
selected launches.

Reporting starts disabled in every extension. At the first statement of
`prepare`, increment a Python generation and disable reporting in already
loaded `bfp__ext` and `_MA_EXTS` objects. New extension objects also start
disabled. Immediately after the original `_check_guards()` and before returning
`st`, arm the already-loaded objects with that generation. Existing tuning,
PIN_PICK, bias restoration, attention self-test, compilation and guard order
remain unchanged. If prepare raises, logging remains disabled. No builder or
selector is called merely to obtain a report. Deduplication and byte budgets
persist across prepare generations; the first observation records its arming
generation. This is a host lifecycle epoch, not a weight identity: if a caller
retains and invokes an older prepared state, its launch still records the current
epoch and its actual scalar launch signature.

This means “selected” is a real host dispatch after a completed prepare. It does
not mean the invocation is a benchmark rather than a correctness call. We do
not inspect or distinguish those phases. Offline matching to profiles can
establish coverage; an unobserved specialization remains unknown. Fresh tuning
may choose differently from 97af's earlier official run even though its algorithm,
timing calls and search space are unchanged.

## Read-only CUDA queries and bounded output

Query the exact `kern` with `cudaFuncGetAttributes`, then
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` at the existing block size and
dynamic shared size. Obtain current device and SM count with read-only runtime
queries. Record each API's numeric status, and emit -1 for unavailable outputs.
Do not add setters, synchronizations, events, callbacks, launches, copies,
fallback dispatch or error-clearing calls. Existing checks retain their original
behavior. CUDA documentation warns these APIs can also report previous
asynchronous errors; a nonzero diagnostic status is not uniquely attributable
to the query.

Fields: registers/thread, local bytes/thread, static shared bytes, maximum
dynamic shared bytes, maximum threads/block, blocks/SM and SM count, with
the full launch identity above. For CL=1, blocks/SM times SM count is a
capacity calculation, not a guarantee that all grid blocks are resident together.
For CL>1, label it **block-only capacity; cluster capacity unmeasured**. No new
cluster admission query is necessary for the main compressed CL=1 question.

Use compact numeric records with a single schema header; full names and enum
meanings live in the adjacent offline parser/document. Deduplicate the exact
template and runtime launch signature, not merely the kernel pointer. Emit one
record on first observation; a changed pmax/dynamic size must remain distinct.
Reserve a strict SK2 budget of 8192 bytes including header and one overflow
record. The expert reporter has a separate combined budget of at most 2048
bytes across extension modules. Concretely, the three possible small builds and
their PRMT0/1 versions receive at most 320 bytes each, maximum six allocations
and 1920 bytes including overflow/schema. MAIN receives no resource query. A
new lazy module inherits the common arm state; it must not report a tune path.
Bounds
must be verified from the formatting implementation, not an assumed line length.
The overflow record explicitly invalidates exhaustive coverage. Keep compiler
verbosity and flags unchanged. Parent logs cannot be suppressed; if their
unanticipated volume removes diagnostic lines from the 16 KiB official tail,
the retained evidence is partial, not a reason for an automatic retry.

The small-expert allocation retains at most one resource identity per module
(device, BIG specialization, threads, dynamic shared size). M/grid/PDL on its
line are the **first observed sample**, not exhaustive runtime-shape coverage.
Identical identities reuse the same compiled resource facts without queries;
a second distinct identity emits partial-coverage STOP before any query. At most
six positive-budget modules are stored; MAIN and unknown modules add no registry
entries. One global unknown-module notice is capped at 128 bytes, bringing the
combined worst-case diagnostic bound to `8192 + 6*320 + 128 = 10240` bytes.

Post-prepare host queries and formatting can add graph-construction/eager host
overhead. They do not add CUDA graph nodes or execute on graph replay; therefore
the device math and replay sequence are unchanged at source level. We make no
claim that this proves bit-identical cubins or identical check timing. This is
not an instrumentation-free speed comparison.

## Source scope and CPU review before a check

One writer owns the isolated candidate. Allowed changes are host-report helper,
one control binding in each extension, launch-site report calls, prepare
arming/disarming, distinct extension cache names and provenance comments.
The expert author supplies a separate snippet for review; no simultaneous edits.
Keep every device function, launch expression, packing helper, numerical define,
compiler flag, autotuner and dispatch generator byte-identical after stripping
only the new host snippets. Preserve the original pybind exports and add controls.
Non-SM90 fallback must export a no-op SK2 control too.
Here “compiler flags” means the explicit build option lists and numerical
defines. The changed extension cache name necessarily changes the generated
module identifier/pybind name; it does not add a device tuning option.

CPU verification must inspect the generated source (including BND/UV), establish
the unchanged device blocks and host launch statements, test lifecycle/repeated
prepare behavior without CUDA, demonstrate exact dedup distinctions, and exercise
budget exhaustion/error-formatting cases. Independent critic then reviews frozen
source. Root owns the one official private check and exact result archival.

## Decision and limitations

Successful outcome: valid resource records tied to actual selected launch
signatures, sufficient to test register/shared-memory constraints and identify
which SK2 geometries deserve a separate architecture hypothesis. Actual expert
flags distinguish the decoder from standalone resource captures.

Attributes do not measure achieved occupancy, stalls, bandwidth, cache traffic,
escape-warp cost, WGMMA serialization or spill traffic. `localSizeBytes` reports
allocation, not dynamic traffic. Block capacity does not establish grid-wave
efficiency or fused-boundary latency. No improvement or new GPU variant follows
without a separate falsifiable plan and critic review. Missing/error/overflow
records remain unknown; correctness failure terminates this diagnostic attempt.

Primary API references checked November 4:
[CUDA execution control](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXECUTION.html)
and [CUDA occupancy](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__OCCUPANCY.html).
