# Dense stage directory V11 — terminal result: no useful gain

Private job `4b9f33ffa6bc4eed` passed13 correctness tests and12 timed cases, with
no flags, at2.1492503148x (CI2.1170115601–2.1814890695). This is -0.044% versus
exact80a4 public and -0.226% versus its private result; current97af public is
2.1811831350x. **Reject promotion and combination with KPERM.**

Preparation logs prove four of nine tables selected the directory, corresponding
to18 profiled launches (eight QKV, eight pooler, two T32 unpooler). Small-token
fusion/dispatch remains intact. Target-case changes are mixed; unchanged controls
also move. There is no useful aggregate gain or basis to tune the gate from these
outcomes. Full audit: `tools/DENSE_STAGE_DIRECTORY_RESULT_AUDIT.md` and
`experiments/iterations/20261104-dense-stage-directory-result-audit.json`.
The source stays frozen as measured evidence; no repeat run is proposed.

Source SHA-256:
`15a7188b3e35b469565cc2d7eb9b37c0804e341dab5f1a4939a16563efdaa172`.
Parent is frozen public decoder80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c.
This includes no KPERM, compact WGS, or rejected global-input correction rewrite.
Plan: `tools/DENSE_STAGE_DIRECTORY_PLAN.md`; CPU implementation approval:
`experiments/iterations/20261104-dense-stage-directory-plan-critic.json`.

Hypothesis: the original correction warps scan every escape index even when a
different warp owns that stage. An immutable per-KST directory can remove that
duplicated metadata work. Every original shared activation read, arithmetic
operation, FP32 plane order, stage wait and ring handoff remains. This saves no
activation traffic; its latency benefit is unknown.

Preparation appends a16-byte-aligned directory only when the original plan's
whole-table logical-work filter passes and every applicable geometry fits.
Existing correction entries retain their original sort/tie order. Entry0.y
locates the directory; entry0.x still holds count. Offsets use1-based entry
indices and the exclusive count+1 sentinel. Unselected tables keep exact original
bytes and pmax. Their traversal/order is equivalent to the parent, but includes
a new runtime directory-offset read/branch; it is not instruction-identical.

Root found static shared memory omitted from the first preparation fit estimate.
Final source uses `max(0,min(232448,device_optin)-2048)` as its dynamic budget.
The2KiB reserve exceeds the target kernel's260 declared static bytes plus up
to1023 alignment padding. Missing device opt-in disables directory construction.
The original ring depth and actual CUDA specialization/BND capacity query remain.

CPU verifier `tools/test_dense_stage_directory_cpu.py` passed on frozen bytes:

- 45 table cases across9 shared immutable table geometries, comparing actual
  parent/candidate Python helper outputs through a NumPy adapter;1,399,712
  original entry bytes checked, zero/one/empty-stage/duplicate-K/boundary/dense
  coverage, work/fit fallbacks and exact fit-boundary tests.
- All14 compressed geometries enumerated from the actual emitted ordinary,
  fused, UV and BND launch source; minimum NESC and maximum fixed shared-memory
  footprint independently checked against that consumer set.
- Extracted original/candidate C++ traversal compiled and executed with read
  instrumentation:9,414,880 ordered entry/address pairs and1,485,760 barrier
  events agree over2,640 table/geometry/physical-M cases. Includes every physical
  M1..TOKR with representative first, last and padded lanes, FUSE router stages
  and inherited clustered fallback; this is not an exhaustive lane enumeration.
- 448 randomized ring interleavings with all original consumers, plus missing/
  extra-arrival negative controls. Source equality outside the specified traversal
  and namespace; only four existing AST nodes and one metadata helper change.

Saved result: `experiments/iterations/20261104-dense-stage-directory-cpu.json`,
SHA78bb1cb9b15eab8ddf067dac6fdbdbab7c697872482cf05f84e9589dc0f278ee.
Verifier SHA70cd4b4eb2f805d45f59fe4b7d65122cceb459a6e756aa9d15263278ee061910.
These checks establish bounded CPU invariants, not actual Torch/CUDA behavior,
device compilation, GPU races, correctness-oracle results, or performance.

Independent exact-source review approved these frozen bytes for one private
whole-decoder check, recorded in
`experiments/iterations/20261104-dense-stage-directory-source-critic.json`.
The critic independently reran the CPU verifier and audited the complete focused
source, input lifetime, resource accounting and actual dispatch. Root owns any
GPU allocation, submission and polling; no public promotion is approved.
For the private check, require13 correctness tests and12 timed cases, no flags,
actual preparation selection evidence, and useful whole-decoder improvement
beyond noise. Target at least3% aggregate improvement with matched small-T
qkv/pooler evidence and no material control regressions. The filter may select
no official tables; that is a failed mechanism test, not a gain. The renamed
kernel alone does not distinguish selected-directory tables from fallback ones:
preserve `SK2_STAGE_DIRECTORY` preparation records. Reject lost fusion, an
inactive mechanism, regressions or no useful gain. Do not tune the gate from
benchmark outcomes or reintroduce global activation reads.
