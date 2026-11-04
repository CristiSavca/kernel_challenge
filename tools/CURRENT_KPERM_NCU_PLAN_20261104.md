# Current KPERM parent: one bounded hardware diagnostic

Read-only feasibility review, 2026-11-04. No runner, kernel, guard, journal or grant changes; no allocation. Conditional on root's separately reviewed Modal accounting correction and a fresh successful allocation guard. This profiles the current parent, independently of the outcome of v19.

## Selected scope and new information

Reuse the existing `hardware-profile` protocol for **both M32 and M256**, one prepare, six ordinary checked graph replays per shape, then exactly two profiled graph launches. Keep EP8/normal/busiest, weight seed7311, profiled input seeds7302/9302. Those routes previously gave U/J64=16/16 and26/27. Record the actual routes again; matching seeds are not a substitute for verifying them.

The old counter run used ef7 before KPERM. Current KPERM changes packed code/scale ordering and repeated scale work; new data can establish whether its cold standalone traffic, issue eligibility and queue pressure differ from the retained old measurement. This is diagnostic evidence, not a new benchmark score, a controlled paired speed comparison, decoder evidence or a rescue variant for v19.

Exact proposed source:

- `experiments/candidates/muse_moe_w4a16/247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0.py`
- Python SHA256 `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
- Embedded CUDA SHA256 `85fb417494c6eab5baabad6813275b256f7bae84c0329d0e94cb61a15030445c`.
- Current decoder lineage only: `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`; no decoder execution.
- Keep image `lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af`, unchanged from the old diagnostic. It is not a claim of current private-judge image identity.

The existing mode is **not currently ready to run this**: `modal_hardware_metrics.py` hard-pins ef7/nativeaf4/decoder80a4, and `modal_h200.py` constructs that source path and rejects a candidate override. A reviewed fixed pin refresh, not a generic arbitrary-source profiler, is the minimum implementation delta.

## Metrics: keep the old set, add at most one narrow group

Retain the four REQUIRED DRAM/duration metrics and seven existing ISSUE metrics. Add one optional backpressure group, contingent on the installed NCU returning all three exact names and suffixes:

```
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
```

These names are proposals for an exact runtime query, **not verified available on a fresh sandbox**. The saved required/ISSUE queries and 280-column raw output do not contain these three counters. Query the fixed group before collection; if any is absent, record the omission and collect only the unchanged available required/ISSUE set. Never substitute another metric, rerun collection, or turn on a broad section set. Require every requested returned value to pass the existing finite/unit/launch-identity checks.

This group distinguishes math-pipeline backpressure, the mixed special-function/branch/shared-memory MIO queue, and local/global instruction-queue pressure. It cannot identify which source instruction caused the pressure. NVIDIA also cautions that stall percentages are not all exposed performance loss. Do not add them, translate them into recoverable time, or treat low pressure as proof of no instruction overhead. [Nsight Compute 2025.3 profiling guide, warp sampling and metric descriptions](https://archive.docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html#sampling).

Do **not** add per-opcode software instrumentation, full SourceCounters, or a source-PC export/parser in this first run. The current kernel already builds with `-lineinfo`, but a PC-to-source attribution pipeline still needs additional collection/output handling; it is not obtained by the current aggregate CSV parser. Software instruction counters change measurement overhead. The present low-cost question is whether a further instruction/queue-focused investigation is supported at all. [Nsight Compute 2025.3 metric and overhead documentation](https://archive.docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html#overhead).

## Preserve method and bounded execution

- Same public reference/helpers, cloned weights, unchanged source/build flags, PDL=False, actual WGS selection/defines recorded. No runtime source instrumentation, autotuning changes or kernel variants.
- Same **128MiB read-only observable sweep**, fresh inputs copied into captured graph storage, correctness before/after collection. Ordinary event timings occur inside the NCU-injected process and remain contextual only.
- Same NCU kernel replay, `--cache-control all`, `--clock-control none`, node graph profiling, profile-from-start off, `okc::moe_fused` filter, exactly two launches, one collection process. Cache-all replay and uncontrolled frequencies limit cross-run inference.
- Actual NCU version, metric queries, selected metric list, launch IDs, two complete case records and actual replay pass counts required. Do not assume the old four passes will remain four after adding metrics.
- One H200, four CPUs/32GiB host memory, network blocked, maximum900s sandbox lifetime and existing860s hardware budget/880s worker bound. Same process-group timeout/output-limit kill and unconditional sandbox termination; no automatic second allocation or retry.
- Prior run `20261104T054636Z-ea14816c` spent223.45s in the hardware parent,220.49s in collection and203.35s in prepare. This supports retaining both cases within900s; it does not guarantee fresh compile duration. The current conservative900s reservation is$2.42 before any separately reviewed policy change, with the existing$5 reserve; only the actual fresh guard can authorize an allocation.

Recovery needs one small explicit review item: the hardware child and parent already write atomic bounded checkpoints, but `modal_h200.py`'s outer exception recovery currently handles attention screens only. Before termination, a hardware-mode exception should attempt bounded reads of `/tmp/okc-result.json` and `/tmp/okc-hardware-child.json`, keep incomplete status and any errors, and preserve termination behavior. Normal result retrieval and raw bounded NCU stdout/stderr already exist. Do not classify a partial child or failed parse as a completed measurement. Preserve raw evidence for offline parser recovery rather than repeating a GPU run.

## Minimal preflight review and decision use

Keep CPU work limited to extending existing fixed-payload/query/parser/lifecycle tests: new exact pins; optional group complete/absent; real retained wide CSV recovery; finite values and two launches; checkpoint recovery without retry; allocation/termination guards unchanged. No new generic profiling framework. Dry run must perform no provider call. Root owns the final one-run command after source review and credit clearance; the eventual command shape remains `hardware-profile --hardware-issue --repeats 6 --timeout 900` with the selected extra group fixed by the reviewed mode.

Compare current versus old actual route counts/weight requests, measured bytes, DRAM fraction, issue/eligible warps and the existing stalls. Report new backpressure values only if collected. This can prioritize a narrow future investigation; it cannot establish a performance improvement, a critical-path attribution or a benefit from reviving closed scheduling/staging experiments. A source-PC investigation remains a separate decision if aggregate pressure warrants it.

Reviewed file identities before any proposed edits:

| File | SHA256 |
|---|---|
| `tools/modal_hardware_metrics.py` | `ba26f5f43d15ba38e57f01766e85f532aa8fd659fab6e0acfe0d2cc28d1e2ec9` |
| `tools/modal_h200.py` | `60cbfa932ba6904f2a82265d1019eff500a8d9020601d3dc3cd5e2be6e0d51c0` |
| `tools/test_modal_hardware_metrics.py` | `9cee4edf11105f8da695b4e01e8f064f73aadbb390c952f534cece9680bfbef8` |
| `muse_moe_w4a16/reference.py` | `a7e0b4fa86aa477a4e7bb5745b2b258954d72636b6da153bd239a717927cba44` |
| `muse_moe_w4a16/checks.py` | `9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055` |
| `muse_moe_w4a16/nvfp4.py` | `cfdf37c7bc14b10f7353aba7d4632f4767401b92b4b073941a8030d9a961aaaf` |

Retained comparison: `experiments/iterations/20261104-moe-hardware-profile-recovered.json`; immutable source run and raw CSV remain unchanged. Its old parser failure was repaired offline in the current helper, so this plan does not repeat the known wide-output parsing defect.
