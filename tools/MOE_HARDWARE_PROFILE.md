# Bounded standalone MoE hardware diagnostic

This diagnostic selects evidence for a next experiment. It is not a candidate,
an official score, a decoder profile, or promotion evidence. It uses the exact
frozen standalone incumbent, without modifying native source or compile flags:

- Python SHA256: `ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a`.
- Embedded CUDA SHA256: `af4f0dba97e347f8afa8f27508e40d5c03f60b402825cadba9a9537700b20d6e`.
- Decoder lineage only: `80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`.
  Standalone has `PDL=False`; decoder PDL/concurrency is not represented.

## Fixed experiment and command

```bash
# Validation only: no billing query, lock, upload, or allocation.
.venv-modal/bin/python tools/modal_h200.py hardware-profile --hardware-issue --timeout 900 --dry-run

# Root-owned execution, only after independent review.
.venv-modal/bin/python tools/modal_h200.py hardware-profile --hardware-issue --timeout 900
```

There are exactly two cases, M32 then M256, `ep8/normal/busiest`, weight seed 7311.
Profile input seeds are fixed at 7302 and 9302; actual IDs and local 32 histograms
are saved, without assuming earlier U/J observations recur. Before profiling,
each case receives two fresh eager checks and six fresh graph checks/timings.
The ordinary graph inputs use 7302..7307 and 9302..9307. `--repeats` may reduce
this to 2..6; source, cases, seeds, and metrics cannot be supplied arbitrarily.

The NCU child prepares the incumbent once and retains both captured graphs and
the incumbent scratch across replays. Every ordinary timing finishes before
the first `cudaProfilerStart`. Immediately before and after each profiled replay,
the exact profiled input is checked against the unchanged public oracle. Stop
is in `finally`. There is no benchmark baseline/candidate duplicate preparation.
These contextual timings still run in an **NCU-injected process** and are not
independent benchmark latency evidence. The profile durations are never used
for promotion or compared as speedups.

The ordinary timings use a 128 MiB read sweep with checked observable sums. NCU
uses `cache-control all`, which flushes GPU caches for kernel replay. These are
different cache conditions, neither established equivalent to the private judge.
Profiler replay may serialize kernels and perturb duration. Counter collection
is therefore labeled **controlled-cold standalone only**.

## Counter scope and acceptance

Version and full suffix metric-query output are saved before collection. The
required group is:

```
dram__bytes_read.sum
dram__bytes_write.sum
gpu__time_duration.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
```

With `--hardware-issue`, the following fixed group is included only when every
name is available. Otherwise the whole group is omitted and missing names/query
failure are retained. No retries or substitutions are made.

```
smsp__issue_active.avg.pct_of_peak_sustained_active
smsp__warps_eligible.avg.per_cycle_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_wait_per_warp_active.pct
```

The collection command uses the fixed `/usr/local/cuda/bin/ncu`, no shell,
`--profile-from-start off --target-processes application-only --replay-mode kernel
--cache-control all --clock-control none --graph-profiling node
--kernel-name-base demangled --kernel-name regex:.*okc::moe_fused.*
--launch-count 2 --kill no --rename-kernels off --disable-extra-suffixes`, explicit
metrics, CSV/raw/base units, and the fixed uploaded helper child path. There are
no sections/full sets/source instrumentation. The complete actual argv is saved.

Requested pass count is null: NCU chooses passes for the fixed metric list.
Actual positive pass counts must be present for both launches: either the
`profiler__replayer_passes` column in raw-wide CSV or progress text accompanying
long CSV. This is bounded
by metrics, two matching launches, subprocess runtime/output, and sandbox lifetime;
it does not claim a user-enforced maximum pass count. Missing/ambiguous progress,
missing or duplicated metrics, wrong kernel identities, nonfinite values, wrong
metric units, nonpositive duration, or an incomplete child make the
diagnostic unavailable. IDs sorted by launch order map to M32/M256 only after the
child confirms both completed checked windows in that order.

Saved results include actual route IDs, U/J, all local counts, job row sizes,
padded rows, contextual timing/checks, output hashes, exact source/helper hashes,
actual main/extension defines, PDL, prepare time/memory, version/query commands,
collection argv, stdout/stderr, actual passes, units, and measured read/write
bytes. `27MiB×U` and `27MiB×J` are logical packed-weight comparisons, not expected
DRAM-byte identities: reads may include cache effects, scratch, activations, or
metadata. No full 32-expert assumption is used.

## Interpretation and lifecycle

High measured DRAM throughput plus traffic consistent with useful weights would
weaken a major consumer-issue rewrite. Low DRAM throughput alone does not support
one. Eligible-warps/issue/tensor/stall counters can describe a service limitation,
but aggregate stalls are not additive critical-path fractions. This first group
does not establish source-PC attribution, exposed dequant time, or an SS-WGMMA
rewrite's speedup. A later code hypothesis must explain all observed evidence;
no speculative large rewrite or counter sweep follows automatically.

The mode uses the unchanged mandatory credit/overlap guard in `modal_h200.py`:
one H200, four CPU cores, 32 GiB, at most 900 s, blocked sandbox network, one allocation attempt,
and confirmed termination in `finally`. The helper's total subprocess budget is
the sandbox timeout minus 40 s. Version/query commands each receive at most 20 s;
combined collection output is capped at 512 KiB and child JSON at 1 MiB. Timeout or
output overflow kills the owned process group. Normal pipe EOF waits for process
exit within the remaining deadline. Permission failures are recorded without
changing driver permissions or retrying the allocation. Partial child evidence
is retained. Only five named payload files plus the runner are uploaded.

CPU validation:

```bash
.venv-modal/bin/python -m unittest discover -s tools -p 'test_modal_hardware_metrics.py' -v
.venv-modal/bin/python -m unittest discover -s tools -p 'test_modal_h200.py' -v
```

Tests cover argument/hash/seed constraints, metric completeness and launch mapping,
route formulas, optional omission, permission failure/no retry, subprocess timeout,
output cap, EOF-before-exit, guard/lifecycle, upload boundaries, and no-allocation
dry run. They do not establish CUDA compilation, oracle correctness, installed
NCU flag compatibility, metric availability, or counter permission.

Primary documentation: [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
defines metric suffix queries, profile start/stop, graph-node/kernel replay,
matching-launch counts, cache/clock controls, and CSV output.
[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) explains
replay perturbations, memory restoration, caching, and stall interpretation.

## First capture and offline recovery

The terminal run `experiments/modal/20261104T054636Z-ea14816c` completed one prepare,
20/20 numerical checks, two profile windows and four replay passes per launch.
Its original parser rejected NCU 2025.3.1's actual output: a 280-column wide header,
one units row, and two data rows. Duration units are `ns`; replay counts appear
in `profiler__replayer_passes` with unit `pass`. All 11 requested metrics are
present for IDs 0/1. The captured output is preserved byte-for-byte as the CPU
regression fixture `tools/fixtures/ncu_2025_3_1_raw_wide.csv`, SHA256
`b471c2c36016f86bfbfc050f82924a925857f26ff4f5b8c7ae89095cced3a598`.

The working parser now accepts both layouts and checks unique columns/IDs,
complete rows, explicit units, finite values, and positive integer pass counts.
Recovery required no GPU rerun. The separate interpretation artifact is
`experiments/iterations/20261104-moe-hardware-profile-recovered.json`.
Original result SHA256 `398a89f345b90b47937ef06fb44e541dbb0bcf61d334843ff25fbdb34d3a6f43`,
manifest and frozen executed helper remain unchanged; the original result still
correctly records the then-current parser failure.
