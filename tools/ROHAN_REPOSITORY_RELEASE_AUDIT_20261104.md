# Rohan repository release audit — November 4, 2026

At09:12 UTC, the public GitHub API still reports default-branch HEAD
`018b7a102348ea3b73701e8be52d24f82d8c91ee`, committed October30 at15:20 UTC.
This is the commit already pinned by this project. The recursive tree has527
entries and is not truncated. All13 public branch heads are dated October30 or
earlier. The latest release is[v1.5.0](https://github.com/rohanarun/Auto-research-agent/releases/tag/v1.5.0),
published October30 at02:37 UTC. No repository code was executed or app built.

**Finding:** useful research coordination and Lean verification infrastructure
is released. The November2 tweet's GPU genetic-search dashboard, billions-of-
configurations performance predictor and Lean GPU physics/memory pruning system
were not found in this public release. This does not establish that the author's
private tools do not exist; it establishes that they are not available here to
reuse today. The tweet's announcement is not a measured result for our kernels.

## What is actually implemented

| Released component | Source inspection | Reuse value today |
| --- | --- | --- |
|[ResearchMethods.py](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Resources/ResearchMethods.py)|`Runner.evaluate` hashes the evaluator and candidate workspace; `choose` ranks independently valid measured scores; `dashboard` reports observed metric diversity; `run` forks branches and continues the leader. Randomness samples dashboard metric subsets.|Source-bound evidence, negative-result retention and branch selection are reusable. This is not a trained GPU performance predictor or parameter-level crossover/mutation engine.|
|[ResearchMethodsPolicy.md](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Resources/ResearchMethodsPolicy.md)|RSI-top views and fork-and-flush are model-configured search policies. The document explicitly distinguishes this adaptation from native OpenEvolve/ShinkaEvolve selection.|Use distinct hypotheses, frozen objective/checkers and retained losing branches. The policy itself does not provide kernel optimization candidates.|
|[ResearchMethods.swift](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Sources/ResearchCore/ResearchMethods.swift)|Optional methods accept Codex or Claude and pass one configured provider/model's CLI arguments to all branches. Normal app runs separately support OpenCode and Antigravity.|Provider integration is real. A released automatic Claude/Astra/Gemini genetic population coordinator is not established by these methods.|
|[ResearchDashboard.swift](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Sources/ResearchCore/ResearchDashboard.swift),[ProjectDashboardView.swift](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Sources/AutoResearch/ProjectDashboardView.swift)|Parses project history, source claims, PR status and metric timelines. Method telemetry separately records branch events.|Useful display components, not evidence of hardware simulation or billions of predictions. Building the app does not improve GPU kernels.|
|[LeanSearch/core.py](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Resources/LeanSearch/core.py),[server.py](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Resources/LeanSearch/server.py),[cli.py](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Resources/LeanSearch/cli.py)|Pinned plans, source hashes, build/axiom/Comparator stages, job handles and a shared verification service. Server initialization generates a `1+1=2` smoke theorem.|Can host mathematical proofs that we supply. The profile named `kernel-check` refers to Lean's trusted kernel, not a CUDA kernel resource model.|
|[LeanSearch/cpu_search.py](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Resources/LeanSearch/cpu_search.py),[qrh_tau_search.py](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/Resources/LeanSearch/qrh_tau_search.py)|Finite rational parameter grids and exact conditional mathematical certificates for quasi-RH. The CPU entry rejects adapters other than `qrh-rational-grid`.|The manifest/checker/retained-outcome pattern transfers. The actual certifier and search domain do not optimize H200 kernels.|
|[benchmarks/gpu/timing_stats.py](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/gpu/timing_stats.py)|A10,000-resample paired timing bootstrap over already measured cases.|Useful statistical reporting, not prediction of unmeasured configurations. Official confidence intervals remain authoritative for our submissions.|

The workflow source scan covered all101 relevant text files under Resources,
Sources, scripts and current docs (681,627 bytes), plus README, licenses and
selected benchmark files. No `.lean` source files are present in the main tree;
the verification server generates its smoke files or operates on external
operator-supplied theorem projects. No GPU hardware-model/proof or performance-
surrogate implementation was found in the scanned application/tool sources.

The only branch head newer than main's commit time is
`claude-cohort-20261030` at4905fc6c. Its tree differs from main only in
`benchmarks/RESULTS.md`, two Pareto chart files and `benchmarks/tools/chart.py`.
This is a publication-chart update, not a new optimization engine or kernel.

## Kernel usefulness and licensing

The strongest published native candidate family remains the Gemini/vjeux-derived
attention, MoE and decoder source we already audited and adapted. HEAD has not
changed since that import. Our current native entries retain that lineage, with
our correctness/packaging fixes and subsequent measured KPERM change. There is
no newly released kernel here demonstrated to improve the current incumbent.

The[benchmark results](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/RESULTS.md)
are local H200 measurements under a separate protocol, explicitly not official
OKC scores. Their best listed decoder2.1724x or MoE2.3768x does not establish a
win against our official2.18118x/2.37369x. Prior review also found suspiciously
repeated attention evaluation timings; preserve that uncertainty rather than
treating the report as independent GPU evidence. Other model cohorts do not
report a stronger decoder/MoE candidate than the already imported family.

The repository's[MIT license](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/LICENSE)
permits reuse of its original workflow code with the copyright/license notice.
The[benchmark NOTICE](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/NOTICE.md)
does not relicense upstream problem packages or kernel sources. Preserve the
Apache/CUTLASS and other upstream notices in kernel adaptations. No code was
executed, provider session started or GPU allocated during this audit.

## Concrete recommendation

Reuse the small, transparent part of the idea: enumerate a bounded configuration
space, reject provably impossible resource/lifetime combinations, retain every
reason and send only a justified survivor to actual GPU measurement. Our current
WGS queue hypothesis already has an executable lifetime checker; wrapping that
checker in a manifest-driven CPU search provides immediate value without waiting
for an unreleased predictor or claiming a Lean proof. Actual compiled registers
and performance must remain unknown until measured.

Do not install the whole app or start a second model-runner fleet merely to
reproduce coordination already available here. Its `snapshot` implementation
uses full `copytree` copies, so direct deployment would also need a compact seed
and careful integration with our existing budget, artifact and single-GPU rules.

Audit evidence is saved in
`experiments/iterations/20261104-rohan-repository-api-audit.json` and
`experiments/iterations/20261104-rohan-workflow-text-manifest.json`.
The pinned text cache contains105 files totaling717,086 bytes including reused
license/benchmark files. It is bounded, source-only and was not executed.
