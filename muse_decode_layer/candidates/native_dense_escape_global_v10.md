# Dense exact escape correction decoupling, V10

Frozen source SHA-256:
`b0387be327a480c84227985566bf086c70d894394f95ad9b0589144c902e260b`.
Independent scored parent: `80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`,
official decoder job `4b216098993e4112`, speedup 2.1501860996x. This candidate
contains no KPERM or rejected attention experiments.

Hypothesis: SK2's exact sparse BF16 correction warps hold back its regular TMA
ring because every correction warp must wait for and release every stage. The
candidate loads the identical BF16 input directly from global memory for those
sparse corrections and removes correction warps from the ring. They still
finish through the original final correction barrier. This affects compressed
single-CTA SK2 projections across small token counts, including the physical
u/v rows used by the fused pooler/router path.

All three/four correction warps, individual FP32 correction planes and their
addition order remain. The compressed weight stream, MMA sequence, ring depth,
launch geometry, shared allocation, PDL wait, fused epilogues and actual
specialization occupancy query remain. Cluster and uncompressed paths retain
the original algorithm. A distinct kernel/extension name identifies this source.

The costs are additional sparse activation gathers, potentially a 32-byte sector
request for each two-byte activation, and an independent correction tail that
could erase the benefit of removing ring waits. No timing or occupancy gain is
established. Actual sparse counts and compiled register use are not known on
the official inputs before the private run. The source-derived resource audit
and exact input/lifetime proof are in `tools/DENSE_ESCAPE_DECOUPLING_PLAN.md`.

The bounded CPU suite passed 2,561,346 patch-entry ownership/order comparisons,
19,673,088 shared-to-global input address comparisons, 448 ring interleavings,
and two mutation controls over 14 geometries and all physical M1..TOKR values.
It confirms AST isolation to the embedded SK2 source and extension name.
`experiments/iterations/20261104-dense-escape-global-cpu.json` preserves the
exact result. This is not evidence of GPU compilation, runtime races,
numerical correctness or performance.

Next authorized GPU experiment, after independent exact-source approval: one
official private full decoder check of these exact bytes. Require all 13
correctness cases and all 12 benchmark cases, no flags/rejections, and profiles
showing `sk2_global_escape_kernel` where selected. The full decoder is needed to
exercise UV, pooler/router fusion and SK2B; standalone projections do not suffice.
Large-token attention/MoE paths are unchanged controls.

The success target is at least 3% whole-decoder improvement with consistent
improvement in the affected small-token cases, supported beyond reported timing
uncertainty. This target would contribute to the remaining gap; it would not
by itself establish first place. Reject promotion on failed correctness, new
flags, a material nearby-shape regression, or no useful gain beyond noise.
Do not rerun a queued job, tune arbitrary escape density thresholds from scored
inputs, or add a new codec merely to rescue this hypothesis. Preserve the
incumbent while the check is pending. Root owns all official submissions.

## Terminal decision — November 4, 2026

**Reject promotion.** Private job `d4c26ebb65e94a84` completed successfully:
13/13 correctness tests and12/12 timed cases passed, with no flags or rejections.
The exact frozen source above scored **2.0876884235×**, 95% interval
2.0563730972–2.1190037499×. This is2.91% below the public parent's2.1501860996×
and3.08% below its earlier private2.1541142783×. Every case/load speedup ratio is
below both parent runs. The proposed≥3% whole-decoder improvement did not occur.

The candidate really executed compressed global-escape SK2 in every small-T
profile; six additional renamed router launches are uncompressed and retain the
old algorithm. Matched qkv SK2 durations increased in all eight small-T captures.
At T1/T8 the tuner also dropped the parent's UV/fused pooler-router path, adding
a separate router and explicit boundary1. T32 o-projection selected cuBLAS.
Those different paths prevent attributing the whole loss to a single instruction
or comparing the changed pooler row directly with the old fused row.

T128/T256 have no SK2 launches, yet their SGLang-relative ratios also declined;
fresh seeds, different nodes, baseline changes and tuning confound precise
cross-job attribution. No device counters isolate global gather traffic,
correction-tail latency, table scans or register pressure. The local log is empty
and the result's log tail contains profiler messages, not compiler resources.
Do not claim a measured spill or exact root cause.

Preserve this failed experiment and the incumbent; no scored submission,
identical retry, or rescue variant. Reproducible analysis:
`tools/analyze_dense_escape_global_v10.py`,
`experiments/iterations/20261104-dense-escape-global-mechanism-analysis.json`, and
`tools/DENSE_ESCAPE_GLOBAL_V10_DIAGNOSIS.md`.
