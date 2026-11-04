# Dense global escapes V10 — rejection and source diagnosis

2026-11-04. Exact candidate b0387be327a480c84227985566bf086c70d894394f95ad9b0589144c902e260b;
private job d4c26ebb65e94a84. **Reject promotion; preserve80a4.** All13 tests and12
benchmarks passed without flags. Speed2.0876884× is below public2.1501861× and
private2.1541143×. Independent critic owns the final result verdict.

Reproduce the saved comparisons with `python3 tools/analyze_dense_escape_global_v10.py`.
It reads completed local receipts only and saves
`experiments/iterations/20261104-dense-escape-global-mechanism-analysis.json`,
including exact source/receipt/profile hashes, each benchmark, kernel mapping,
matched template durations, and both parent comparisons.

## What ran and changed

All eight T1/8/16/32 profiles contain the renamed kernel:26 compressed CL1
launches execute the new correction path. Six uncompressed router launches have
the new name but the unchanged algorithm. T128/T256 contain no SK2.

| Load | SGLang-relative load speed change vs public | vs private | Dispatch evidence |
| --- | ---: | ---: | --- |
| decode1 | -3.64% | -4.13% | Parent UV/fused pooler-router replaced by separate pooler/router plus boundary1. |
| decode16 | -3.10% | -4.26% | Same four compressed template prefixes, including fused pooler/router and unpooler/boundary. |
| decode32 | -3.29% | -3.05% | Qkv/pooler/unpooler remain SK2; o changes from SK2 to cuBLAS. |
| verify1 | -4.04% | -4.87% | Parent physical2T UV pooler replaced by ordinary T8 pooler/router plus boundary1. |
| verify16 | -0.64% | -1.31% | No SK2; large-T control, with tuner/backend differences. |
| verify32 | -2.69% | -0.82% | No SK2; large-T control, with tuner/backend differences. |

Every one of12 case speedup ratios is below both parents. Target case-ratio
geomean is0.96597 versus public /0.95844 versus private; control geomean is
0.98421 /0.98894. These descriptive cross-job ratios are not paired experiments
between our implementations. All three jobs use the same evaluator but different
nodes and fresh seeds. The candidate's large-T normalized microseconds improve
while its SGLang-relative ratios decline because baseline times also change.

Matched qkv captures are more specific evidence: public→candidate µs are
10.7→12.0/11.8 (T1 sliding/full),10.5→12.7/12.3 (T16),11.7/11.0→12.4/12.4
(T32), and10.3/10.5→12.6/12.2 (T8). All increase. These rows have the same
template prefixes, but PDL waits and weight/seed differences still prevent a
precise isolated kernel-cost estimate. T1 o looks slightly faster, and total
MoE durations also vary substantially, so no uniform component regression is
established.

Profiles support an actual dispatch consequence of the experiment: losing UV
adds a boundary kernel and separate router. They do not expose the timed tuner
comparisons that selected it. The T1/T8 benchmark captures therefore do not
individually validate the unused UV candidate path, despite the whole-suite pass.
No compiler/register/pmax/counter evidence is available: the saved `.log` is
empty, while the result log tail contains profiler-start/stop messages only.

## Escape-table scanning: correction to the hypothesis

Let E be one CTA's escape count and N=NESC. The **parent already traverses every
entry in every escape warp**. Non-owning stages advance `pe` using each entry's
index; owning stages load the correction. Its successful pointer increments are
N×E, plus repeated boundary lookaheads. V10 also traverses N×E; it did not newly
introduce a3–4× scan factor.

Ignoring lookaheads, source-level metadata payload is4(N+1)E bytes in the parent
(one8-byte owned entry plus N−1 index-only4-byte reads), versus8NE if V10's full
`int2` loads materialize before ownership selection. For N3 this is16E→24E;
for N4,20E→32E. V10 also derives stage and modulo ownership per entry instead of
once per stage. Compiler predication/load sinking may reduce this difference;
warp-uniform shared reads broadcast, so multiplying by32 would be wrong. These
are source-operation estimates, not measured bandwidth or instruction counts.

V10 additionally replaces the parent's shared activation load with a sparse
global BF16 load and removes ring pacing. Extra sectors, cache behavior, or an
exposed correction tail could outweigh removed waits. The present data cannot
identify which effect dominates. Reject this implementation without claiming a
causal hardware diagnosis.

## Separate feasibility observation, not a rescue implementation

Immutable weight metadata could identify each stage's entry interval, letting
only its owning escape warp scan it while retaining the original shared-input
correction and ring handoffs. Stable filtering of the existing K-sorted table
preserves each plane's FP32 order. It must not repartition one table by a single
NESC: the `pooler` table is used with NESC4 at TOKR8/16 and NESC3 at32.

A universal K64 directory costs4(K/64+1) bytes per CTA before16-byte padding:
68/260/516 bytes at K1024/4096/8192. A KST-specific directory is smaller
(20–132 bytes across active configurations) but requires geometry-specific
metadata selection. Neither saves activation traffic. Actual escape counts
were not recorded; sparse/empty tables can lose from directory overhead, so a
new proposal must retain the exact parent path for unprofitable immutable tables.
See the separate stage-directory plan; no new candidate or GPU run is authorized
by this diagnosis.
