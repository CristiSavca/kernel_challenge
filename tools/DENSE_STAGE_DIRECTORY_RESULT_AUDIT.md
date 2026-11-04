# Stage-directory V11: correct, active, no useful gain

2026-11-04. Offline audit of private job `4b9f33ffa6bc4eed`; no official calls.
Exact source `15a7188b3e35b469565cc2d7eb9b37c0804e341dab5f1a4939a16563efdaa172`.
Decision: **reject promotion and combination with KPERM**. Preserve the measured
incumbent; do not tune the immutable gate from these timings.

All 13 correctness tests and 12 timed cases passed, with 12 profiles and no flags
or rejections. Speedup is **2.1492503148x**, CI [2.1170115601, 2.1814890695].

| Comparison | Exact source | Speedup | Candidate speedup change |
| --- | --- | ---: | ---: |
| Original public `4b216098993e4112` | 80a4 | 2.15018610x | -0.044% |
| Original private `f8e991f1a2c2469e` | 80a4 | 2.15411428x | -0.226% |
| Current public `7a8c4941741645ee` | 97af | 2.18118314x | -1.464% |
| Current private `b2a773e2d7194de8` | 97af | 2.17665445x | -1.259% |

Intervals overlap. The current97af comparisons are context, not an isolated
directory ablation:97af also has KPERM. All jobs share the evaluator hash.
The directory and current public97af ran on the same reported node; the other
parents used different nodes. All jobs used fresh case seeds and preparations.

## Activation and resources

The saved `.log` contains nine `SK2_STAGE_DIRECTORY` preparation records; they
match the corresponding lines in the receipt's longer `log_tail`. Four tables
selected the directory, all passing the conservative230400-byte dynamic budget:

| Table | KST | CTAs | Mean/max escapes per CTA | pmax before/after | Added bytes per CTA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooler R32, fused geometry | 512 | 128 | 36.31 /48 | 50 /60 | 80 |
| Pooler R32, T32 geometry | 256 | 128 | 36.31 /48 | 50 /68 | 144 |
| QKV R24 | 512 | 96 | 26.20 /46 | 48 /58 | 80 |
| Unpooler R64 | 256 | 128 | 26.20 /45 | 46 /56 | 80 |

The two o tables, pooler R16, unpooler slice R8 and unpooler R32 retained their
original table layouts. Selected directories added46,592 bytes across immutable
tables. The unpooler R64 work heuristic had only208 aggregate logical bytes of
surplus; this is source-level metadata accounting, not measured traffic or a
latency bound. No register, spill or actual occupancy measurements are present.

Mapping the logged geometries to the frozen source and reported templates shows
**18 active-directory launches**: eight QKV, six fused pooler/router, two separate
T32 pooler and two T32 unpooler. Ten compressed launches use unchanged-table
fallback (eight o, two T16 unpooler); two uncompressed routers are unchanged.
The four large-T captures have no SK2. The kernel rename alone would not have
established activation; the preparation records do.

## Per-case evidence

Values below compare each job's paired-baseline speedup; positive means better.
They are not raw candidate-latency percentages, because baseline timings differ.

| Tokens / layer | Candidate normalized us | vs80a4 public | vs80a4 private | vs97af public |
| --- | ---: | ---: | ---: | ---: |
| 1 sliding | 71.69 | +2.13% | +1.61% | +0.02% |
| 1 full | 78.07 | +1.56% | +1.05% | +1.41% |
| 16 sliding | 130.30 | +1.70% | +0.14% | -0.62% |
| 16 full | 210.96 | -0.75% | -1.30% | -0.33% |
| 32 sliding | 154.16 | -0.62% | +1.25% | -5.04% |
| 32 full | 301.47 | +4.64% | +2.88% | +2.62% |
| 8 sliding | 112.64 | -1.76% | -1.62% | -0.42% |
| 8 full | 117.36 | +0.33% | -3.11% | -1.33% |
| 128 sliding | 250.13 | -1.25% | -2.16% | -3.36% |
| 128 full | 334.55 | -1.85% | -2.00% | -3.27% |
| 256 sliding | 331.77 | -2.72% | +0.06% | -3.65% |
| 256 full | 486.12 | -0.76% | -0.59% | -0.87% |

Eight targeted small-T case ratios have geometric means +0.884% versus public
80a4 and +0.098% versus private80a4. The four unchanged large-T controls are
-1.650% and -1.177%, respectively. This mix does not isolate a tiny positive
mechanism effect from run variation and does not meet the3% aggregate target.

Every reported dispatch matches public80a4 after comparing the visible SK2
template prefix rather than its longer renamed/truncated text. T1/T8 retain UV
and the fused pooler/router; T16 retains its fused pooler/router; T32 retains
separate pooler/router and explicit boundaries. There is no lost fusion like
the rejected global-input V10. Later template arguments are absent from the
official reports, so this is not proof that all hidden runtime choices match.

QKV profile durations are10.1–11.8us and mixed against the parents. Examples:
T1 sliding10.5 versus10.7/10.6us; T16 full10.1 versus10.5/10.5us; T32 full11.8
versus11.0/11.7us. T32 unpooler19.3/19.2us is slower than public18.7/18.4us,
but faster than private19.6/19.7us. Pooler changes are also small and mixed.
These rows include possible PDL waits; they cannot be added as independent
latency savings. The result supports no large reduction in the targeted paths.

The only reported dispatch differences versus private80a4 are large-T controls:
candidate T256 uses `rq_deep_tma_pooler`, while private80a4 uses a cuBLAS plan.
Current97af also differs in large-T dense autotuning. Those changes and the
unmodified-control movement preclude attributing total score changes solely to
the directory. Preserve the correct candidate and evidence as a failed gain
hypothesis; no publication, repeat upload or rescue variant is justified.

Reproduction: `python3 tools/analyze_dense_stage_directory_v11.py` reads only
saved receipts and profiles. Full per-case kernels, provenance hashes, table
records and all four comparisons are in
`experiments/iterations/20261104-dense-stage-directory-result-audit.json`.
