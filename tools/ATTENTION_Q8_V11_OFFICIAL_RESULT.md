# V11 official result: reject for performance

Private job `88f84f79fb11462d` passed16/16 tests with no flags or static-check
rejections, and completed12 benchmarks plus12 profiles. Receipt and frozen
uploaded file both identify exact wrapper SHA256
`6724284b1e8aef03a3af38d1d9d008c1df680b241a37b2d688464b8eeb60633f`.
No source changed during this audit.

Aggregate speedup is2.270900, CI[2.236836,2.304963], versus active87a9's
2.611970, CI[2.572791,2.651150]. This is a comparison of different wrapper
policies, not an isolated measurement of QK/PV ownership.

## Per-case comparison

Indices follow `ATTENTION_Q8_V11_OFFICIAL_COMPARISON.md`. Times are official
reported µs. Relative paired speedup is V11's same-job SGLang paired ratio
divided by the historical active87a9 ratio; values below1 favor the incumbent.
It is still a cross-job/different-flag comparison, not a direct paired test.

| Index | Batch / Q length / layer | Active87a9 µs | V11 µs | Latency change | Relative paired speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 1 / 1 / sliding | 4.9192 | 5.2956 | +7.65% | 0.946566 |
| 1 | 1 / 1 / full | 10.7860 | 11.1483 | +3.36% | 0.973451 |
| 2 | 16 / 1 / sliding | 7.5785 | 7.7704 | +2.53% | 0.987192 |
| 3 | 16 / 1 / full | 86.7566 | 87.6559 | +1.04% | 0.986823 |
| 4 | 32 / 1 / sliding | 9.8504 | 10.2627 | +4.19% | 0.975673 |
| 5 | 32 / 1 / full | 166.1107 | 167.3227 | +0.73% | 0.990856 |
| 6 | 1 / 8 / sliding | 6.5373 | 7.5306 | +15.20% | 0.887950 |
| 7 | 1 / 8 / full | 14.2501 | 14.9438 | +4.87% | 0.961221 |
| 8 | 16 / 8 / sliding | 11.2186 | 12.0608 | +7.51% | 0.938872 |
| **9** | **16 / 8 / full** | **100.4905** | **151.3042** | **+50.57%** | **0.662660** |
| 10 | 32 / 8 / sliding | 14.7387 | 15.4073 | +4.54% | 0.953044 |
| **11** | **32 / 8 / full** | **183.1197** | **276.7057** | **+51.11%** | **0.661622** |

The intended ownership targets9/11 regress substantially in all measurement
views. Their raw graph medians change104.176→155.008µs and187.200→281.056µs;
their separate kernel profiles change100.3→147.4µs and186.0→274.9µs. Both retain
one `attn_kernel<8,true,0>` call. Historical exact decoder-parent attention
profiles were105.2/186.1µs on soft decoder inputs; those are context, not a
matched baseline for the new coherent-input profiles.

The target SGLang baseline times were nearly unchanged across jobs:
205.184→204.720µs and389.360→389.264µs. Both jobs used the same evaluator code,
all listed evaluator-file hashes, judge image and driver580.159.03. They ran
on different H200 nodes with reported1980MHz SM/3201MHz memory clocks. These
facts strengthen the negative performance observation; they do not remove
wrapper-policy and input-seed confounding.

All ten controls also have lower relative paired ratios. Most changes are
small compared with the targets; B1 Q8 sliding is notably15.2% slower by
reported latency. The wrapper's QDIRECT0, PK2=0, W0ENABLE1 and small-T PDL
policy differ from active87a9, as documented in the checklist. Consequently
do not attribute every regression, or an exact fraction of the target loss,
to role ownership or the88-byte local footprint. Do not normalize targets by
control ratios. No spill-location or dynamic-stall evidence was collected.

Target-path correctness tests5/7 passed with relative L2 errors0.08890 and
0.08092 under0.15 bounds. Scored coherent target replays also completed.
The saved-suite coverage limitations remain: the new TP8/full/Q8 scale1/64
path and soft decoder-input performance were not directly established by
this standalone suite. Other test results and unchanged-control profiles
introduce no new correctness failure.

**Decision:** reject V11 for performance. The official run supplies numerical
evidence for its covered inputs, but fails the proposed useful target-gain
criterion by a large margin. It does not justify a V12, a whole-decoder
follow-up, integration or promotion. Preserve the earlier local resource
rejections and the partial allocator-visibility finding separately. Root owns
the terminal decision JSON; this file is the compact per-case interpretation.

Evidence: `experiments/runs/88f84f79fb11462d.json`, compared with
`experiments/runs/8f9e5971f95841a0.json` and
`experiments/runs/4b216098993e4112.json`.
