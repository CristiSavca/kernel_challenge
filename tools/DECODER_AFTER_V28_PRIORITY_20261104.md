# After V28: preserve the public decoder; prioritize serving integration

November 4, 2026. **No further high-value public-kernel candidate is evidence-ready from this bounded review.** This is a prioritization decision, not a proof that the public kernel cannot improve. V28 is closed; do not tune or retry it. The next useful experiment is actual V27 serving-call integration validation, after its separate source review and resource authorization. It cannot improve the public decoder leaderboard by itself.

## What the new measurement establishes

V28 job `2976c25cc3dc41b1`, exact source `d1b1bcf5…36d9b`, passed correctness and activated the renamed MAIN kernel, but scored 2.114626680x. Against public97af and private97af respectively:

| Metric | Public parent | Private parent |
|---|---:|---:|
| Whole-score ratio | 0.969486 | 0.971503 |
| Affected three-load speed GM | 0.951014 | 0.954594 |
| Control three-load latency GM | 1.011821 | 1.011417 |

The raw profiles show MAIN increases in all six affected cases. Sliding T32 is 80.3→102.8µs; sliding T128 is 143.9→179.1µs; sliding T256 is 205.8→210.3µs. The unchanged T8 sliding MoE profile also moves 51.8→64.6µs while its normalized benchmark is nearly unchanged. Therefore profile differences are not isolated causal timing: inputs/routes differ, PDL can overlap, and rows include waits. The failed fixed screen is decisive; it does not establish that scratch size, additional FP32 adds, register allocation or atomic service caused the regression. No V28 compiled resource measurement was retained.

Parent resources remain separate evidence: current247888 standalone MAIN has 96 registers and zero local bytes in the retained compile diagnostic. Current NCU M32/M256 reads are 1.008754/1.007692 times nominal packed-weight requests; reported DRAM throughput is 77.30/76.99%, math-pipe throttle 10.21/10.96%, MIO 0.07/0.35%, and LG 0.07/0.52%. These are aggregate standalone diagnostics, not decoder critical-path bounds or recoverable fractions. They supply no new premise to reopen shared-A, prefetch, wait-distance, register, SC6 or vector-reduction variants.

## Why the remaining broad dense mechanism is not ready

The concrete untested representation is the existing eight-entry exponent dictionary per 8×64 BF16 unit, with exact raw-unit fallback, documented in `DENSE_UNIT_DICTIONARY_FEASIBILITY_20261104.md`. It would replace the SK2 matrix-window approximation plus separate FP32 escape correction with direct reconstruction of every original BF16 weight. It removes correction warps and their table scans/handbacks, rather than merely moving those corrections to global activation reads.

Its measured scope is meaningful: current SK2 accounts for 18.83–44.38% of profile time in its eight cases. But source accounting gives approximately 3% more requested weight bytes in a distribution proxy and 15 rather than 12 integer decode operations per eight weights, plus dictionary/header loads and mode branches. Actual selected-weight cardinality/fallback-stage distributions are absent. Proxy eligibility is not an official-weight measurement.

The diagnostic-only97af resource records show QKV dynamic shared memory 167,776/182,096/219,200 bytes for observed physical M1/16/32, registers 76/71/101, zero local allocation and block admission one. Fused pooler examples use 206,784/214,960 dynamic bytes, 88/87 registers, zero local allocation and admission one. These are first observed post-prepare identities, not exhaustive benchmark-instance attribution. Removing correction planes at unchanged geometry does not make those important paths fit two CTAs per SM; the prior dictionary audit already includes the header growth. A new smaller ring or row geometry would add another unmeasured premise and risk the fusion/resource failures already observed in R16.

A profile-scaled scenario that reduces **all SK2 time** by 20% yields 1.049735x whole score (4.7378% less latency); 30% yields 1.077063x. Deleting all SK2 time gives an unrealistic 1.336901x scenario. Matching the saved 2.576163x leader from current2.181183x would require about 62.77% SK2 reduction in that model. These are scope scenarios using exact six-load 46:16 aggregation, not forecasts or physical ceilings. No current evidence supports even the 20% reduction. The source-stage/fallback complexity is substantial enough that this should not be implemented solely to fill the remaining kernel deadline.

The smaller untested QKV row-balancing donor is source-compatible but has only about 0.13–0.28% modeled whole-score scope for its claimed local gain; actual current escape distribution and critical tail are unmeasured. The Q1 `ldmatrix` value-fragment lead still lacks a byte/lane mapping that eliminates more work than its required permutations. Boundary fusion has no new once-only owner mapping: route→WGS scatter needs global job ranks, attention→O repeats reductions or needs publication, and moving unpool into experts crosses normalization/BF16 boundaries. None has new evidence after its saved negative feasibility review.

## One concrete next experiment, outside the public-kernel lane

Validate the already implemented V27 pre-Marlin installer and actual `_forward_marlin` branch against pinned stock SGLang. Source boundaries are `process_weights_after_loading`/`try_install_native` in the quantization method, `_forward_marlin` around lines578–602 of the isolated candidate, and `native_forward` in its owned kernel module. The adapter replaces the packed bank, executes the reviewed V26 expert core, retains actual pre/post norm metadata and the stock finalizer, and chunks large calls. This is different from V28: the stock Marlin serving chain is the comparator, not the faster public97af/247888 kernel.

The V26 1.685110x stock component score supplies a real reason to test the integration, but not a full-serving speedup. The next check must exercise the **installed call path**, bank ownership and execution lanes, not repeat the unchanged component benchmark. At minimum retain independent stock comparison, generic rank and all-remote→live reuse, changed-input graph replay, separate-workspace concurrent streams, the384/385 chunk boundary and representative large prefill. Include all allocation/copy/finalization work. T16384 needs43 chunks under the current bound; TTFT and full-model memory remain open.

Falsifier: any stock-contract numerical, graph/lifetime/alias, rank/empty-route or memory-ownership failure rejects the integration as deployable; a correct but slower complete chain or unacceptable prefill cost rejects its performance claim. No relaxed tolerance or model-loader edits. Root must separately fix a bounded session and gate that fit available resources before allocation. The reviewed source is available, but no amount of CPU proof substitutes for this device check.

There is no defensible whole-serving ceiling without the actual model's expert-time fraction and communication/TTFT traces. The **public decoder score ceiling of this serving-only experiment is exactly no change**, because it does not alter the public submission. The saved leadership gap remains +18.1085% score/−15.3321% latency. Favoring serving integration acknowledges the separate prize objective and available evidence rather than claiming the kernel goal has been reached.

Evidence: `20261104-decoder-v28-root-result.json`, its pinned raw receipt/profile; exact97af public/private profiles; current selected-resource and current-KPERM NCU analyses; `DENSE_UNIT_DICTIONARY_FEASIBILITY_20261104.md`, `DENSE_LOSSLESS_STRUCTURE_REVIEW_20261104.md`, `ATTENTION_NEW_KV_PACKING_FEASIBILITY_20261104.md`, `DECODER_BOUNDARY_REVIEW_20261104.md`; and the isolated V27 manifest. Exact hashes and computed scope values are in the paired JSON. No candidate edits, GPU/provider/official calls or commits were made for this review.
