# Router score payload experiment, November 4

Frozen candidate `native_route_payload_v20.py` has SHA256
`8ab1cf33ef5b4a4eea8721625ad446358fcbd603dfe1277eee17b7e76a53d232`.
It starts from the exact scored decoder `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`
(public `7a8c4941741645ee`, private `b2a773e2d7194de8`). The active submission
and all CUDA sources, dispatch, preparation, projection, attention and expert
math remain byte-identical. Official private job `99924cdcdb1f4952` completed and this candidate is rejected
by the fixed whole-decoder performance screen.

The two currently selected routing functions, `_route_split` at small token
counts and `_route` at larger counts, each perform eight sequential argmax
reductions followed by eight separate masked sums to retrieve the winning
unbiased sigmoid scores. This experiment replaces each pair with one reduction
of `(biased_score, expert_index, original_sigmoid_score)`. A single new combiner
selects all three fields together. All eight exclusions, ID order, local-ID
mapping, weight normalization, pooled-input RMS normalization, casts, PDL calls
and UV reset operations are untouched. The dormant `_route_a` is untouched.

The predicate is copied exactly from
[Triton v3.8.0 standard.py](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/language/standard.py#L137):
compare the scores, taking the lower expert index on equality. That file's
SHA256 is `113296c74fb8b8dcf67a1b82f71926842eb66d0c77fe2c4f8d0f315715a38fd4`.
Our pinned Modal image reports Triton 3.8.0; this does not independently establish
the official worker's exact installed source. No special NaN rule is added.
The payload is the existing sigmoid result, not `biased_score - bias` and not
a recomputed sigmoid.

The hypothesis is fewer separate collective phases. At the source level, eight
masked-sum reductions disappear. A conceptual 256-element binary tree replaces
255 float additions and the input mask per selection with 255 payload selects.
These are source operation counts, not emitted instruction counts. The original
two-field argmax plus one-field sum and the new three-field argmax can both need
three scalar shuffles per tree step. The extra payload can increase register
pressure. [Triton's reduction lowering](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp)
reduces locally, within warps and then across warps as required by the chosen
layout. The prospective saving is a second inter-warp conversion/synchronization
phase per selection; no claim is made that actual PTX has eight fewer barriers
or that the compiler has not already optimized some parent operations.

The exact97af profile rows are 3.5–3.8 microseconds for `_route_split` and
4.1–5.2 for `_route`. They include PDL waits and are not removable critical-path
bounds. Applying the already retained six-load/46:16 profile-share scenario to
a hypothetical uniform 0.5/1/2 microsecond reduction gives approximately
0.328%/0.656%/1.313% whole-decoder improvement; 1% would require about
1.524 microseconds uniformly. These are conditional scenarios, not forecasts or
an implementation hurdle. A smaller measured gain can still be useful.

The bounded dense-path review found no additional candidate to bundle here:
SK2 decodes each weight fragment before its token-tile loop and reuses it;
UB2 similarly reuses decoded fragments across its token work. Required BF16
round trips and sparse escape corrections cannot simply be removed. Current
MoE NCU math pressure does not attribute time to this separate Triton router.

`python3 tools/test_route_payload_cpu.py` passes exact full-parent restoration,
332 finite FP32 comparisons across both actual extracted loops and two modeled
reduction trees, an independent sorted-index oracle, 32 same-tree NaN
classification comparisons, and two negative controls (wrong tie direction and
wrong payload). It exercises positive zero, subnormals, the normal boundary,
saturated sigmoid values, large finite biases, ties and randomized finite
scores in [0,1]. The report is
`experiments/iterations/20261104-route-payload-cpu.json` (SHA256
`e0ab4720b704ba753f46d8a9e5be97e27f988255d89ad1f202f43dd780f003ee`).

There is deliberately no universal bit-exact claim. The old masked sum adds
positive zeros; direct payload selection does not. The CPU suite retains three
counterexamples/limits: negative-zero payloads (not a finite sigmoid result),
subnormal payloads under a deliberately flushed-add model, and the parent
comparator's NaN dependence on reduction tree. Carrying a payload does not prove
that Triton picks the same tree. The finite CPU proof uses IEEE round-to-nearest
FP32 and does not emulate sigmoid, GPU FTZ, generated layout or NaN payload bits.
The official correctness suite must therefore validate actual compiled behavior.

This is one isolated experiment. Required evidence before promotion is all
13 correctness tests and 12 benchmark/profile cases, comparison with both exact
incumbent receipts, and an actual aggregate improvement. A small gain inside
the reported timing interval is not statistical proof. No resource markers,
tuning changes, separate diagnostic launches or source mutations are bundled.


## Terminal official result

Job `99924cdcdb1f4952` passed all 13 tests, 12 benchmarks and 12 profiles with
clean checks. Score 2.148848476661277, CI [2.1166157495113582, 2.181081203811196].
The exact six-load aggregate ratios are 0.9851756334 against the public parent
and 0.9872253613 against the private parent. Zero of six weighted loads improve
against both. The 2% weighted-load latency cap also fails for multiple loads.
Both reported confidence intervals overlap the candidate's and nodes/seeds
vary; this result does not establish a causal router slowdown.

All twelve profiles show the intended changed route paths: eight `_route_split`
and four `_route`. The eight small-route profile rows are 0.1–0.4 microseconds
lower than both parents, while larger-route rows are mixed. Those non-exclusive
profile measurements do not override the failed whole-decoder performance gate.

Close this isolated attempt: no duplicate retry, public promotion or port.
Public `97af...` remains unchanged. Root independently reproduced the six-load
score and comparisons; independent terminal evidence is
`experiments/iterations/20261104-route-payload-result-critic.json`.
