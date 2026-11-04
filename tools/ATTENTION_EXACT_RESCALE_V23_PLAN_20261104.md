# V23: exact-one Q1 output-rescale bypass

One isolated implementation authorized by root on November4. No GPU run,
official submission or active-source replacement has been performed by the
author. This is a counterexample test of a small work-removal hypothesis, not
a forecast of1% whole-decoder gain or first place.

Parent: `muse_attention_decode/submission.py`, frozen source
`87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c`.
Candidate: `muse_attention_decode/candidates/exact_rescale_v23.py`, SHA256
`c6b90f168bfdc21a8583b6a4bd03cebc0e6c4e03f994f74759c54ab5b677335c`.
The first-line problem header and upstream MIT/Apache notices are preserved.
The extension namespace is `okc_attn_exact_rescale_v23{tag}_{arch}`.

## Exact delta and work

Inside `segment`'s `if constexpr (L == 1)`, `#else` of `OKC_LAZY1`, and
`else` of `if constexpr (ONE)`, retain the row-sum expression and surround
only the existing loop with a scalar conditional:

```cpp
l_r[0] = l_r[0] * alpha + ls;
if (alpha != 1.f) {
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    o[0][t][0] *= alpha;
    o[0][t][1] *= alpha;
  }
}
```

The candidate preserves the parent's actual one-line loop body; the display
above expands it only for reading. Alpha is the original computed value, not
a tolerance or a substitute maximum comparison. No warp vote, synchronization,
new data access, probability cast, MMA, row-sum, maximum/exponent computation,
launch geometry, tuning choice, ring lifetime or precision flag changes.
The Q8, LAZY1 and ONE paths remain exact parent bytes. Every input key and value
still participates in attention. This conditional does not skip an attention
block based on its contribution.

The released SM90 [PipeLink role_core.cuh](https://github.com/PKU-DAIR/PipeLink/blob/4d8ebab9f1527589e1cfd892a8a77a3b6211991a/campaigns/attention/src/sm90/role_core.cuh)
uses an exact-one scalar check around accumulator scaling. We implement a local
guard; no deferred-rescale scheduling or WGMMA rewrite is imported. Our current
LAZY1 branch changes the softmax offset/FP16 rounding and remains untouched.

At most16 FP32 source multiplies per thread/tile are omitted. A comparison and
conditional execution are added. Alpha is quad-shared, not necessarily
warp-uniform, so different row groups can diverge. The compiler may predicate
the multiplies or change code layout; no register, instruction or speed claim
is established by this source change. Current attention SASS is unavailable.

## Numerical and control boundaries

For finite O values, ordinary IEEE multiplication by positive one preserves
the value and signed zero, including subnormals when the operation is not FTZ.
Current build flags contain `-O3` without global fast-math or FTZ. The explicit
FTZ exponent operation does not establish the lowering of this separate O
product. GPU subnormal handling and generated control flow remain official
test concerns; do not claim universal bitwise equality.

AlphaNaN satisfies `!=1.f`, so that case keeps the original product. Alpha0
on the first live finite tile also keeps the product. Existing all-invalid
row control flow is unchanged. No collective has been placed inside the
quad-uniform `m_new != -INFINITY` branch.

Nonfinite O payload equivalence is explicitly **not** asserted. The actual-loop
host test observes a difference at signaling-NaN bits`0x7f812345`: multiplying
can quiet it, whereas skipping copies it. Both remain NaN, which cannot be a
valid finite output under the task contract. This is a documented boundary of
the bitwise claim, not a reason to omit any official numerical case or relax
its bound. The CPU checks do not prove all possible GPU intermediates finite.

## Bounded CPU evidence

Run `.venv-modal/bin/python tools/test_attention_exact_rescale_cpu.py`.
The verifier imports no candidate module, Torch or CUDA. It pins parent and
candidate hashes, parses Python syntax, and reverses the three exact changes:
provenance comment, extension namespace, scalar guard. The resulting full file
must be byte-identical to87a9. This establishes unchanged surrounding macro and
template branches, including their casts, MMA/barrier sequence and dispatch.

It compiles the actual extracted parent/candidate loop excerpts into host C++
functions. All4091 finite O bit patterns (directed zeros/subnormals/extremes
plus fixed CPU random bits) across nine alpha bit patterns pass1,178,208
state-element comparisons and130,912 alpha-one copy checks. Those counts
include both sixteen affected and sixteen untouched O elements; they are not
independent GPU execution tests. Row-sum output is checked per invocation.
The host compiler uses no fast math and disables fusion equally for both
wrappers; this does not certify GPU FMA lowering. Five nonfinite O examples
are reported separately, including the signaling-NaN difference above.

Three source mutations are rejected: tolerance comparison, changed row sum,
and a full-mask vote. The host build/run are bounded30s/10s and use a temporary
directory removed on completion. No broad simulator or profiler was added.
Author output: `experiments/iterations/20261104-attention-exact-rescale-cpu.json`.

## Pre-result acceptance gate fixed by root

This supersedes the weaker suggested gate in the research/strategy notes.
Compare exact candidate receipt against public`8f9e5971f95841a0` and
private`c21d391dbf604bb5`, both87a9. Verify evaluator/image and case contracts;
record node/seed changes and confidence intervals without calling overlap a
significance result. Use official normalized timing/paired ratios, not raw
profile rows.

- Clean16 correctness tests,12 benchmarks and12 profiles.
- Official score ratio, aggregating the six weighted loads, at least1.01
  against **both** parents. Each load weights sliding/full by46/16 before
  the six-load geometric aggregation.
- Five affected cases: Q1B1full, B16sliding/full and B32sliding/full. Their
  geometric normalized-latency speed ratio is at least1.03 against both;
  at least3/5 are individually faster against both; no affected normalized
  latency ratio exceeds1.02.
- Seven controls: Q1B1sliding and all six Q8 cases. Control normalized-latency
  GM must lie in[0.98,1.02] against each parent.
- Source/profile binding must preserve the intended path: parent selected
  symbols are `attn_kernel<1,false,11>` for B1full,
  `<1,false,1>` for B16/B32sliding, and `<1,true,0>` for B16/B32full.
  `ONE = !FLAT && (MODE & 4) != 0`, so B1sliding `<1,false,13>` stays a control.
  The source keeps `OKC_LAZY1=0` and unchanged build flags. Profiles can establish
  the selected specialization; they cannot establish how often the new branch
  skipped products. Do not invent that frequency or require instrumentation.

All gates are a fixed consistency screen, not proof against timing noise.
Root submits at most one private check after independent source review. Any
failure closes this fixed attempt; no repeat inside noise, threshold retune,
branch-policy sweep, Q8 extension or decoder port. A passing useful attention
result still needs a separate unchanged-scope decoder check before promotion.

## Practical payoff limit

`NEXT_KERNEL_STRATEGY_20261104.md` recomputes the exact97af profile-share model:
5/10/20% less affected-attention time corresponds to approximately
0.393/0.788/1.587% less whole-decoder normalized latency. Even impossible deletion
of those entire paths gives only8.413% in that model, below the15.333% current
gap. Separate captures and PDL overlap mean these are scenarios, not runtime
bounds. A verified small gain is useful; this proposal cannot credibly promise
to close the leaderboard gap by itself.
