# Whole phase-A publication experiment

Status: CPU/static checks passed; GPU untested. Root owns the decision to run
one bounded screen after the Dense V9 result. This candidate has no GS4,
attention V5, or dense-policy changes. It is not promoted.

| Artifact | SHA-256 |
| --- | --- |
| Decoder `native_wgs_early_a_publish_v10.py` | `102c0d1509d39bb44d3884ea41ea9eef9624bf6465527eeee7d5bcec9ef3eb3a` |
| Exact decoder parent | `80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c` |
| Standalone `muse_moe_w4a16/candidates/native_wgs_early_a_publish_v8.py` | `2931be209c1ef902158883b1e1aceeb5a34f5ac72f15bb525c13d58643e7a488` |
| Exact standalone parent | `ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a` |

Both are derived directly from the hash-named files under
`experiments/candidates/<problem>/`. Existing MIT/Apache-2.0 notices and source
provenance remain. The only changed top-level AST nodes are the native CUDA
string and its extension builder (`MOE_CUDA_SRC`/`_ma_ext` in the decoder,
`CUDA_SRC`/`_ext` in standalone). Extension names are isolated with prefixes
`dl_okc_moe_earlya1_v10_` and `okc_moe_earlya1_v8_`.

## Hypothesis and exact change

The incumbent delays a completed whole A unit's `doneAs` release until after
the next segment's joined weight/activation barrier and its first eight k16
MMAs. Publishing that already completed unit immediately may shorten a later
B activation dependency. After the existing NTT switch in `wgs_phase`, before
advancing `gs`, both candidates add:

```cpp
if constexpr (!PB) {
  if (pd.kind == 1) wgs_flush<false>(p, S, pd);
}
```

No macro, ring, math, weight layout, dispatch threshold, or PDL behavior changes.
Split A records and all B records retain their original deferred flush. The
source's `WGSDFIN=2` makes A split records use a last-arriver atomic/reduction;
they must not be described as fire-and-forget publications.

The primary counter-hypothesis is that immediate publication exposes xB store
drain which the original delay hides beneath later MMAs. The incumbent comment
records a historical 0.5–1% benefit from delaying its flush. No measured large
gain is claimed. Observed routes have at least 74.4–85.9% whole A units, but this
counts events; one unchanged split unit can remain critical. No recoverable
latency bound has been established.

## Execution/order argument and static validation

For each whole unit, `wgs_seg` finishes `wg_wait<0>()`, writes every xB epilogue
element, and assigns a warpgroup-uniform kind-1 record. The existing flush then
executes its same 128-thread barrier and cumulative `red.release` publication.
It clears `pd.kind`, so the next segment and phase-tail flush cannot duplicate
the increment. Every phase-B stage still waits for exactly eight publications.
No release is advanced before its producer math or stores. Scratch resets and
generations are untouched.

Python parses/compiles without imports. Reversing the one native insertion and
extension-name replacement reproduces each parent's complete executable Python
AST. Reversing the insertion alone reproduces each parent's native CUDA bytes;
`wgs_phase` is the only native function changed. The two candidates have the
same `wgs_phase` body. Active submissions still match their original hashes.
Machine-readable evidence is `native_wgs_early_a_publish_v10.static.json`.

These checks do not establish CUDA compilation, absence of C7510/C7512,
register allocation, runtime correctness, or performance. Execution timing can
perturb unrelated split FP32 atomic accumulation order despite unchanged math.

## One bounded screening run

The standalone cases are M32 → M128 → M256 → M16 → M32, with the original
`normal` distribution, `busiest` rank, and EP8. M16 is an unchanged-path control;
the last M32 revisits shared scratch after the transition. Six fresh timed
inputs cover more than a single routing. Actual route histograms and extension
resource records should accompany the result.

Root may run the following **without `--dry-run` only after deciding the next
allocation is warranted and the existing allocation has ended**:

```sh
.venv-modal/bin/python tools/modal_h200.py benchmark \
  --problem muse_moe_w4a16 \
  --candidate muse_moe_w4a16/candidates/native_wgs_early_a_publish_v8.py \
  --incumbent experiments/candidates/muse_moe_w4a16/ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a.py \
  --cases muse_moe_w4a16/candidates/native_wgs_early_a_publish_v8.cases.json \
  --seed 7202 --repeats 6 --resources --profile --timeout 900 --dry-run
```

Continue only for a repeatable complete-MoE improvement of at least 3% at M128
or M256, without greater than 1% systematic losses elsewhere, after every
numerical/scratch check passes. A useful result needs a second paired
confirmation before whole-decoder integration and an official check. A
flat/negative result ends this probe; no CI sweep. This experiment alone has no
evidence of the approximately 14.76% whole-decoder latency reduction needed for
the prior leader.

Public-source context and independent design critique:
`tools/MOE_WGS_EARLY_PUBLICATION_PROPOSAL.md`. The independent critic verified
both final source hashes, reversed the insertion and extension rename to prove
complete AST equality to the frozen parents, and returned a bounded-screen GO.
That review is static; CUDA remains untested until root's run.
