# Whole phase-A publication experiment

Status: **rejected for promotion and decoder integration**. The standalone
screen compiled and passed 80/80 checks, but failed the 3% continuation bar.
The decoder counterpart remains GPU-untested. No repeat or official upload is
planned. This candidate has no GS4, attention V5, or dense-policy changes.

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

## Completed screen and independent decision

Result: `experiments/modal/20261104T052256Z-4c806134/result.json`, SHA-256
`8b9568e6ec37bf39daea37430a323e5eebf4c235ca2d2b0151acf9fb89e633a0`.
The run finished in 415.62 seconds, passed all 80 eager/graph checks, and
terminated successfully. Preparation including weight cloning took
208.97/199.14 seconds for baseline/candidate. The 128 MiB read-sweep selfcheck
passed; actual DRAM traffic was not measured.

Six fresh pairs per case gave these median speedups (baseline time divided
by candidate time):

| Case | Paired ratio | Candidate faster pairs | Separate profile, baseline → candidate |
| --- | ---: | ---: | ---: |
| M32 | 1.001020× | 3/6 | 116.631 → 113.839 µs |
| M128 | 0.986994× | 2/6 | 187.094 → 188.736 µs |
| M256 | 0.993350× | 2/6 | 225.393 → 227.494 µs |
| M16, unchanged path | 1.002000× | 4/6 | 106.672 → 107.049 µs |
| M32 after transitions | 0.998470× | 2/6 | Baseline unavailable; candidate 109.109 µs |

Both target sizes lost four of six pairs and failed the 3% bar. Every first
pair improved, including the unchanged M16 path by 7.06%; isolated early-pair
gains are not convincing evidence. No samples were discarded after the fact
and no particular cause is asserted. Separate single profiles use another
fresh input, do not enter timing medians, and do not measure dependency waits.
The return-M32 baseline profiler captured no CUDA events, so no paired profile
comparison is available there; its eager/graph correctness checks still passed.

Actual routing was recorded for every replay, including seed 7302's 16 jobs
and 39 local routes. Define/resource tuples match across implementations:
main REG=96, STACK=0, LOCAL=0; other variants retain 144/148/149 registers and
zero stack/local bytes. Static resource equality does not prove the absence
of outlining or WGMMA serialization and does not explain the timings.

Independent critic `leaderboard_critic` rejects immediate whole-A kind-1
publication under these tested routes. This does not reject all readiness
scheduling or establish anything about advancing split-A records. Preserve
the incumbents; do not run a second confirmation, CI sweep, or official check.
The next separate proposal is a minimal hardware-counter diagnostic of the
exact incumbent before considering a shared-memory dequantization rewrite.
Decision: `experiments/iterations/20261104-moe-early-a.json`.

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

## Historical bounded screening plan

The standalone cases are M32 → M128 → M256 → M16 → M32, with the original
`normal` distribution, `busiest` rank, and EP8. M16 is an unchanged-path control;
the last M32 revisits shared scratch after the transition. Six fresh timed
inputs cover more than a single routing. Actual route histograms and extension
resource records should accompany the result.

The following command documents the completed screen. Do not rerun the
rejected variant:

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
That review was static. The completed standalone run above supplies local
CUDA evidence; it does not validate the unrun decoder counterpart.
