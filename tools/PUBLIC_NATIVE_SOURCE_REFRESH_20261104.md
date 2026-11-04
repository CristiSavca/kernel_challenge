# Public native source refresh — November 4, 2026

Checked 09:36–09:38 UTC through bounded live GitHub API reads. **No new native
kernel implementation was found in the pinned donors or their directly
attributed projection/decoder upstreams.** This pass does not justify another
variant or GPU check. No repository was cloned, source imported or executed,
GPU allocated, official job polled, or participant contacted.

## Exact revisions and branches

| Repository | Current revision and last commit | Finding |
| --- | --- | --- |
| [vjeux/okc-2026](https://github.com/vjeux/okc-2026/commit/f35a41ee1c13e316dabbf772830d47b23b453393) | `f35a41ee1c13e316dabbf772830d47b23b453393`, October 28 02:03 UTC | Matches our pin. Sole public branch is main. No newer decoder, MoE, or BF16 source. Apache-2.0. |
| [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge/commit/890809cee421500b00126d084e81b338ef9181ab) | `890809cee421500b00126d084e81b338ef9181ab`, November 2 19:51 UTC | Matches preserved donor. Sole branch is main. Latest commit changes README; preceding source release is November 1. Apache-2.0 with preserved NOTICE. |
| [rohanarun/Auto-research-agent](https://github.com/rohanarun/Auto-research-agent/commit/018b7a102348ea3b73701e8be52d24f82d8c91ee) | `018b7a102348ea3b73701e8be52d24f82d8c91ee`, October 30 15:20 UTC | Matches pin. All thirteen branch names and hashes exactly match the 09:12 audit. No newly released kernel or GPU search tooling. Workflow MIT; upstream kernel notices still apply. |
| [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels/commit/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0) | `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`, October 29 21:11 UTC | Sole main branch matches the revision attributed in our keell0renz donor. No later source revision. Apache-2.0. |
| [chenwainuo/okc](https://github.com/chenwainuo/okc/commit/3d418e1625f6de2075b5684a27ddbc3d1442422d) | `3d418e1625f6de2075b5684a27ddbc3d1442422d`, October 29 03:22 UTC | Sole main branch matches the donor's BF16 rs128 source. This is the same mechanism already reassessed in `DONOR_RQ_DENSE_REASSESSMENT.md`, not a fresh implementation. Apache-2.0. |
| [NathenFernandes/muse-bf16-proj](https://github.com/NathenFernandes/muse-bf16-proj/commit/86aff75e6aca72511bf9b41b6dc190bf1e763373) | `86aff75e6aca72511bf9b41b6dc190bf1e763373`, October 24 20:30 UTC | Sole solution branch. Three commits after the donor's `7385e5d`, but the retained submission is unchanged. MIT. |

The [Nathen comparison](https://github.com/NathenFernandes/muse-bf16-proj/compare/7385e5d8f70980cfd5322ea254edc0b2b8ca90f7...86aff75e6aca72511bf9b41b6dc190bf1e763373)
has six changed files: README, ranking/source-hash records, and deletion of an
older checkpoint and two older result files. `submission.py` is absent from
the complete change list. A newer repository HEAD here does not mean a new
projection kernel.

## Availability limit and mechanism decision

The donor also credits [ZhimaoL/sglang](https://github.com/ZhimaoL/sglang)
branches `okc/x-vfgpdl`, `okc/cand12pre`, and `okc/x-n72-naw`. All three direct
GitHub branch endpoints returned 404. Prefix matching for `heads/okc` returned
an empty list, both with and without a trailing slash. The repository itself
remains public and Apache-2.0. Its broad branch listing has eighteen pages of
inherited SGLang branches; this pass deliberately did not traverse them all.
Therefore these cited refs are unavailable through the checked endpoints;
this is not a claim that every possible branch or historical commit is absent.
Relevant donor extracts remain preserved in our licensed source cache.

The already reviewed donor mechanisms remain KPERM (already measured and
promoted), MFIN per-expert completion, lossless SC6 scale packing, and rs128
dense projections. Their prior scope/correctness/ROI findings remain in
`PUBLIC_MECHANISM_PROPOSALS.md`, `PUBLIC_KERNEL_SCOUT_20261104.md`, and
`DONOR_RQ_DENSE_REASSESSMENT.md`. An unchanged release supplies no new evidence
that revisiting them will close the remaining whole-decoder gap. Do not repeat
an experiment merely because the source was rediscovered.

README scores, local timings, historical ranks, and labels such as “fair” are
author claims until tied to independently verified official receipts. This
refresh verified no new official score and established no superiority over
our active decoder. Root owns the separate live leaderboard refresh and the
pending wait2 experiment.

## Evidence and reuse

`experiments/iterations/20261104-public-native-refresh.json` retains timestamped
API URLs, response hashes, sizes, pagination/availability evidence, branch and
commit records, the Nathen comparison, and explicit conclusions. The existing
vjeux, keell0renz and Rohan caches were reused; no new kernel source was saved.
All applicable existing LICENSE/NOTICE files remain intact. Total new evidence
is approximately 291 KB of metadata plus this note; no bulk checkout or build
was created. The concrete decision is **NO NEW SOURCE LEAD**, with the exact
refs above available for a later meaningful-change check.
