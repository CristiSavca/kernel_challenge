# X announcement and public-release scout — November 4, 2026

**Result:** FlashInfer has new, released source worth learning from, but none of
the inspected implementations is a direct Muse/H200 replacement. Rohan's public
repository has not changed since our earlier audit. The strongest immediate
action remains validating the serving finalizer that just passed our fixed
component screen; these announcements do not justify replacing that evidence
with a new architecture or installing another research harness.

Direct access to [Rohan's X profile](https://x.com/RohanArun) and
[FlashInfer's X profile](https://x.com/FlashInfer) was unavailable; attempted
original-post/profile fetches returned HTTP 403. Search surfaced Rohan's November 3
GA/predictor/Lean announcement through a mirror. I used that only as a discovery
lead, not as technical or release evidence. The conclusions below rely on the
authors' public GitHub source, release records and technical posts. This is not
an exhaustive inventory of X posts.

## Released findings and applicability

| Primary source | Verified mechanism and measurement scope | Decision for Muse |
| --- | --- | --- |
| [FlashInfer v0.7.3](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.3), released November 4 | Signed tag resolves to `775207deb34055667df89ed618fb4b06fc84de74`. It includes the finalize/all-reduce work below and packaging support for the Hopper BF16 MegaMoE sources. | A real new release, not evidence that its whole stack improves our pinned serving checkout. Inspect individual mechanisms. |
| [Hopper BF16 MegaMoE #5958](https://github.com/flashinfer-ai/flashinfer/pull/5958), merged October 2 | Native BF16 dispatch, grouped WGMMA FC1/FC2, BF16 intermediate and combine wire. Fuses small-token dispatch and the receiving wait/reduce/ack tail. Published paired H100 EP2/4/8 measurements include graph mode, baseline-autotuning anomalies and weaker tiny-token gains. Generated GEMM sources and tests are released; its development generator is outside the repository. | The communication/lifetime design is useful reference. BF16 weights and conventional expert combine differ from Muse's NVFP4 weights and per-expert RMSNorm before route weighting. No wholesale backend port. The benchmark is not Muse or our native incumbent. |
| [Pre-reduced BF16 wire #6058](https://github.com/flashinfer-ai/flashinfer/pull/6058), merged October 5 | Groups routes by owning rank, accumulates in FP32 in route order, rounds to a BF16 wire row, then reduces rank rows. Also removes repeated clears and folds worklist creation into compaction. Authors report a 1.031 geometric mean over 35 paired rows against round 1; the table's T ≤ 128 ratios are approximately 0.998–1.011. Default capacity ≤ 8 retains the old per-route format. | Do not import the BF16 wire rounding or rank-group reassociation into our FP32 route-sum contract. Protocol reuse requires a separate lifetime proof. Its gains concentrate at larger token counts and do not predict our decode gain. |
| [Finalize/all-reduce #6166](https://github.com/flashinfer-ai/flashinfer/pull/6166), merged October 8 | Actual diff batches four indexed route loads as raw 128-bit words before their ordered scale operations. For TP4 it moves an independent residual load ahead of Lamport polling; TP8 retains its original residual placement and TP2 is unchanged. Authors report small-token improvements on B200/B300, with explicit regressions for rejected variants and increased TP8 registers. | Keep the load-scheduling idea, not this SM100/103 kernel. It has RMSNorm after reduction, unlike Muse's per-expert normalization. Its scaling/rounding and communication contract are also different. A future local adaptation must retain our exact per-route operations and masks. |
| [FlashInfer v0.7 technical post](https://flashinfer.ai/2026/09/22/flashinfer-v07.html), September 22 | Released Prims-TS implementations, explicit experimental backends, and autotuning policies that distinguish eager execution from graph replay. The published MegaMoE NVFP4 comparison uses Blackwell and a cast checkpoint. | Use matching execution conditions and retain the untuned baseline in comparisons. The precision-changing/model-changing comparison is not a permissible speedup forecast here. No need to install another tuner. |

The [released pre-reduced publisher source](https://github.com/flashinfer-ai/flashinfer/blob/775207deb34055667df89ed618fb4b06fc84de74/flashinfer/moe_ep/kernel_src/sm90/cake_bf16_megamoe/src/cake_combine_prereduced_bf16.cu)
explicitly documents the BF16 rounding boundary; its later load-hoisted form
retains an ascending-route FMA chain. The
[released receiving tail](https://github.com/flashinfer-ai/flashinfer/blob/775207deb34055667df89ed618fb4b06fc84de74/flashinfer/moe_ep/kernel_src/sm90/cake_bf16_megamoe/src/cake_combine_tail_bf16.cu)
combines waiting, summation and acknowledgement without applying Muse's
per-expert RMSNorm. Those concrete differences prevent an API-name-based port.
Both inspected files carry Apache-2.0 notices; retain those and applicable
upstream notices if adapting code. The repository's
[license](https://github.com/flashinfer-ai/flashinfer/blob/775207deb34055667df89ed618fb4b06fc84de74/LICENSE)
was checked; no source was imported or executed.

## Rohan: announcement is ahead of the available release

Fresh public API reads confirm main remains
`018b7a102348ea3b73701e8be52d24f82d8c91ee`, all 13 branch heads match our09:12 UTC
audit, and the latest release remains
[v1.5.0, October 30](https://github.com/rohanarun/Auto-research-agent/releases/tag/v1.5.0).
Therefore there is no newly published GPU genetic optimizer, learned hardware
predictor or Lean resource-pruning implementation in that repository to reuse
since the source review in `ROHAN_REPOSITORY_RELEASE_AUDIT_20261104.md`.
This does not dispute private results or establish that no other repository
exists. The public workflow remains useful source-bound experiment coordination;
its benchmark results do not supersede our official measurements.

## Concrete next-action implication

V22's exact-source official component check improved M1/M8/M16 over in-run stock
Marlin by 48.47%/27.15%/21.93%, with unaffected controls near unity. That is
stronger project-specific evidence than the announcements above. Complete its
generic-shape, graph and concurrent-stream validation before claiming full
serving performance; do not promote it as a replacement for native 247888.

If a subsequent measured bottleneck remains in the new combine kernel, #6166
offers one specific scheduling question: can several route loads be issued
before their ordered normalization/weight/sum operations without extra rounding,
unsafe invalid-route loads or harmful register growth? Our Triton compiler may
already schedule such loads; source resemblance alone does not establish missing
overlap. No new candidate or GPU experiment is authorized by this scout.

## Bounded source evidence

No clone, installation, credentials, provider call, GPU allocation or message to
an external author was used. Bounded raw/API reads totaled 719,312 bytes including
aborted prefixes; raw downloads were not retained. The #6166 diff was inspected
as an 80,001-byte prefix, sufficient to verify the first quad-load and residual
hoist hunks, not a complete kernel correctness review. Release/author pages were
also read through web search. Only this concise note is saved by the scout.

| Evidence | SHA-256 |
| --- | --- |
| Rohan main-ref response | `0ee3a398e10f3bef37af81078f542c76e4342025d58508c3a90a68b507389f1a` |
| Rohan 13-branch response | `8b67a052826f18c1163139e7f85da7b7fa4e4e49e2eff0e69b97f40f95ab0f52` |
| FlashInfer annotated v0.7.3 tag object | `e8b87786da13804dffbcaa5afc55fd8cbc1d3e3847dd025a77dc87cd4d29c84f` |
| v0.7.3 pre-reduced publisher source | `f23b6c163cecb228293014928c556e540e361b660c17abb04752e88fcf044642` |
| v0.7.3 receiving-tail source | `f352622fe66389562b66b83d926b5837e31d75745738fc8ba166fdbf4111393a` |
| v0.7.3 LICENSE | `cb67c224f503e0a063908950b12f89a7280c6e527dcffac972aa114e4bf3c5de` |
| #6166 inspected diff prefix | `79cf51a20cb86f1ffbd965ba3f735475c870c7726f1492f31174f15c19b3ed3a` |

PR merge commits: #5958 `46340689a5ab5665ab39f6281dac4d021d7ae94b`,
#6058 `f7a4874ccf0926ac9ba32e259b0d5c940bdc1c4a`,
#6166 `fb2b35145ef84ba25f56b58b3579ece14c242c0c`.
