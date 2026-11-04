# Public source scout for listed decoder owners

November 4, 2026, 09:43–09:48 UTC. **No compatible new kernel lead found in the
scanned scope.** This is a bounded public-source finding, not a claim that these
competitors have no source elsewhere or that GitHub handles establish their
leaderboard identities. No private access, clone, source execution, GPU job,
official-job lookup, or participant communication occurred.

## Account coverage

Public GitHub repository metadata was requested in most-recently-updated order,
with a maximum of100 repositories per account. Names/descriptions of the returned
repositories were inspected, followed by relevant pinned trees and README files.

| Requested account | Public repositories examined | Scope |
| --- | ---: | --- |
| AlexCuadron | 14 | Complete returned listing; no recent matching kernel repository. |
| dolanzhao | 15 | Complete listing; relevant-looking profiling/framework projects predate the challenge. |
| caoshiyi | 28 | Complete listing; inspected EAT and eat-kernels. |
| gogowhy | — | GitHub endpoint returned404. Account unresolved; this does not establish private content. |
| lxe | 100 | Stopped at requested cap; two further pages exist. Inspected three recent inference engines. |
| tjiang-fair | 0 | Endpoint returned an empty public repository list. |
| Charbel199 | 23 | Complete listing; inspected gpu-anatomy. Other kernel examples lack an explicit license or predate this task. |
| mpeng19 | 11 | Complete listing; inspected auto-inference. Framework forks predate the challenge. |

Total:191 repository records across seven resolved accounts. No named OKC or
Muse kernel release appeared in these metadata listings. That name-based result
was followed by actual source inspection of the most relevant licensed hits.

## Relevant released code and why it is not a new candidate

**[caoshiyi/eat-kernels at5b9e5d3](https://github.com/caoshiyi/eat-kernels/tree/5b9e5d3f4d21fb7b814f927c34d522bddc912b45)**
is Apache-2.0. Its complete213-entry tree publishes Kimi Delta Attention for
B200 and MI350X, including serving interfaces and author benchmark claims.
That is different attention math and hardware from our H200 Muse decoder.
The companion [EAT at0a90500](https://github.com/caoshiyi/EAT/tree/0a90500b5f87c26d2fb11013b95591dcf81ed0f8)
has43 tree entries and releases the agent-team workflow, not another matching
kernel. No superior Muse result was established by either repository.

**[lxe/dsx at60e9870](https://github.com/lxe/dsx/tree/60e9870b74509825f2f8de088306d82260f0fd87)**
is MIT and contains real CUDA kernels; the complete tree has675 entries.
Actual `nvidia/src/kernels/expert_fast.cuh` describes custom GGUF encodings and
INT8 tensor-core operands. `quant_act` at line692 rounds current activations
to INT8 with per256-element scales; this cannot replace our required BF16
activations/NVFP4 experts while preserving model math. Its optional one-layer-
late expert deferral also explicitly changes computation. The separate
`gemv_bf16` in `dsx.cu` at line1182 uses eight-value vector loads, one warp per
row, and FP32 dot products. It is concrete source, but supplies no demonstrated
advance over our existing lossless compressed tensor-core projections or a
comparable Muse measurement. The sparse-attention kernels also target a
different cache/attention definition.

**[lxe/pocket-engine at7cbdb50](https://github.com/lxe/pocket-engine/tree/7cbdb50bc6834ac9b5bd63b9f123f8f64bc894b9)**
is MIT, with a complete134-entry tree. `src/linear.rs` identifies BF16 weights
as the cuBLAS path and dispatches `Kind::Bf16` to `gpu.gemm_ld` at line654.
Its actual `qwen35.cu` contains Gated Delta Net recurrence and Hopper prompt
attention beginning at line1622; its Qwen heads are256 wide. The source is
useful inference engineering, but it does not release a new matched Muse MoE
kernel, lossless BF16 decoder mechanism, or competitive Muse receipt. Existing
fusion concepts such as norm/RoPE/cache-write and gated output already occur
in our attention parent. No broad rewrite is justified by the model-level
throughput claims alone.

**[lxe/volta at1e738e2](https://github.com/lxe/volta/tree/1e738e2f61bd9ca9f3fcb160f3189ac42a2477b2)**
is MIT. Its README targets V100 with custom2/3-bit expert encodings and an FP4
cache, so its reported throughput does not preserve our precision/workload.
The recursive-tree response exceeded this scout's250KB cap; it was not saved
or retried with a larger allowance. This is a README-based incompatibility
finding, not an exhaustive source review.

**[Charbel199/gpu-anatomy at2995821](https://github.com/Charbel199/gpu-anatomy/tree/2995821fae99b5e595c9a4638d8e5f07e81397ec)**
is MIT. The complete49-entry tree contains hardware-learning experiments:
memory access, bank conflicts, register pressure, warp behavior, WMMA and
asynchronous copies. It supplies no competitive Muse implementation or score.

**[mpeng19/auto-inference atf28ff93](https://github.com/mpeng19/auto-inference/tree/f28ff938446fe214e7e44452a6008314e9db0589)**
describes a serving-price simulator and agent harness. The complete215-entry
tree contains no matching Muse/OKC kernel release. GitHub reports no license,
and no LICENSE file appears in the tree. No implementation was copied or
reused; the README read is retained only as URL/hash evidence and findings.

## Decision and evidence

Do not create another candidate from these results. The only newly inspected
expert fast path requires activation quantization and a different weight
format; the intact BF16 paths provide no new demonstrated advantage. All
performance statements above remain author claims for their own models and
protocols; none was verified as an official OKC score or improvement over our
incumbent. Root owns the separately pending experiment and live board review.

`experiments/iterations/20261104-top-owner-public-source-scout.json` preserves
account counts, pagination limits, pinned heads, complete relevant tree
responses where within budget, read URLs/hashes, and the scoped conclusions.
Eight selected MIT source/license files are cached under
`third_party/public_scout/lxe-dsx-60e9870b7450/` and
`third_party/public_scout/lxe-pocket-engine-7cbdb50bc683/`, totaling362,773 bytes.
Both LICENSE files are preserved. Metadata evidence is approximately722KB;
no models, generated builds, images, or bulk checkouts were downloaded.
