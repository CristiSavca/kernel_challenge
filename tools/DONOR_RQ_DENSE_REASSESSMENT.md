# Donor RQ dense lead, 2026-11-04

Offline decision: a distinct compressed RS-WGMMA GEMM exists, but do not transplant
the donor implementation or implement it now. No source change, download, code
execution, GPU job or submission accompanied this audit. The research direction
has shifted to recent primary literature.

The preserved keell0renz decoder at commit
`890809cee421500b00126d084e81b338ef9181ab`, SHA256 `82c098ae...`, contains chen's
RQ RS128/RS256 path (lines 14334–14340, 15196–15894): lossless 12-bit BF16 weights
decode directly to WGMMA register-A, activations use shared-memory B, and a joined
weight/activation TMA ring feeds all 128/256 tokens in each output-row tile. The
codec is already familiar; the large-token register-A GEMM is the distinct part.
Frozen80a4's `rq_deep_tma_pooler` (15002–15018) instead streams BF16 weights through
Triton tensor descriptors. The DenseV9 author confirmed its selector did not test
this donated CUDA kernel; V9's null outcome does not reject this mechanism.

Direct adoption changes accumulation behavior. Donor lines 15688–15696 and
15753–15771 round FP32 split-K partials to BF16 and TMA-reduce-add them through a
BF16 output map (15814). The incumbent's cuBLASLt wrapper explicitly selects FP32
reduction (15096–15101). The BF16 split reduction cannot be presented as an
unchanged-precision port. A clean version needs FP32 partials and a final FP32
reduction, or a full-K CTA that writes BF16 only once. Either changes the donor's
costs and requires a separate numerical/resource analysis.

The final donor dispatch permits only qkv128, qkv256 and unpool128 (16209).
Official profile4b216098993e4112 contains roughly 18.7/18.2 us qkv-plus-reduction
and 20.9/21.2 us unpool at T128, and 22.9/22.8 us qkv at T256. Kernel-to-projection
mapping is inferred from pinned QKV dispatch, available RQ paths and the other
projection sizes; it is not a new annotated profile. Weighted whole-layer medians
are 279.871 us and 369.631 us. A hypothetical 25% cut of those eligible kernels
would yield only about 0.85% aggregate decoder latency reduction. This is generous
accounting, not a prediction.

Raw qkv weights are 36 MiB and unpool weights 64 MiB. Twelve-bit payloads save
9/16 MiB before escape/header traffic, equivalent to 1.97/3.50 us at an assumed
4.8 TB/s. Added decode and reduction costs cannot be ignored. FP32 partial
workspaces add cache traffic and potentially a launch; that traffic must not
automatically be called DRAM traffic because these buffers can fit in L2.

A precision-preserving narrower possibility is unpool128 with 64 output rows
per CTA and KS=1: 8192/64=128 CTAs, each completes its full K before one BF16
store. Donor `h128s8e`-style half-width variants contain this structure, but they
are dormant in its final candidate list. They also double each CTA's K traversal
relative to full-width KS2, increase activation requests, and change consumer
parallelism. No saved measurement justifies promoting that dormant variant.

Small-token donor GLE largely repeats the incumbent's existing UV deferred
normalization/fused pooler-router structure. Its FP32-router handling is a math
and route-fidelity difference, not evidence of a missing large dense speedup.
Retain this note as a bounded lead; do not repeat the dense selector or infer
new gains from the public repository's unverified score.
