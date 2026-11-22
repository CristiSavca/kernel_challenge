# Independent V75 review

Reviewed frozen SHA-256
`fe02db5dc89b286b914766c0a4079feba0027cbdfe960817ca648d885920bdaa`,
137,577 bytes across 26 source files. Every frozen patch section matched the
reviewed stage after normalizing Git object-ID abbreviation.

The V61 projection delta adds three clean source paths to V72. The model merge
carries `hopper_decode` alongside V72's `grouped_decode` and destination tensor.
Both the direct multicast reduction return and the ordinary return pass
`hopper_decode` into unpool. The independent checks executed the exact combined
`_forward_tp`, `_reduce_and_unpool` and deferred-gather selection bodies in
384 cases covering TP4/TP8, decode/verify/prefill, direct and ordinary reduction,
expert fallback, replicated projection, full gather and deferred gather.
Every case reduced once and returned the expected full or rank-local width.
Ten additional cases verified the projection fallback with compiler tracing,
batch invariance, scattered inputs, sequence sharding or the feature disabled.

The two collective paths use separate regions of the same symmetric allocation.
V61 uses the push plane and its alternating phases. V71 uses the pull plane
and the existing multicast barriers. Projection reservation reconstructs those
regions before graph capture. V71 borrows the current pull view at capture,
so it follows the reconstructed communicator. The model consumes attention
output in the gated norm boundary and expert output in the unpool projection
before another collective can reuse the pull buffer. The router side stream
computes logits and routes without taking either collective plane. Main-stream
join and tensor lifetime handling precede expert execution.

Independent address and ownership checks covered all row counts 1 through
512, both 4,096 and 8,192 output widths, TP4 and TP8, and both push phases.
The 2,048 shapes fit the reserved eight-MiB phase. All 270,336 individual
counter assignments had exactly one owner, including unused tail counters.
A further 2,000 mixed push-reduction, gather, deferred-norm and pull-reduction
uses retained uniform push phase. The grid remains bounded by the existing
132 Hopper counters and the original gather kernel uses its incumbent PDL
producer wait. These models check source bounds and ordering under the existing
device contracts. Actual CUDA compilation and multi-GPU progress require H200.

The model excludes single-batch and two-batch overlap, deterministic mode,
DP/CP/DCP/PP, and unsupported projection geometry. Prefill keeps its existing
SP dispatch. The fused gather-plus-norm guard still requires the gathered tensor
to fit one slot. At TP8 with the one-MiB reserved slot, the 8,192-wide unpool
uses that fused boundary through 64 rows. Larger admitted shapes gather into
the ordinary full tensor and run the normal norm boundary. This conservative
guard preserves the incumbent fused-kernel capacity contract.

The push reservation grows final TP8 storage by 14 MiB relative to the tuned
128-KiB slots. V64's measured 16-user peak was stock +1.7 GB, and V75's complete
peak, graph buffers and post-capture KV sizing still need official validation.
Projection weight reads can fall while gather traffic and launches increase.
The V61 bandwidth estimates remain component bounds. Compilation, dynamic
tree replay, numerical agreement, exact requested outputs, later turns, TTFT,
memory and throughput remain the GPU gates. [V61 component](../v61_hopper_decode_projection_gather/README.md),
[V72 review](../v72_tree_multicast_decode/independent_review.md).
