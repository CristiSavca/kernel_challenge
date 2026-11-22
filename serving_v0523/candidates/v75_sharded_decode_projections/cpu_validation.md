# V75 CPU validation summary

This summary applies to the frozen V75 patch with SHA-256
`fe02db5dc89b286b914766c0a4079feba0027cbdfe960817ca648d885920bdaa`,
137,577 bytes across 26 source files. The integration owner supplied the
combined component results below. The independent reviewer separately
checked the combined model and communicator ownership. These checks run
Python source bodies and CPU models on an Apple Silicon host.

| Combined-source suite | Passed coverage |
| --- | --- |
| V71 workspace reduction | 387 selector cases, 28 helper rejection cases and 3 alias/mutation checks |
| V61 projection/prefill integration | 54 integration cases and 324 prenorm calls |
| V66 grouped decode | 600 mapped planner cases, 48 grouped-plus-finalizer cases, 56,157 BF16 route-value comparisons and 71 dispatch cases |

The older component fixtures needed V75's new projection field and keyword
parameters. Fixtures for the inherited projection path set
`_hopper_decode_gemm_ag=False`. Mock expert, reduction and gather entry
points accept the added `hopper_decode` argument so the original component
cases reach their intended paths. These fixture changes adapt interfaces.
The independent cases below also exercise the enabled projection path.
[V71 suite](../v71_direct_multicast_workspace/README.md),
[V61 component](../v61_hopper_decode_projection_gather/README.md),
[V66 suite](../v66_humming_grouped_decode/README.md)

The independent review executed the exact combined `_forward_tp`,
`_reduce_and_unpool` and deferred-gather selection bodies in 384 cases.
Coverage includes TP4 and TP8, decode, verification and prefill, direct
and ordinary reduction, expert fallback, replicated projections, full
gather and deferred gather. Every case reduced once and returned the
expected full or local column width. Ten additional cases exercised
compiler tracing, batch invariance, scattered inputs, sequence sharding
and a disabled feature switch.

Address and ownership checks covered every row count from 1 through 512,
output widths 4,096 and 8,192, TP4 and TP8, and both push phases. All 2,048
shapes fit the reserved eight-MiB phase. Each of the 270,336 individual
counter assignments had one owner, including unused tail counters.
A further 2,000 mixed push-reduction, gather, deferred-norm and
pull-reduction uses retained a uniform push phase. The review traced
push/pull region reconstruction, immediate consumption of borrowed
results, router side-stream joins and the incumbent PDL producer wait.
[Independent source and ownership review](independent_review.md)

The review matched every frozen patch section to the inspected stage after
normalizing Git object-ID abbreviation. The recorded digest, payload size,
file count and maximum added-line length were checked while preparing
this documentation. The added projection code retains the existing GEMM
dispatcher and gather arithmetic. The component checks validate source
selection, shape, route and ownership logic.

CUDA compilation, generated register/local-memory use, actual barriers
and multi-GPU progress, graph replay with changing requests and routes,
output agreement, later turns, exact output counts, TTFT, throughput and
complete peak GPU memory remain unverified.
