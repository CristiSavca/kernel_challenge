# V11 terminal resource outcome

The bounded run `experiments/modal/20261104T062254Z-94807c25` is terminal
`failed`; sandbox termination is confirmed. Elapsed worker time117.411934s.
Only one baseline eager oracle check passed. There are zero candidate oracle
checks and zero paired timings. The candidate warm call was enqueued and its
native dispatch recorded before the resource gate rejected it; this does not
establish completed execution or numerical correctness.

Frozen core remains
`7b57f0f8a1936a818ac6a399c047d9a1739e3c865070621a57afdd9e420cc5ed`;
wrapper remains
`6724284b1e8aef03a3af38d1d9d008c1df680b241a37b2d688464b8eeb60633f`.
No source, runner, resource policy or historical artifact was changed by this audit.

| Exact `attn_kernel<8,true,0>` | Parent | V10 | V11 |
| --- | ---: | ---: | ---: |
| Stack/local bytes per thread | 0 | 296 | 88 |
| ptxas reported spill-store bytes | 0 | 1128 | 132 |
| ptxas reported spill-load bytes | 0 | 864 | 192 |
| C7512 WGMMA serialization diagnostic | absent | present | absent |
| Initial registers/thread | 80 | 80 | 80 |
| Requested dynamic shared bytes | 114688 | 114688 | 114688 |
| Actual maximum resident CTAs/SM | 2 | 2 | 2 |

The V11 compiler capture is complete and has actual ptxas diagnostics.
There is no serialization line anywhere in that complete log. Native arguments
and the unchanged host guard establish the intended first-case plan:
Q8/full/noRoPE, flat264×1×1, cooperative, resident. Thus the failure is the
zero-local-memory criterion, not loss of two-CTA occupancy or fallback dispatch.

## What the result supports

The controlled revision kept96/112/32 role budgets, arithmetic, storage and
launch geometry while exposing QK/PV as distinct compile-time roles. Removing
C7512 and reducing the aggregate spills supports the hypothesis that control
flow and allocator visibility materially contributed to V10's resource issue.
That is a partial result: the full predicted condition, spill-free compilation,
did not occur. The experiment does not establish that ptxas assigned96 registers
to the old PV body, nor that all changed spill traffic belonged to that body.

The80-register attribute is the initial allocation for the kernel, not a
measurement of each warpgroup after `setmaxnreg`. It does not demonstrate that
the112-register PV budget was ignored. Similarly, ptxas's reported spill byte
counts are static compiler summaries, not observed dynamic memory traffic or
latency. Their reductions cannot be converted into a performance speedup.

## Remaining uncertainty

The saved artifacts include full compiler logs and host resource attributes,
but no PTX, SASS, spill source-line map or dynamic counters. They cannot tell
whether the remaining88-byte local footprint belongs to a repeated hot loop,
rare rescale path, segment prologue, common finalizer or inactive fallback.

Three source-level pressure sites remain plausible, without attribution:

- PV retains76 FP32 scalars (`o`:64, `l`:8, final offsets:4), plus descriptors,
  addressing and loop/rescale state under112 registers.
- QK's common segment normalization/staging now executes under96 registers,
  whereas the parent's unsplit consumer region had104. Its score/offset state
  and temporary address/conversion work also remain.
- The role branches still reconverge for the common finalizer after their
  accumulators die; its allocation and surrounding scheduler state are not
  identified by the aggregate log.

These possibilities do not justify another variant or a register-count sweep.
The current screen remains rejected under its declared zero-local gate. It
does not show V11 to be incorrect, slow, fast or safe to promote. Any decision
to pursue an official private correctness/performance check despite this
heuristic gate requires a separate explicit review; this audit changes no gate
and launches no test.

Evidence digests (complete candidate compile logs):

- V10: `68964240d0d414d8fc96fc223b65030d6c7201e85a4395bdd9c3d7fda9b49e0b`
- V11: `b638f68f4230694d150eef48c0aafd3a4b246e1a3d8251f2f55f2eb5ac25a0a1`
