# Bounded whole-decoder boundary review

2026-11-04,14:28–14:33 UTC. Read-only review of exact decoder
`97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`
and public profile `7a8c4941741645ee`. **No new implementation-ready lead.**
This is a prioritization result, not a proof that boundary fusion cannot help.
No candidate, GPU call, official query or source change was made.

Reviewed the existing structural/attention gate notes,
`NEXT_COORDINATION_FEASIBILITY_20261104.md`, closed fusion experiments and
current source before considering two possible copy removals. This review does
not reopen V8 finalization, SK2 escape/directory/R16, or geometry sweeps. The old
notes'5% aggregate requirement is not used to reject a small useful change here.

## Actual selected boundaries

`kernel_fn`/`_after_attention` at14871/14942 select fused agent attention in the
retained12 profiles. Attention already performs QK norm/RoPE/cache write and
output norm/gating; its output is BF16 `[T,1024]`. Small O projections use SK2,
while T128/256 use the selected cuBLASLt kernels. `_uv_on` at15304 enables
T<=8 O-epilogue U/V plus boundary sums, followed by fused pooler/router and UB2;
these four profiles have no separate boundary1/2 kernel. T16's selected SK2B
unpooler already incorporates boundary2. T32 has separate pooler/router and
boundary1/2. Large cases use `_route` after combined pooler/router projections.
All selected expert profiles are the agent MoE, not Marlin.

`_route`/`_route_split` at14026/14072 perform the pre-expert RMS once, multiply
the prepared FP32 effective norm weight and round to BF16 `xm`. WGS gather near
7980 only copies/reorders those rounded bytes into expert-job `xA`; it does not
repeat that RMS. MoE post-expert normalization and routed accumulation finish
before the BF16 `m` store. The unpooler consumes that BF16 sum. Thus moving the
unpooler matrix inside individual experts would cross a required nonlinear
normalization and a BF16 round-after-sum boundary; ordinary linear distributivity
does not preserve the computation.

## Screen1: attention output directly consumed by O

The uniquely materialized output is2KiB per token,2–512KiB across scored T.
Removing its write and one equivalent complete read saves4KiB/token in a
unique-payload accounting model; actual O requests may reread it from cache.
Current output owns complete64-dimension head rows and performs their reductions
once. O owns small output-row tiles across8192 rows, each depending on the full
1024-dimension input. A naive O-side finalization repeats16 head reductions,
gate sigmoid work and workspace merging across output CTAs. Avoiding that
duplication needs a new cross-CTA publication/retirement protocol and a place
to retain the same rounded BF16 vector, or a major owner/resource redesign.
PDL already allows eligible independent prologue overlap. There is no exact
mapping that removes the write while preserving the once-only producer work
and admitted resources in hand. **No candidate proposed.**

## Screen2: route writes WGS xA directly

For selected MAIN T32/128/256, write normalized BF16 values directly to the
existing KPERM WGS activation layout and omit `xm` plus the gather's source
read. Let R be the actual number of local routed token/expert rows. The work
removed would be `8192*T` bytes of `xm` stores plus `8192*R` source-read bytes;
the `8192*R` xA destination writes and all norms/casts remain. These are requested
payload bytes, not measured HBM traffic. Actual decoder R is not present in
the retained text profiles, so no expected-R or speed forecast is supplied.

The current route CTA owns one token and top8 selection. Its destination job,
expert-local row rank, NTT padding and `S.jrow` base are not yet available:
`moe_fused` constructs them from all tokens. Assigning arrival-order slots in
route would additionally change the parent's token order/job partition unless
followed by sorting. Stable ranks need global counts/prefix publication before
the scatter. Holding4096 normalized values per token across that coordination
adds lifetime/register pressure; spilling them recreates an intermediate.
Alternatively rereading pooled values repeats the norm and/or pooled traffic.
The existing xready release protocol would need to be produced externally,
with generation, reset, chunking and PDL publication proofs.

This is the already identified centralized route/job metadata dependency with
an added direct-scatter payload, not a ready independent shortcut. The route
kernel's3.5–5.2us retained profile rows include required top8/norm work and PDL
waits; deleting those rows is not a removable-cost bound. No evidence shows the
saved cached copy outweighing added coordination or lost early weight progress.
**No implementation proposed on current evidence.**

Across the remaining pooler/router and MoE/unpool boundaries, the BF16 tensors
are consumed by many output CTAs and the norms need full rows. A last-CTA
finalizer can remove a launch while retaining the same intermediate reads;
that alone does not meet this review's work-removal criterion. Extending UV or
reenabling SK2B1 would revisit existing geometries/fusions, not identify a new
boundary mechanism. Stop the bounded review here.
