# Distinct architecture screen after shared-A rejection

November4,2026. **No implementation GO on present evidence.** Exact standalone
parent247888 and decoder97af remain the references. This closes a bounded
read-only review, not a claim that these mechanisms can never help. No candidate,
GPU allocation or official call was made.

The one new architecture assessed was **CL2 paired-row-unit activation
multicast**: retain incumbent RS-WGMMA, KPERM, output256/K256 tiles, GS2/wait1 and
three raw/X slots; two clustered CTAs process neighboring output units of the
same job/K sequence. One producer multicasts the shared X tile to both CTAs,
replacing two independent global-to-shared X transfers. All weights, routed
experts, tensor work and intermediate precision remain unchanged.

This differs from the rejected shared-A path: it removes repeated activation
source requests rather than adding decoded-weight shared traffic. NVIDIA's
[bulk-copy API](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk.html)
supports this multicast form on sm90a. The
[PTX specification](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
requires common CTA-relative destination/barrier offsets across recipients.
API support does not establish a performance case.

## Amount of work it can remove

Per complete job the parent requests27MiB packed weights and
`768 × padded_job_rows × 512` activation bytes. CL2 would halve the latter and
reduce total bulk-copy issue count by25%; it would not halve total traffic or
the bytes written to the recipient shared memories. Nine retained ef7 diagnostic
routes give:

| M | Weight requests | X source requests | Removed share of W+X requests |
| --- | --- | --- | --- |
|32|432–486MiB|48–66MiB|5.0–5.98%|
|128|648–702MiB|108–144MiB|6.67–9.09%|
|256|702–783MiB|171–243MiB|9.5–12.86%|

These are old standalone routes, not observed decoder routes. The saved ef7
NCU samples measured DRAM reads only1.0088×/1.0069× nominal job-weight bytes at
M32/M256. Thus a claim of large activation HBM savings is unsupported. No
retained L2/TMA service measurement identifies the removed requests as a
critical-path bottleneck.

The paired JSON applies two deliberately unrealistic service models to exact
97af profiler shares and normalized load weights: all MoE time proportional to
W+X requested bytes, or all MoE time proportional to bulk-copy instruction count.
They are scope scenarios, not attainable gains or upper bounds; neither accounts
for PDL overlap, multicast delivery or synchronization costs.

## Fixed design that would need evidence before implementation

Use two-CTA clusters only, retaining576 threads and existing shared storage per
CTA. Assign cost ranges over pairs: phaseA16 pair-units/job and phaseB8, mapping
rank r to original unit `2u+r`. Both CTAs must share stage sequence, ring slot and
phase parity. Retain each rank's original accumulators and separate split-unit
fixups; remap boundary ownership without merging or reordering FP32 partials.
The pairing is convenient for phaseA: two original256-row gate/up units produce
the256 intermediate columns consumed by one phaseB stage.

Both local X slots must be free and both recipient barriers must expect their
transaction bytes before the leader issues a multicast. Local readiness alone
is insufficient. A producer-only DSM rendezvous would need an actual release/
acquire and generation proof; remote mbarrier polling cannot simply replace
local waits. Cluster initialization/exit and all-grid resident admission must
also preserve the inherited spinning grid barriers. Coarser cost ranges,
GPC placement and waiting for the slower partner are new costs, not free reuse.

This is too much unvalidated coordination for the presently supported traffic
scope. Reopening needs exact-parent L2/TMA service evidence and a bounded
readiness/retirement design, followed by actual cluster admission. If those
justify one experiment, falsify it with unchanged correctness, retained traffic
counts and paired timing across MAIN cases plus small controls. No cluster-size,
tile or schedule sweep is proposed now.

## Root's cost-table lead

Current phase tables remain A=`112,135,140,158,173,185,200,207` and
B=`100,104,115,123,132,144,156,174`. KPERM removes repeated scale construction,
so a different cost curve is possible. However saved per-WG completion windows
are from ef7 and include cross-CTA dependencies/fixups; they do not isolate
post-KPERM per-NTT service. A constant instruction saving is not a measured
constant latency saving. Uniform-NTT cases also gain no relative assignment
change from rescaling that phase's single cost. No defensible new fixed table
can be inferred from these captures, so no replacement values are proposed.

Evidence and exact request arithmetic are saved in
`experiments/iterations/20261104-next-distinct-candidate.json`. This review stops
here; a smaller independently supported lossless representation change can be
a better next experiment than another broad rewrite.
