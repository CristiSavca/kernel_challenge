# V68: integrated grouped decode and verification

V68 combines V65's grouped sequence-parallel prefill and paged verification
attention, V66's device-planned grouped decode, and V67's DFLASH verification
capacity repair. Its 16-user private check passed at 17,512.9 tokens/s/GPU, with 1.7 GB/GPU above stock and zero short outputs. V64’s matched check reached 17,349.9 tokens/s/GPU. Median decode speed fell from 48.8 to 48.0 tokens/s/user. Each check had one repetition. V64 remains the scored leader at 1.4119×. [V68 result](../../runs/04b01f958aa5487c.json) · [V64 check](../../runs/ac89fddc4ee147ab.json)

- Base: SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Patch SHA256: `6248a58842e1a22c1cda5a957d5ab464a6ec824efcdf8308bdcc9c6a1046e505`
- Payload: 85,080 bytes, fifteen permitted source files, maximum added line 99 characters
- Flags: `moe-runner-backend=humming`, `speculative-moe-runner-backend=marlin`,
  `enable-layernorm-sp`, `swa-full-tokens-ratio=0.25`
- Best measured four-load serving score at the time of the V68 check was V64’s 1.4119×.

## Changes and hypothesis

V65 removes replicated prefill projections, groups live prefill expert rows,
and partitions all visible causal or sliding-window keys during compatible
target verification. V66 extends grouped Humming to decode without a CPU read
of routing data. Its planner maps, counts and sorts routes into a live suffix
of a static buffer; this matches the pinned Humming last-expert boundary.
The original routing order is restored for expert normalization and summation.

The normal EP expert sequence falls from about eleven GPU launches to six.
Grouped GEMMs can use Hopper TMA for activations and the original packed
NVFP4 weights. Stream-K is disabled, accumulation stays FP32, both global-scale
products round to BF16 at the original boundaries, and SiLU uses the stock
helper and compile flags. Grouped and indexed Humming use the same tile
geometry and logical weight volume. The hypothesis is faster data delivery
and lower launch overhead; no reduction in expert math is claimed.

V67 provides enough request-map and target verification page-table capacity
for two in-flight DFLASH blocks plus page rounding near the context boundary.
It leaves the block length, accepted model context, KV pool allocation, and
ordinary decode/EAGLE dimensions unchanged. This is a correctness repair with
no claimed throughput gain. It changes key-axis capacity while V66 derives
route counts from query rows. V62 reads the actual live lengths inside the
wider page tables.

## Composition and checks

The V66-minus-V64 delta touches six files. The V67-minus-V65 capacity delta
touches three other files. The final patch equals V67 plus that V66 delta;
every file has exact intended-parent parity. Forward/reverse applicability,
whitespace, paths and Python syntax pass. The official `okc serve diff`
export from a reused isolated sparse checkout produces identical bytes.

Root executed the following against the combined source:

- 600 route-planning cases, 48 combined expert/finalizer cases, 56,157
  BF16-rounded values, and 71 mode/geometry dispatch cases.
- 54 sequence-parallel prefill integration cases with 324 prenorm calls.
- 2,286,109 allocation boundary checks and 33,792 captured graph shapes.

The independent reviewer repeated the decode suite and V62's 110 numerical
attention cases and 890 ownership plans on V68. The additional independent
planner proof covers 3,036 cases including signed ID extrema. These CPU
models exercise source bodies but do not execute CUDA, Triton, NCCL or
Humming cubins. [Independent review](independent_review.md)

At 2,048 route slots, decode keeps 48 MiB of BF16 arenas, the same as the
indexed parent at that shape; routing metadata is about 25 KiB. Allocator,
graph pools, attention scratch and KV capacity still determine the measured
peak. H200 compilation, graph replay with changing routes, numerical and
later-turn agreement, exact output lengths, peak memory, TTFT and throughput
remain open. The source audit found no blocker against the current rules;
it does not establish the runtime gates or organizer approval.

## Experiment and next decision

The revised Nov 17 00:03 UTC private-check plan is V64, V62, then V68, all at
16 users. This replaces the unsubmitted V60 page-size experiment so the
integrated decode path gets an early H200 check. Existing queued jobs and
both component checks are unchanged. No automatic scored promotion is armed.

The experiment asks whether the integrated path passes every judge gate and
provides at least a useful 5% gain over the stronger passing component at the
same load. The 2x and 3x objectives remain unachieved; component operation
counts and launch counts cannot establish those end-to-end gains. A failing
capture, correctness, memory or TTFT result blocks promotion. A regression
requires diagnosis or a different design before spending another check.

Tree speculation remains a separate [design study](../v67_dflash_verify_capacity/tree_design.md).
The pinned worker needs packed sliding-window tree masks and ordered
accepted-path KV/auxiliary-state commits. Its illustrative acceptance model
does not establish a large serving gain, so no tree code is in this patch.
