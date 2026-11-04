# WGS instruction footprint: conditional steady-loop lead

Read-only review of source `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`,
native CUDA `85fb417494c6eab5baabad6813275b256f7bae84c0329d0e94cb61a15030445c`,
and retained CPU build `experiments/modal/20261104T155445Z-1ed7603c`.
The complete BIG1 section is 340,480 bytes / 21,280 instructions. SASS hash is
`0c896a5f3fa98170e1df09c50efa5dab60fd21a7b0bb62d1caec7707e55b2d27`;
cubin hash is `28fbd6ab9577e40741469244d6e9b096adf41f3523163712681a1bb0b2bcc92c`.
BIG0 is incomplete. Existing provenance/count audit:
`experiments/iterations/20261104-current-kperm-sass-result-critic.json`.

**Decision:** one technically plausible code-size experiment exists, but the
saved counters do not establish an instruction-fetch bottleneck. No source
change, compilation, provider call or candidate is authorized by this note.

## What a segment can actually execute

There are eight NTT specializations, two phases, and two stage bodies per
specialization: first and steady. This gives 32 static stage bodies, each with
16 HGMMA instructions. A segment selects one phase/NTT, executes its first body
once, and its steady body zero to fifteen times. The stage-count loop already
has `#pragma unroll 1`; it is not sixteen statically copied stages. A one-stage
split does not execute the steady body. A whole-unit or split epilogue follows.
Another segment may select another NTT; no claim that one SM only ever visits
one specialization is justified.

The following counts follow actual SASS branches from each switch-case entry
until the phase's B0 join. Both arms of conditional branches are included, loops
are visited once, and `BSYNC B0` terminates the switch arm. They are conservative
intrasegment reachable-code envelopes, not dynamically executed instruction
counts or measured hot sets. All traversals contain exactly 32 HGMMA instructions
of the expected N width and no HGMMA of another specialization. Some cold
tails are noncontiguous, so min/max-PC spans would overstate their footprint.

| NTT | Phase A entry | A instructions / bytes | Phase B entry | B instructions / bytes |
|---:|---:|---:|---:|---:|
| 1 | 0xadc0 | 939 / 15,024 | 0x2fcb0 | 866 / 13,856 |
| 2 | 0xe800 | 991 / 15,856 | 0x33260 | 897 / 14,352 |
| 3 | 0x6cd0 | 1,046 / 16,736 | 0x2c2c0 | 934 / 14,944 |
| 4 | 0x16ed0 | 1,109 / 17,744 | 0x3a9e0 | 979 / 15,664 |
| 5 | 0x12600 | 1,172 / 18,752 | 0x36a80 | 1,021 / 16,336 |
| 6 | 0x25a30 | 1,247 / 19,952 | 0x471a0 | 1,063 / 17,008 |
| 7 | 0x1b3f0 | 1,306 / 20,896 | 0x3e6e0 | 1,095 / 17,520 |
| 8 | 0x20520 | 1,368 / 21,888 | 0x42ae0 | 1,139 / 18,224 |

The stops are `.L_x_107` (A, PC0x2a7c0) and `.L_x_442` (B, PC0x4b3b0).
Case mapping comes from the actual BRX target lists and NTT6/7 comparisons,
cross-checked against HGMMA widths. Source annotations alone are unsuitable:
scheduled operations can retain a neighboring specialization's line annotation.
These envelopes exclude common routing/gather, producers, outer dispatch,
phase-final pending flush and final normalization, and do not prove all included
predicate combinations occur on one execution.

The current NCU capture's actual M32 input has 16 jobs, all NTT1. M256 has
27 jobs: NTT1=16, NTT2=5, NTT3=2, NTT4=1, NTT5=1, NTT8=2. Thus the full
340 KB is particularly misleading as M32's hot-code claim. These are two saved
inputs, not specialization assumptions or a reason to remove any NTT branch.

## One bounded reduction in duplicated code

Roll only the four-iteration `lg` loop in the **steady** stage, retaining full
unrolling in the first stage and retaining the current unrolled `h` and `jj`
loops, GS2, two AF banks, NTT variants, geometry and all memory traffic. A
constexpr-dependent unroll hint or equivalent explicit structure could express
this; actual compiler acceptance/compliance is untested.

For GS2, `ci = 2*lg + h`, hence `ci & 1 = h & 1`. Both AF-bank indices and
NTT1's accumulator selection can therefore remain compile-time constants when
`h` is unrolled; `js = 2*h + jj` and scale-pair selection also remain constant.
The next code load, final weight-slot release, ordered descriptors and wait1
remain in the same logical iteration order. The last wait1 retires the older
AF bank before its next write, including the rolled-loop backedge. The first
stage's mma0 initialization, previous-X release and FCI3 flush remain untouched.
The steady-stage previous-X release still occurs only at `lg=0,h=0`; it cannot
be applied at every loop iteration. Final wait0 and epilogues remain unchanged.

Nominal arithmetic-body copies become 4+1 instead of 4+4 per phase/NTT:
37.5% fewer static copies of the decode/HMUL/MMA body, **not** 37.5% fewer total
instructions, dynamic work or latency. Loop control, runtime descriptor offsets
and final-load/release conditions add work and may limit scheduling across groups.
The rolled backedge may alter register allocation. The parent's 96 registers,
zero local memory and absence of C751x in this build do not establish those
properties for a rewrite. Exact emitted AF indices, unchanged commit/wait order,
resource admission and absence of newly serialized WGMMA must be checked before
claiming that it preserves the intended pipeline. Outlining stage code or
merging FIRST is not part of this single proposal.

## Missing discriminating measurement

The 14 selected metrics in current capture
`experiments/modal/20261104T150731Z-ef5e7aac/result.json` contain neither an
instruction-fetch stall nor an instruction-cache request/miss metric.
Issue-active, eligible warps and math/MIO/LG pressure do not fill that gap.

One bounded extra counter, if later authorized and advertised by the installed
NCU, is `smsp__warp_issue_stalled_no_instruction_per_warp_active.pct`.
[NVIDIA's CUPTI mapping](https://docs.nvidia.com/cupti/13.0.0/main/main.html)
maps this name to instruction-fetch stalls. The
[Nsight guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
explains that No Instruction includes both waiting to be selected for fetch and
instruction-cache misses; high values alone do not prove cache capacity caused
the problem. Negligible values would weaken this proposal materially. High values
would support a bounded source experiment, not predict its recoverable latency.

This would be an addition to the existing exact-source/cold-graph M32/M256
measurement, with no new metric substitution or retry. Availability and replay
pass cost have not been queried. NVIDIA also documents ICC request/hit/miss
counters in its [Hopper instruction-cache case study](https://developer.nvidia.com/blog/?p=86868),
but they are not included in this one-counter proposal. That case study explicitly
distinguishes hot-code footprint from total executable size; its speedups do not
transfer to this kernel. No current timing evidence establishes a gain here.
