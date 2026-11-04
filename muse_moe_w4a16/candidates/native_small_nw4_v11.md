# Small-token NW4/NS2 candidate, November 4

One isolated implementation from the scored KPERM parent, awaiting an official
GPU check. It has not replaced either active submission. Root owns submission
and polling; the author used CPU checks only.

| Artifact | SHA-256 |
| --- | --- |
| Parent `native_wgs_kperm3_v9.py` | `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0` |
| Candidate `native_small_nw4_v11.py` | `34ee99c242c4ffb5028ec3511fd8740c5f15461e1c73e9c1343ae6611573e3e3` |
| Candidate CUDA string | `72d66697a8000cad91d5d4550798762a4374b3fb9648ac780fe9941b17f1279f` |
| Actual-source verifier `tools/test_small_nw4_cpu.py` | `4634fc03226cba3bd76cbeb14d2fc183e4e2e187f3f07b3d9934c1e6f4e16408` |

The earlier source prefix `265ca69b` has identical executable code; the final
header and Python tuple comments distinguish inherited measurements from this
unmeasured rewrite. Licenses and upstream notices remain intact.

## Mechanism and exact scope

Only TINY, TINY2 and MID builds change from eight compute warps plus one helper,
one CTA/SM, to four compute warps plus one helper, two CTAs/SM. Their weight ring
changes from three to two slots. The host stops reserving the unused fat path's
workspace for this exact TINY+WS+NW4 build. Shared allocation becomes 101,632
dynamic bytes per CTA; two CTAs plus the inherited 1 KiB static allocation each
fit within 233,472 bytes. This is a capacity calculation, not a runtime result.

Each unit still produces sixteen output rows from all 4096 K elements. Every
job still requests 512 A units and 256 B units, each 36 KiB: 768 TMA requests and
27 MiB of packed weights. Four compute warps each process sixteen chunks using
two inherited eight-chunk scale records, 512 bytes apart. All four accumulators
span both scale groups. Both the PRMT scale path and general E4M3 LUT path retain
their exact NVFP4 operand construction and every route.

Two CTAs retain eight compute warps per SM, add a second helper, and increase
aggregate ring slots from three to four. They also duplicate routing/prologue
work and more per-job activation-cache loads. The activation cache grows from
64 to 128 registers per thread. Parent small binaries already used 144–149
registers, so spills or slower scheduling are a substantial risk; capacity two
alone supplies no performance evidence. Four-way instead of eight-way FP32
partial reduction changes association, not precision or the set of terms.

Both compiled BIG specializations must report actual occupancy capacity at
least two at the precise 160-thread/dynamic-shared allocation before either
ordinary or PDL launch doubles the grid. Opt-in allocation, device queries,
the shape scope and incoming nSM grid are checked. One host marker records
threads, shared bytes, both capacities, registers and local bytes. A failed
gate aborts before any enlarged spin-barrier launch; there is no silent fallback.

WGS/KPERM, weight packing, remaining specialization flags, global scratch,
all epilogues, PDL trigger/wait, FINCOL and FINPF stay inherited. FINCOL now has
eight segments rather than four, with four values per thread in either case.
For M≤16 each active finalizing CTA still owns at most one segment; there are
more metadata readers and different finalizing CTA placement.

## CPU evidence and limits

`experiments/iterations/20261104-small-nw4-source-cpu.json` records the completed
author run. The verifier executes the actual candidate `dq4`, `load_x_regs` and
inner chunk/scale loops as native CPU C++ in four MULDEC/PRMT combinations. It
checks 23,592,960 MMA operand values across every lane, warp, job count 0–8,
A/B activation source and four coordinate patterns. It reconstructs expected
values from original unpermuted FP4 codes/E4M3 scales, using actual Python weight
packers; it does not merely compare the new address formula with itself.

The separate packed-coordinate check covers 524,288 codes, 32,768 scales and
all 32,768 activation coordinates per job, in both inherited code encodings.
Eighteen preprocessed builds using the old flags are byte-identical to the
parent, including MAIN and fallback combinations. AST checks preserve every
other top-level node and packing helper. Routing, helper/TMA functions, final
grid/norm/reset tail and activation-loader source are unchanged. Launch-order
checks place both occupancy gates before the grid change and PDL choice.

The ownership model covers J=0–44, grids 132/264/396/528, 59,400 ranges and
3,041,280 unit assignments, including empty B ranges, NS2 wrap, WSBLK1/2 tails,
every A credit and final output coordinate for M=0–16. Grid×compute-thread
count remains unchanged, so inherited scratch reset coverage remains complete.
The independent critic's separate asynchronous model is saved in
`experiments/iterations/20261104-small-moe-ring-critic.json`: 22 interleavings,
372,072 events, and two intentionally broken deadlock controls. Its assumptions
include eventual TMA completion and full-grid residency; it is not GPU ordering
or scheduler validation.

CPU checks do not establish tensor-core numerical results, CUDA compilation,
register/spill allocation, actual residency, forward progress or speed. Fresh
official graph replay and numerical checks remain mandatory. Plan evidence is
in `tools/SMALL_MOE_TWO_CTA_PLAN.md`. The independent critic reran the complete
actual-source suite successfully and issued GO for one private check of exact
34ee99c in `experiments/iterations/20261104-small-nw4-source-critic.json`.
That verdict permits the GPU experiment; it does not approve promotion.

## Bounded decision

The target is at least 15% less normalized latency across M1/M8/M16, no affected
case more than 3% worse, and stable unchanged WGS controls. A clear regression
kills this candidate without an automatic parameter sweep. A useful standalone
result requires a separately reviewed integration into exact scored decoder
97af3994, preserving decoder-specific small flags, followed by a full-layer
private check before scoring.

Historical profiler-share scenarios translate a 20% small-MoE reduction into
roughly 4.27% whole-decoder reduction; even 50% gives only about 11.08%. These
are conditional estimates, not expected gains. This candidate alone has no
established path to the approximately 14.1% remaining decoder latency gap.
