# Independent SK2 escape corrections

November 4, 2026. Parent decoder SHA:
`80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`.

Hypothesis: sparse exact BF16 escape corrections unnecessarily hold back the
regular compressed-weight TMA ring. Decoupling them can improve several small-T
dense projections while leaving all weight bytes, regular MMA operations,
precision, fusion and preparation unchanged. This is a hypothesis, not a
measured bottleneck or predicted speedup.

The main loop already decodes weights directly to tensor-core registers, so
generic compressed-register GEMM is not new here. ZipServ (arXiv 2603.17435v1,
March 18, https://arxiv.org/html/2603.17435v1) supports keeping common decoding
regular and fragment-aligned, but its 11-bit bitmap scheme and Ada/consumer
Blackwell speedups are not a drop-in H200 result. SplitZip (arXiv 2605.01708v1,
May 3, https://arxiv.org/html/2605.01708v1) also separates regular encoded values
from sparse exceptions; its transfer-codec throughput is not a GEMM gain.
GAAC (arXiv 2610.38210v1, October 28,
https://arxiv.org/html/2610.38210v1) avoids per-element exceptions with group
modes, but those new formats would add addressing/decode work to our existing
12-bit path. The selected test changes the dependency structure, not the codec.

## Precisely isolated change

Retain the existing three/four escape warps, their individual FP32 correction
planes, block dimensions, ring depth, shared allocation, patch table, final
reduction order and `cbar` participant count. For `COMP && CL==1` only, every
escape warp independently walks the same K-sorted table and computes only
entries whose `st=(k/KST)` has `st % NESC == ew`. Read BF16 `x[lane*K+k]`
from global memory, after the existing all-thread PDL wait. Then arrive at the
existing final correction barrier. Remove these warps from ring-empty arrival
counts and remove their per-stage full waits and empty arrivals. Regular
consumer arrivals remain exactly `NW`, including fused router stages.

The original stage-based implementation remains for clustered variants. The
uncompressed router `COMP=false` remains unchanged. No single-warp consolidation,
new density threshold, new codec, tile search or precision change is bundled in.
Keep source and extension identity distinct from the incumbent.

## Input and lifetime audit

`launch_sk2` constructs its TMA map and `xp` from the identical `x.data_ptr()`
and identical M,K. The kernel does not transform input x before MMA; transformations
are in its epilogue. SK2B receives the MoE BF16 output; SK2B1/SK2T receives BF16
attention output. Fused pooler/router UV mode receives actual interleaved BF16
u/v rows with M=2T, written by the preceding output-projection epilogue. It
combines those rows with norm factors only after the dot products. The new
corrections must use the actual u/v rows without applying norms themselves.

Every thread already executes `griddepcontrol.wait` before entering the escape
branch; plain launches use normal stream ordering. No new activation read may
move above that point. `ptab_s` and all correction planes live beyond the TMA
ring and survive its epilogue reuse. Each escape warp keeps the same table-entry
order within its plane, and all lanes, including padding lanes, still arrive at
`cbar` after writes. Consumers wait for cbar before reading corrections. Add no
CTA-wide barrier after producer or escape warp returns.

## Resource and traffic consequences

All following values are source-derived, not newly measured CUDA attributes.
Dynamic bytes shown exclude the unchanged `8*pmax` patch-table term.

| Configuration / TOKR | Stages in ring | Escape warps | Threads | Dynamic bytes +8*pmax |
|---|---:|---:|---:|---:|
| qkv8 /8 | 6 | 4 | 288 | 167424 |
| qkv16 /16 | 5 | 4 | 288 | 181744 |
| qkv32 /32 | 4 | 4 | 288 | 218848 |
| o8 /8 | 4 | 4 | 288 | 75232 |
| o8 /16 | 4 | 4 | 288 | 91616 |
| o32 /32 | 3 | 3 | 256 | 99920 |
| pooler /16 | 5 | 4 | 416 | 214512 |
| pooler32 /32 | 5 | 4 | 288 | 161520 |
| pooler8 /8 | 3 | 3 | 256 | 65936 |
| unpooler16 /16 | 5 | 4 | 288 | 112112 |
| unpooler32 /32 | 3 | 3 | 256 | 149072 |

No occupancy increase is claimed. Compiler register changes can still affect
occupancy, including SK2B's grid-residency check; its capacity query remains on
the actual modified specialization. Source/compile evidence must be checked.

Each escape now adds M logically requested BF16 activation values: `2*M` bytes.
Uncoalesced lanes can cost up to `32*M` bytes of L1/L2 sector requests per patch,
before reuse. Across E escapes the conservative request count is `32*M*E`.
These are request estimates, not DRAM measurements. The x array is small and
shared across CTAs, but a cache-hit assumption is not proof. Retaining four
warps avoids making all corrections serial in one warp. A slow correction
group may still replace ring stalls with a final correction tail.

## Validation and decision

Independent critic `result_and_strategy_critic` approved this revised plan for
implementation only. Required CPU evidence: sorted empty/dense/edge patch
lists, every active SK2 geometry, M1..32 including physical UV M=2T, exact
per-plane entry order and TMA-to-global address equivalence, and a ring model
showing slots cannot refill until every actual consumer releases. Verify that
only the intended embedded CUDA source and extension name change.

Then require an independent exact-source review before root submits one
official private decoder check. A full decoder check is necessary because UV,
pooler/router fusion and SK2B epilogues are not all exercised by standalone
projections. Preserve all large-T paths as controls. Promote only after full
correctness and useful measured whole-decoder gain; reject if gains are noise,
the sparse gather cost dominates, or any nearby shape regresses materially.
No GPU allocation or official submission is authorized to this author.
