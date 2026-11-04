# Exact SK2 stage directory — plan only

2026-11-04. Proposed separately from rejected global-escape V10. Parent is exact
decoder `80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`.
No candidate, GPU run, or source approval exists. Request an independent critic
before implementation. Preserve the parent's shared activation reads, all escape
warps, FP32 correction planes, ring stages, waits, and handoffs.

**Hypothesis:** the current escape warps unnecessarily scan the same immutable
table. A stage directory can make only the owning warp visit a stage's entries,
without introducing global activation gathers or changing numerical order. The
gain is limited to metadata traversal and repeated index comparisons; it neither
reduces correction FMAs nor guarantees a shorter critical path.

## Exact layout and minimal interface

Current `bfp_sk2_table` returns an int2 table `[ncta,pmax]`; entry0 is `(count,0)`;
entries1..count are `(row<<16|k, delta_bits)` in the existing K order. All SK2,
fused-pooler, UV, and SK2B consumers receive this same tuple interface.

For a selected table, append a directory after its original even `pmax` slots:

- Keep entry0.x and entries1..count byte-identical. Do not sort again; equal-K
  entries must retain exactly the old order.
- Set entry0.y to `old_pmax`, the directory's offset in int2 slots.
- Store int32 `d[0..S]`, where `S=K/KST`, `KST=64*WSK*UPW`, and
  `d[s]=1+number_of_entries_with_k<s*KST` for that CTA. Thus `d[0]=1` and
  `d[S]=count+1`; empty stages have equal consecutive offsets.
- Pad the directory to16 bytes: `D=16*ceil(4*(S+1)/16)` bytes.
  `pmax_new=old_pmax+D/8` remains even. Padded words are zero.
- For an unselected table, return the exact old table and old pmax, with
  entry0.y still0. This includes empty/insufficient-count tables.

No CUDA signature, state tuple, weight stream, or correction value changes are
needed. The Python table builder needs geometry for its static work filter;
`bfp_sk2_prepare` is its only existing caller and already has the config. Keep
the legacy `bfp_build_patch_table` for other backends unchanged. Expected source
scope: SK2 CUDA, table builder, its preparation caller, one pure metadata helper,
and the extension namespace. No global-read V10 code is imported.

**Sharing invariant:** existing key `(layer,n_rows,R,NT,NWR,WSK,UPW)` includes
WSK/UPW, so KST is fixed within every shared table. Explicitly assert this over
all emitted ordinary/fused/UV/SK2B dispatches. The fused `pooler` table changes
NESC4→3 across TOKR, but KST stays512; a stage directory works for either NESC.
Do not partition the table permanently by `stage%NESC`. If future table sharing
allows different KST, retain separate geometry-keyed tables or use a K64 directory
indexed by stage*UPS; neither broader change belongs in this first experiment.

## Kernel access, exact order, and old consumers

After the existing pbar wait, read entry0.y. Only `COMP && CL==1` with a nonzero
directory offset uses the new traversal. Keep the old scan as an exact fallback.
At each original stage, wait for its full barrier as before. Non-owning escape
warps skip entry traversal but still synchronize their warp and arrive on the
empty barrier. An owning warp loops `pe=d[st]..d[st+1]-1` using the original
shared-input address and correction expression. Preserve the bounds check and
all padded-token behavior. No per-entry stage/modulo calculation is needed.

Stages are visited in the same order; within each owned stage, entry order is
the original table order. Therefore each correction plane's FP32 addition
sequence and the final cross-plane reduction remain identical. No atomics,
correction-warp count change, or new CTA barrier is introduced.

FUSE router stages are handled by the existing `st>=nst` branch before any
directory lookup; they still wait and hand back the ring slot. Uncompressed
router and clustered paths keep the original traversal. If clustered support is
ever enabled, indices would use `gs0+st` while ownership remains local `st%NESC`;
do not accidentally use global-stage ownership in this first CL1-only change.

The old pmax-based global→shared copy and transaction count include the appended
tail automatically. Directory reads occur after that same pbar. Shared table
storage and zrow move by D through the existing pmax arithmetic; verify alignment
and every shared offset, rather than adding a separate overlapping allocation.
Actual launch/occupancy and SK2B capacity queries must include the larger pmax.
Keep the original NSTAGE and ring geometry. Disable directory construction for
any immutable geometry/table whose added shared bytes cannot fit the real
allocation limit; do not reduce the ring or weaken capacity checks to make it fit.

## Work filter and reason to consider one test

Actual official escape counts/pmax were not recorded. The0.5% compression
eligibility threshold is a ceiling, not a typical escape density. A blanket
directory is unjustified. Use a whole-table immutable work filter, accounting
for the fact that every CTA copies the fixed-size tail even if its count is0.

For N the minimum NESC across eligible consumers of that shared table, the old
source traverses approximately4(N+1)E metadata bytes/CTA, excluding boundary
lookaheads. A directory visits8E entry bytes plus8S offset bytes, and adds D
bytes to the initial copy. A conservative work filter is:

`4*(N-1)*sum(E_i) > ncta*(D+8*S)`.

This compares logical bytes across different memory spaces; it is **not a latency
bound or proof of benefit**. It deliberately excludes parent lookahead savings.
Use exact immutable counts, not workload seeds, timing, or a tuned threshold.
Collect the full consumer set before choosing N. Report selected/fallback tables,
counts, mean/max entries, pmax, directory bytes and geometry during preparation
so the eventual result establishes whether the mechanism ran.

| Representative table | S / min N / D | Required mean E | Distribution-only mean E |
| --- | --- | ---: | ---: |
| qkv R24, K8192, KST512 |16 /4 /80 B | >17.33 | ~27.26 |
| fused pooler R32, K8192, KST512 |16 /3 /80 B | >26 | ~36.34 |
| separate pooler32 R32, K8192, KST256 |32 /4 /144 B | >33.33 | ~36.34 |
| unpooler16 R32, K4096, KST256 |16 /4 /80 B | >17.33 | ~12.19 |
| unpooler32 R64, K4096, KST256 |16 /3 /80 B | >26 | ~24.38 |

The final column is the critic's independent CPU Gaussian/sqrt(K), RN-even BF16
model, not official weights or expected timing. It suggests some qkv/pooler
tables could qualify while unpooler tables fall back. This is a concrete, limited
reason to investigate metadata traversal, not evidence for a broad3% decoder
gain. A K64 universal directory would cost528 B at K8192 and likely fail this
filter; avoid paying for compatibility the current sharing key does not need.

## Critic and validation gates

Before any implementation, critic must accept the exact layout, sharing/consumer
map, whole-table filter, and source-byte cost assumptions. Require actual-entry
CPU comparisons on zero/one/dense tables, empty stages, repeated K values,
boundary K values, all active KST/NESC, and physical UV inputs. Compare each
plane's complete ordered `(entry,input-address)` trace to parent. Prove fallback
table bytes and old non-target CUDA preprocessing remain unchanged; check
pbar bytes, zrow alignment, actual shared footprint and BND capacity queries.

Only after a frozen implementation and exact-source critic GO would one private
whole-decoder check be considered. Inspect preparation selection evidence and
matched qkv/pooler traces,13 correctness tests and12 benchmarks. Reject an
inactive experiment, lost fusion, material regression, or no useful whole-decoder
gain. Do not tune gates from the result or reintroduce the failed global-input
mechanism. Root alone owns any GPU/check decision.
