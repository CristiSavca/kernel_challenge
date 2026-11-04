# One new MoE instruction-removal lead: publish final route scalars

2026-11-04, bounded read-only review begun 14:28:49 UTC. Exact parent is
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
No candidate, GPU work or submission. **One isolated source-local experiment is
plausible; no performance claim or automatic implementation is implied.**

Reviewed the closed SC6, shared-A, L2-prefetch, WAIT2/register redistribution,
compact/NW4, FT128, metadata/residency and MFIN notes first. This proposal changes
neither weight encoding nor the main pipeline. It does not repeat a ring-depth,
cache-hint, consumer-count, finalizer-kernel or completion-counter experiment.

**Actual duplicated work.** In `OKC_WGS && OKC_WGSFINSS` finalization, each CTA
owns one token at a time and 512 consumer threads own eight output columns each.
Each route's squares and warp reductions are already computed once. Threads 0–7
then sum the 16 warp partials, in the existing order, and store `tsum` at
`red[NW*8+tid]`. After the existing CTA barrier, every output thread recomputes
the same scalar for each live route:

```cpp
p.topk_w[tk * 8 + sl] *
    rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f)
```

**Fixed proposed delta.** In that existing `tid<8` reduction block, preserve the
entire `tsum` accumulation, then have the same route owner evaluate that exact
FP32 scalar expression and store the result in the same shared slot. Guard on
`S.pos[tk*8+tid]>=0`, otherwise store 0. Use that shared metadata read rather than
dynamically indexing register array `qv`. After the unchanged barrier, replace
the scalar expression with a shared load. Preserve route iteration, cached y
values, every final FP32 product/add, and every BF16 conversion exactly.

The route-sum region is `red[0..127]`; the eight result slots are 128..135. The
existing barrier already publishes all eight slots. No read aliases another
owner's write. The final CTA barrier before moving to the next token remains.
All producers and phase-B stores already complete before this finalizer begins.
No readiness protocol, routing, expert job count, work distribution or PDL edge
changes. Leave tiny/MID and non-WGS fallback finalizers untouched.

Exact intended owner replacement, inside the existing `if (tid < 8)`:

```cpp
float tsum = 0.f;
for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
float r = 0.f;
if (S.pos[tk * 8 + tid] >= 0)
  r = p.topk_w[tk * 8 + tid] * rsqrtf(tsum * (1.f / HD) + 1e-8f);
red[NW * 8 + tid] = r;
```

Exact consumer replacement inside the unchanged `if (qv[sl] >= 0)`:

```cpp
const float r = red[NW * 8 + sl];
```

H=4096 and NT=512 are enforced by the existing `static_assert(HD == NT*8)`;
there are no partial output columns. Token assignment `tk=blockIdx.x;
tk<p.M; tk+=gridDim.x` stays identical. At the current grid132, M128 has at most
one token per CTA; M256/384 include multiple tokens per CTA. An outer MMAX=384
chunk's short remainder keeps the same mapping. Each valid route's
owner is `tid=sl`; every owner writes its slot, including zero for remote routes.
A token with no local route still writes zero output if the inherited earlier
phases reach this finalizer. No new M0/global-zero-job support is claimed.

For a token with R local routes, source scalar evaluations fall from 512R to R:
511R repeated rsqrt/multiply expressions are removed. Shared scalar stores and
consumer loads are unchanged in number/width; eight owner metadata reads are
added. Global route-weight loads move to owners, but cache broadcasts mean this
is not evidence of removed HBM bytes. GPU compilation may already optimize some
uniform work: these are source counts, not measured instructions or a 512× speedup.
If lowered directly as SIMT operations, the consumer expression corresponds to
16R warp executions (at most 128) versus one owner-warp expression with R active
lanes. This is an instruction-issue model, not observed SASS or a latency forecast.

**Correctness case.** A stored/reloaded FP32 `tsum` has the same precision as its
FP32 register value. The same owner sum, constants, rsqrt implementation and
route weight produce the same scalar expression; no algebraic reassociation is
requested. Inactive slots are not consumed. All-live, one-live, all-remote,
multiple token iterations and chunked calls retain the existing control flow.
This does not fix the parent's separate global-zero-job problem. Actual compiler
contraction and GPU arithmetic still need checking; CPU math is not a substitute
for `rsqrtf` validation.

**Cost and measurement limits.** Scalar arithmetic moves before the existing
publication barrier, so it extends the owner path while reducing duplicate work
afterward. The throughput/latency tradeoff is unknown. Extra owner temporaries
could affect the common kernel register allocation; retain compiled resource
evidence. The 64 cached y values/thread and all finalizer traffic remain. Only
MAIN shapes are affected; this is a modest finalizer opportunity, not a whole
decoder architecture breakthrough. Saved post-B tail measurements of roughly
5–8us include other work and cannot be attributed to this scalar computation.

A cheap isolated check can be justified by the explicit removal without a large
gain hurdle. Preserve exact parent math and all other CUDA branches, inspect
the small diff and actual resource output, and compare both current parent
receipts with unchanged small-shape controls. Do not create a parameter sweep.

**Second lead rejected during this review.** Precomputing or hoisting
`(1+normw)*global_scale` in the phase-B epilogue is less supported: compiler CSE
may already hoist same-column values across NTT, and a prepared per-expert FP32
array increases resident/loaded weight data and introduces rounding/interface
questions. No net generated-instruction reduction is established. KPERM already
removes repeated scale conversion; scale/3-scale tables remain the closed decode
tradeoff. No second candidate is recommended.
