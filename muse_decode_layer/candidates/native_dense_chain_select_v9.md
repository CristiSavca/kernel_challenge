# Full-chain dense backend selection, V9

Status: isolated, CPU source-preservation checks passed; independent critic
approved a bounded local comparison. Preparation runtime, CUDA correctness
and performance remain unverified. Active decoder and attention V5 are frozen.

Candidate: `native_dense_chain_select_v9.py`, SHA-256
`3d5b34620a05faa628c30077728328a28e362b4c9dfc54dbb9ee6d5f6dd4d58c`.
Parent: scored decoder `4b216098993e4112`, SHA-256
`80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`.

## Evidence, hypothesis and kill gate

The decoder has two nested dense selectors. `bfp__autotune` first selects an
inner implementation for each projection, timing it alone. `_tune` then
selects the outer combined/separate projection path, also in isolation.
The actual decoder adds the residual boundaries, expert routing, PDL overlap
below 33 tokens, and SK2B/UV/UB2 fusion. Existing `PIN_PICK` comments already
document cases where isolated choices lost in the chain, including a roughly
9 us difference for the T32 O projection. Those are historical source
comments, not new measurements establishing a V9 gain.

Falsifiable hypothesis: selecting the existing O, pooler/router and unpooler
implementations in the real post-attention chain yields at least 3% total
decoder improvement over the frozen incumbent, without new arithmetic.
This measures one defined dispatch-policy change, not a tile or stage sweep.

Corrected V8 trace attribution: `rq_deep_tma_pooler` is also used for the
unpooler. Both M128 traces in
`experiments/modal/20261104T040903Z-f1613663/result.json` contain `_route`,
not `_route_split`; the combined pooler/router path therefore remained in
use. QKV is pinned to torch, and RQ TMA is offered only for pooler/unpooler,
so the changed RQ kernel is the **unpooler**. Its single-profile comparison
was approximately 21.719 us cuBLAS versus 21.926 us RQ. This supplies no
positive evidence for pinning RQ. The independent critic withdrew its
earlier pooler-only recommendation after this attribution correction.

No raw profile establishes 3% available headroom. Roughly 18-25% of each
affected load is selectable post-attention dense work after excluding QKV;
T1 and T8 remain unchanged. A 3% overall improvement would therefore need
roughly 20% average reduction of that selectable work across four loads.
This is ambitious for already tuned backends. The theoretical ceiling of
removing all such work is much larger, but has no relevance to expected
backend-selection gains.

Proceed only if a bounded paired whole-decoder comparison demonstrates
repeatable >=3% improvement in the six-load metric with no >3% load
regression. Kill on a preparation timeout, correctness failure, unchanged
dispatch/no useful whole-decoder gain, or gains explained by different
expert work. Do not run another parameter search after a null result.
Local results require an exact-source official private check before scoring.

## What changes

Every pre-existing CUDA/CPP/Triton string, kernel and runtime function is
unchanged. The CPU verifier removes only the two new helpers and three
prepare statements and then proves that the whole module AST equals the
incumbent. UV, SK2B, UB2, RQ, attention, expert math, every route and every
precision rule are retained.

After the old tuners and `PIN_PICK`, preparation performs one coordinate
pass over O, pooler/router and unpooler for T16/T32/T128/T256. Each trial
uses an existing implementation already offered by the original tuners.
The inner router backend, QKV backend, MoE backend and all fusion rules are
held fixed. T<=8 is excluded because UV and UB2 bypass these picks.

For each token count, preparation creates three common synthetic streams
and attention outputs from independent generator seeds. It selects the
busiest of eight EP partitions for those synthetic routes, using the same
partition for every choice. This does not use `kernel_fn` inputs or move
any input-derived computation outside the timed call.

Before timing, each alternative must produce exactly the same eight global
expert identities per token on all three inputs, and its output and stream
must each stay within 1% relative L2 of the current selection. Exact route
identity prevents choosing a dense backend merely because fewer experts
ran. This conservative gate can reject a numerically valid alternative.

Each accepted numerical trial is compared with six alternating-order pairs
per seed, from a read-flushed L2. The geometric mean of the three median
paired ratios must exceed 1.02 and every seed must be nonregressing before
the dispatch changes. The choice is measured in `_after_attention`, which
includes O, boundaries, pooler/router, routing, MoE, unpooler and final
boundary. Attention and QKV are unchanged and excluded from this selector;
the later validation gate uses the entire decoder.

The selector has a 25-second wall budget and stops starting work at 270
seconds from preparation entry, reserving time for existing attention and
scratch checks. These are soft boundaries between operations: an in-flight
JIT compile or GPU call cannot be preempted. Any selection exception restores
both original dispatch dictionaries; ordinary budget exhaustion preserves
only completed, validated coordinate choices. Existing CUDA failure checks
still apply; a fatal device error cannot be made safe by dictionary restore.

Scalar logs record the token count, coordinate, before/after outer and inner
choice, selected paired score, local partition, route count and distinct
experts for each seed. Detailed trial outcomes remain in
`state["dense_chain_log"]`. The logs also record budget skips, elapsed time
and fallback exceptions. Production `kernel_fn` contains no diagnostics.

## Review and validation

The critic supported one bounded full-chain selector, with common seeds,
existing compiled choices, conservative margins and incumbent fallback.
Its main concern was unrepresentative synthetic expert routing. V9 responds
with three common seeds, a busy synthetic partition, exact identity matching
and recorded route counts. This cannot establish representativeness for the
actual decoder, so complete-decoder comparisons remain mandatory.

The critic's source review found no lifecycle blocker in `4c0c1305...` and
approved a bounded local comparison. The final `3d5b3462...` differs only by
moving flush/sink allocation inside the existing exception/rollback guard.
The critic correctly noted that per-coordinate numerical differences can
accumulate, `gate=None` covers the fused attention chain rather than every
fallback, and a budget skip or no accepted choice means no distinct policy
was exercised. These remain explicit validation limits.

`native_dense_chain_select_v9_verify.py` passed locally. It proves complete
source preservation outside the helpers/prepare hook, checks the frozen
parent hash, and exercises the pure candidate enumerator without loading
Torch or Triton. Python AST and whitespace checks passed. No CUDA allocation,
compile, upload or official check was initiated by this agent.

The next run should preserve the prepared dispatch logs for both sources,
check that the selector changed actual launches, report preparation time,
and compare all scored token counts with fresh inputs and cold-L2 graphs.
Keep the parent and V9 expert backend identical. Report route counts and
identities outside timing. Do not interpret an internal 2% post-attention
tuner result as a 3% whole-decoder result.

Storage added: one ~800 KB source candidate, one small verifier and this
note. No source assets, active files or result evidence were deleted.
