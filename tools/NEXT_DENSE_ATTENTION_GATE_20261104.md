# Next dense/attention mechanism: no implementation GO yet

2026-11-04, following the terminal stage-directory result. Current exact decoder
is `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`, scored
job `7a8c4941741645ee` at2.1811831350x. The saved leader snapshot implies roughly
14.1% normalized latency still to remove; no new live board query was made here.

**Decision: NO GO for a new dense or attention implementation on present
evidence.** One distinct dense architecture below is worth retaining as a
research hypothesis, but its required payoff is unsupported. Do not repeat
the directory/global-escape rewrites, mix them into KPERM, or run a tuning sweep.

## Required scope from current evidence

`experiments/iterations/20261104-next-dense-attention-roi-gate.json` records the
current public profile hash, all12 case rows and scenarios using the actual
46/16 sliding/full load weights and reported normalized benchmark times. For
each scenario, an operator fraction is removed from that case's profile share,
then the six weighted loads are combined geometrically. These are conditional
scenarios, not measured gains, guaranteed bounds, or critical-path attribution:
PDL rows include waiting and can overlap.

| Scope | Operator reduction needed for5% whole decoder |20% operator-reduction scenario |
| --- | ---: | ---: |
| All dense projections |14.32% |6.99% |
| All small-T dense projections, T<=32 |18.86% |5.31% |
| All large-T dense projections, T>=128 |54.44% |1.78% |
| All attention |33.76% |2.96% |
| Q1 full attention only |71.49% |1.36% |
| Small-T QKV only |85.87% |1.15% |

Even zero-cost deletion of Q8 full-attention rows gives only4.53% in this model;
zero-cost deletion of all explicit route/boundary rows gives4.95%. Thus a small
attention subpath or another boundary-only fusion cannot credibly meet the
requested5% gate. These deletion scenarios are deliberately unrealistic checks
of scope, not claims that deleting actual computation is permissible.

## Strongest distinct dense hypothesis considered

Use smaller output-row tiles and enough simultaneously resident SK2 CTAs to
maintain more independent TMA/decode/MMA pipelines per SM. Preserve the original
12-bit lossless BF16 representation, exact escape corrections, shared activation
loads and model computation. This differs from decoupling corrections or adding
an escape directory: it changes the main projection resource unit and dispatch.

Source anchors in exact97af:

- `bfp_SK2_CONFIGS`, `bfp__sk2_geom` and `launch_sk2`: output-row tile, K split,
  ring storage and actual occupancy admission are coupled.
- `sk2_kernel` consumer loop: each unit loads all compressed words and activation
  fragments into registers before decoding and MMA. `decode8` already uses
  packed LOP3/PRMT reconstruction, so generic advice to decode directly into
  tensor-core operands is already implemented.
- Fused pooler/router requires `CL==1`, `UPW==1`, `UPS==8` and `WBYR<=WBY`.
  Simply halving compute warps and doubling UPW violates that fused path's
  assumptions. Losing its fusion would repeat a confound observed in global V10.
- QKV's R24 configuration has96 output CTAs. Merely shrinking that tile can
  improve SM coverage, but QKV's entire small-T scope is only a5.84% deletion
  scenario. It cannot supply a credible5% gain by modest utilization improvement.

The broad version would have to cover pooler, QKV, o and unpooler while retaining
UV, fused router and boundary kernels. Smaller row tiles increase activation
loads, metadata/producer/escape work and grid arrivals. Shorter rings reduce
per-CTA lookahead. More resident CTAs do not establish higher useful throughput.
The failed compact-WGS experiment already demonstrates that resource fit can
coexist with a substantial regression; do not transfer a two-CTA headline here.

There is currently no measured SK2 register/issue/barrier breakdown establishing
which main-path stalls this architecture would remove, nor an exact feasible
resource design for all fused consumers. The stage-directory experiment proved
activation and preserved fusion but produced no useful score gain; it supplies
actual sparse escape counts, not evidence that the main pipeline can improve19%.
The source-level `trace`/`dg` hooks could support a separately reviewed local
diagnostic when budget and authorization permit. Such diagnostics must remain
outside any scored candidate; no diagnostic or allocation is proposed now.

## Attention and fusion checks

The incumbent already performs grouped FP8 cache reads, split-KV reduction,
fused QK norm/RoPE/cache write and output norm/gating. Both direct probability
conversion and skipped refinement-plane suggestions were rejected with explicit
precision/ROI reasoning in `ATTENTION_ARXIV_RESET_20261104.md`; the alternative
producer/consumer rewrite also regressed badly. No new precision shortcut is
introduced to reach the required33.76% all-attention improvement.

The reviewed megakernel papers motivate dependency and resource measurements,
not indiscriminate fusion. Existing PDL already overlaps eligible prologues, and
the small paths already contain UV and pooler/router fusion. The visible profile
contains too little independently removable boundary work to support5% from a
boundary-only rewrite. A tile-ready pipeline across projections would need to
show both dependency-ready work and admitted resources; neither is established
by whole-kernel profiler durations.

The concrete next decision is therefore to finish the already reviewed broad
small-MoE NW4 experiment and preserve these constraints. A new dense/attention
proposal should return only after it identifies a real main-path bottleneck and
an exact resource/lifetime design with enough affected scope. No candidate,
GPU run, submission or public promotion was performed by this feasibility review.
