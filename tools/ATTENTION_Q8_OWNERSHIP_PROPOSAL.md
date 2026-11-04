# Q8 attention ownership proposal, November 4

Status: **isolated prototype implemented; independent static GO, no GPU run.**
Candidate `muse_decode_layer/candidates/native_q8_role_pipeline_v10.py` is frozen
at `6eb318df58c3ef5576c4ea456b4b494e1ef4548090fa7707c84e247f960c2222`.
The simple extra-buffer approaches below fail a resource precheck. The alternative
changes warpgroup ownership and reuses dead storage. It is a bounded speculative
prototype, not a measured 10–20% gain forecast.

Parent is exact decoder `80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`,
official job `4b216098993e4112`. All line references below are to its preserved
source under `experiments/candidates/muse_decode_layer/`. V5's successful
numerics and null performance result do not support another QK-instruction-count
optimization. No active submission, runner, baseline or reference changed.

## Evidence and conditional ceiling

| Full case | Standalone raw attention us | Decoder attention profile us | Nominal 260-byte/key floor us | Standalone nominal headroom |
| --- | ---: | ---: | ---: | ---: |
| B16 decode | 86.384 | 90.0 | 69.333 | 19.7% |
| B32 decode | 165.776 | 169.1 | 138.667 | 16.4% |
| B16 verify | 100.304 | 105.2 | 69.333 | 30.9% |
| B32 verify | 182.976 | 186.1 | 138.667 | 24.2% |

Standalone raw means `median_raw_ns - replay_overhead.overhead_ns`, from
`87a9bddc...8f9e5971f95841a0.json`, not the normalized `median_ns`. Decoder rows
come from the raw profiler report, which can include PDL waiting. The floor is
`B * 80000 * (128 K + 128 V + 4 page-index bytes) / 4.8 TB/s`; actual saved
sequence-length sums are unavailable, so this is conditional on nominal lengths.
The 74K–86K range gives 64.133–74.533 us at B16 and 128.267–149.067 us at B32.
These bounds omit other traffic and overhead and establish no removable phase.

Q8 is the sensible target: a 10% raw reduction means 90.274/164.678 us, and 20%
means 80.243/146.381 us. At nominal lengths the latter B32 target already needs
4.547 TB/s for cache and page bytes alone (94.7% of specified peak). Treat 20%
as a stretch. Improving only these two full-verify cases by 20% contributes
roughly 0.8% whole-decoder gain under the recorded weighted profiler shares;
this cannot independently close the current decoder gap.

## What is already present, and what does not justify a new run

- Each K/V row is loaded once and reused across all eight grouped query heads
  and all eight query tokens. There is no duplicate HBM stream to remove.
- Q8 already has FP8 QK/PV, lazy maxima, packed-P range checks, loop unroll two,
  four load stages, independent head warpgroups and vectorized merge atomics.
- The source contains disabled TMA-page, producer-transpose, one-CTA/two-stream,
  continuous-segment and polynomial-exp experiments. Comments report
  several losses; those comments are historical evidence, not reproduced tests.
  Merely enabling their macros is not a new structural hypothesis.
- Exact decoder compiler overrides matter: `_att7` (5180) sets `QDIRECT=0`,
  `PK2=0`, `W0ENABLE=1`. Thus per-half P packing and uneven two-CTA partitioning
  are active, despite different native macro defaults. The new candidate keeps
  these flags and the existing W0 mapping. Do not call this the active87a9
  standalone implementation, which uses different compile settings.
- `SEGK8=600` already compensates request-crossing CTAs (lines 517–518,
  4581–4583). Nominal B16/B32 shares are approximately 4.8K/9.7K keys versus
  minimum request length 74K: at most one crossing per CTA, at most 15/31
  crossings among 264 CTAs. The historical approximately 7-us crossing cost
  would be only 6.7%/3.8% of decoder Q8 attention if wholly exposed and removed.
  **That comment is not a measured upper bound**, and sparse crossings do not
  by themselves bound the critical CTA. They do argue against assuming a
  10–20% gain from removing segment drains alone.

## Resource no-go for simply adding buffers

For `L=8, FLAT=true`, the parent has 384 threads: 128 load threads at 32
registers and 256 consumers at 104, or 30,720 allocated registers per CTA.
The initial pool is 384 × 80; two such CTAs use 61,440 registers. One extra
32-FP32-score tile per consumer raises demand to 38,912 registers per CTA,
or 77,824 for two CTAs, beyond Hopper's 65,536-register SM.

Manual C++ layout using the exact fields/alignment in `Smem` (1457–1509) gives
113,664 bytes; launch code adds 1,024 bytes of alignment slack (4921), hence
114,688 bytes per CTA. Two use 229,376 of 233,472 shared bytes, leaving only
2,048 bytes per CTA. Even one extra 8-KiB P tile loses two-CTA residency unless
storage lifetimes change. This is a static calculation; ptxas resources and
the CUDA occupancy API must confirm any compiled alternative.

[NVIDIA's Hopper guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy)
documents the register/shared-memory limits. The
[FlashAttention-3 paper](https://arxiv.org/html/2407.08608v2#S3.SS2)
explicitly identifies extra score registers as the cost of intra-warpgroup
two-stage overlap. Its large-attention speedups are not forecasts for this
short-query paged workload.

## Proposed rewrite: separate QK/softmax and PV ownership

Apply initially only to the existing SM90 full, multi-request, Q8 flat path.
Keep the current 264-CTA partition, four K/V slots, PDL protocol, query/cache
normalization, precise FP8 conversions, every key and all fallbacks.

| Warpgroup | Work after the common prologue | Register budget |
| --- | --- | ---: |
| 0 | QK, masks, lazy-max/packed-P checks, V transpose, FP8 P publication; head 0 then head 1 | 96 |
| 1 | PV and its FP32 output/row-sum accumulators for both heads | 112 |
| 2 | Existing page gather and K/V copies | 32 |

The sum `(96 + 112 + 32) * 128 = 30,720` fits the parent's initial register
pool exactly; the loader releases the same registers that the compute groups
acquire. PV's main live state is 2 × (32 output + 4 row-sum) = 72 FP32
registers/thread, leaving 40 for offsets, descriptors and control. QK no longer
owns output accumulators. These are explicit budgets to validate, not compiler
resource predictions. Preserve the original 256-thread common normalization
prologue before dividing its work between the new roles.

The important dependency is visible at 3618–3628: production `PVSPLIT=1`
waits for PV at every tile edge, avoiding previously reported compiler WGMMA
serialization. In the new design the PV worker can still finish its own tile
before advancing, while the independent QK worker computes the next tile.
This overlaps successive tiles without keeping two score tiles in one thread.

**Storage and ownership:**

1. After QK for head h completes, its K bytes are dead. Write FP8 P into that
   head's original 64-byte half of each 128-byte K row, retaining the SW128
   interleaving. The logical row becomes query row; the logical column is the
   existing permuted key index used by PV. Head-0 writes must never overlap
   head-1 K, which is still live until its QK completes. An SS PV A descriptor
   reads this P layout; V retains the existing transpose and key permutation.
2. Keep the K/V slot occupied until PV has read P and completed. Per-head,
   per-K32-half, per-slot generation barriers publish P. Retain the existing
   256-arrival empty barrier:128 QK threads arrive after both heads are published,
   and128 PV threads arrive only after both heads' WGMMA operations complete.
3. Reuse the 8-KiB `kvst + gst` prologue region through an aligned lifetime
   union. Their raw K/V reads end before the tile loop; flat mode does not use
   the cluster gate staging. Per-half alpha arrays use4,096 bytes, final offsets
   use2,048, per-warp move flags use256, and P-ready/V-empty barriers use160:
   6,560 bytes total. The host layout proof confirms the actual aligned union
   and every L1/L8 flat/nonflat shared layout remain unchanged.
4. Preserve the two transposed V slots per head. QK waits for a V-slot-empty
   credit before overwriting it; PV returns that credit only after WGMMA has
   completed. K-ring depth four does not authorize overwriting V at depth two.
5. On a rare softmax-offset move, QK publishes exactly the incumbent's FP32
   rescale factor for that K32 half. A second-half move first waits for half0's
   PV, then rescales all32 output and four row-sum components before half1 PV.
   The last offset is handed over after half1; PK2=0 order remains exact.
   This includes underflow/empty masks and the same FP8 packing and rounding.
6. PV publishes both heads' partials through the existing FP32 reduction math,
   looping over heads with the original lane mapping. Both compute groups join
   the existing consumer barrier before global finalize. Segment transitions
   drain all PV reads before the union becomes raw K/V staging again.

There is no additional HBM load, and no work moves out of `kernel_fn`. Added
traffic is 8 KiB of P stores plus approximately 8 KiB of shared P reads per
64-key tile, plus metadata/barriers. QK/softmax-active warpgroups per SM fall
from four to two if two CTAs remain resident; that can reduce instruction and
MUFU latency hiding. Load warpgroups remain two per SM. This trade is distinct
from old F2, which reduced load warpgroups to one per SM. If the compiled
candidate needs one CTA/SM, do not use the old cooperative 264-CTA grid: a
132-CTA plan and safe finalize are required, and per-CTA work doubles while
load concurrency halves. Treat that as a separate design decision, not a
silent occupancy fallback.

## Decisive validation and kill criteria

Before GPU: reconstruct SS-A P fragments for all lanes, both heads and both
K32 halves; prove all 8,192 P bytes lie only in their head's dead K storage;
simulate four K slots/two V slots through short tails, ring wraps and multiple
segments; verify phase counts and prologue-union lifetimes. Preserve fallback
AST/native strings exactly and record the isolated candidate digest.

First compiled check: no spills, no ptxas WGMMA serialization warning, verified
shared size and actual two-CTA occupancy. If budgets fail, stop for a deliberate
one-CTA redesign instead of squeezing arbitrary parameters. Numerics must cover
fresh repeated calls, B2/4/16/32, page1/64, short and long tails, scale1/64,
outliers/coherent/mixed lengths, and every official decoder condition.

One same-GPU screen should use B16/B32 full Q8 at ~80K, with full Q1 and sliding
Q8 untouched controls, frozen dense/MoE choices and actual sequence-length sums.
Target at least 10% lower affected attention-kernel time in both cases; 20% is
the stretch. Whole-decoder time must improve consistently with attribution.
No ≥3% affected gain, any material regression or numerical failure rejects the
candidate without a parameter sweep. Promotion still needs the exact official
private check and useful whole-decoder evidence.

## Optional diagnostic only if it changes the prototype decision

Do not enable `OKC_PROF`: even together with `OKC_PROFPV8`, it disables PCHK,
PVSPLIT, PK2/LSYNC-related paths (3067–3072, 3354, 3461), changing what it measures.

A separate local source could insert sparse, per-CTA globaltimer stores around
stage-ready wait, QK wait and the production PV wait at 3619, plus loop-end and
finalize completion. Preserve every production predicate and native instruction
block; stripping explicitly delimited stamp inserts must reproduce the parent
native string byte-for-byte, and Python AST must match apart from debug-buffer
plumbing. Compare the instrumented source against the exact parent on the same
inputs first; reject attribution if overhead exceeds 2%, register/shared/spill
resources change, or output/routes differ. Never sum wait times over CTAs into
removable wall time.

Its single decision is whether cross-tile PV overlap deserves this ownership
rewrite: on the CTAs setting the completion tail, recurrent PV-drain latency
must be material (screening threshold ≥15% of their loop duration) while the
producer usually has the next tile ready. If PV wait is small or memory waits
dominate, stop this hypothesis. This diagnostic is optional: a directly bounded
prototype comparison is more decisive once static ownership proofs pass.

Independent critic contributed the consumed-K-slot reuse, rejected an
unmeasured10–20% segment-drain claim, reviewed the native code, and approved
final6eb318df for one bounded screen. The verifier checks8,192 P bytes,
109,312 randomized scheduling events with exact phases/counts,128 rare rescale
moves and256 fast halves, host C++ syntax and exact113,664-byte shared layout.
It does not compile CUDA/PTX or establish GPU numerical correctness/resources.
See `tools/ATTENTION_Q8_V10_SCREEN_HANDOFF.md` for the lightweight wrapper and
mandatory compiled-resource/log capture plan. No GPU/upload occurred here.
