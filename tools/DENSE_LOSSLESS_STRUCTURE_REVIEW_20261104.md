# Dense lossless structure review — 2026-11-04

Read-only audit of exact decoder `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
No candidate, GPU operation, provider call, or change to production source.

**Finding:** there is no newly identified redundant decompression pass or broad
byte-saving replacement ready to implement. A per-unit exponent dictionary is a
distinct way to remove correction execution, but increases the main decoding and
metadata work; the independent critic owns its pending cardinality/codec study.
The public standalone's QKV row balancing is a smaller, concrete alternative
that has not appeared in the inspected decoder history. Neither is a measured
gain on this decoder.

## Current work and bytes

SK2 stores each 8-row × 64-K unit as 512 low bytes plus 256 sign/exponent-code
bytes. The low byte retains all seven mantissa bits and the low exponent bit;
the nibble retains sign and an offset within eight exponent-high values. An
immutable matrix window and exact FP32 corrections cover escapes; unsuitable
weights retain the original fallback. `bfp_sk2_compress` is at12254,
`bfp_sk2_stream` at12292, and `bfp_sk2_table` at12307.

| Matrix region | Units of512 weights | Main packed payload |
| --- | ---: | ---: |
| QKV,2304×8192 | 36,864 |27MiB |
| o,8192×1024 |16,384 |12MiB |
| pooler,4096×8192 |65,536 |48MiB |
| unpooler,8192×4096 |65,536 |48MiB |
| unpooler slice,1024×4096 |8,192 |6MiB |

These are logical prepared-weight bytes, not measured DRAM requests. The fused
router additionally retains its separate FP32-as-two-BF16 stream. No reduction
of that precision or its work is proposed.

The SK2 consumer at10525 loads each unit's24 bytes per lane with LDS128+LDS64,
then decodes directly to eight BF16x2 operand registers. Four activation
fragments per MT are reused across NT output tiles. There is no global decoded
weight intermediate and no duplicate decode across MT. `decode8` at9360 has,
at source level, two shifts, two masks, two LOP3 selections, two additions and
four output PRMTs:12 integer operations per eight weights, or24 warp operations
per512-weight unit. This is not an actual SASS count or a critical-path bound.
Each unit then participates in four MMA instructions per MT. Decoder weights
already have the tensor operand layout; generic fragment-layout advice does not
identify a missing conversion pass.

UB2 (`UB2_CUDA_SRC`:13118; `_ub2_pack`:13492) uses the same12-bit information
in another direct MMA layout. Each thread preloads48 packed words, constructs
64 BF16x2 operand words, and does this before the MoE dependency wait. Compiler
liveness can reuse packed registers, so48+64 is not an actual allocation.
UB2's remaining global/shared traffic performs input delivery, exact correction,
cluster partial exchange and the fused boundary. Moving its decode after the
wait would not remove work. Its table already stores original BF16 escape bits
and reconstructs deltas; replacing SK2's prepared FP32 deltas with that form
would add runtime arithmetic while leaving its eight-byte table entries intact.

Existing evidence closes simple narrower codes, table padding/index changes,
global-input escapes, stage directories and R16 geometry. The recorded escape
tables are only approximately0.1–0.2% of main payload. The earlier11-bit
proportional-time scenario was2.191% whole-decoder latency reduction before
extra escapes/unpacking, not a guaranteed less-than1% limit. See
`LOSSLESS_BF16_CODEC_REASSESSMENT_20261104.md` and the terminal global/directory
audits. No measured dense instruction attribution establishes a profitable
register/decode reschedule; the recent compiled SASS evidence is MoE, not SK2.

## Distinct dictionary mechanism: feasibility constraints, not source GO

An eight-entry e71 dictionary per512-weight unit can represent disconnected
exponent values with the same sign+three-bit-index nibble. It may eliminate
correction warps even when a matrix-wide consecutive window has escapes.
Matrix escape counts do not establish per-unit distinct cardinalities. The
critic is independently evaluating a bounded distribution proxy and exact
adversarial reconstruction; those proxies are not the actual judged weights.

A concrete layout preserving aligned768-byte primary units can append16-byte
headers per unit at each stage's tail: two dictionary words, raw-sidecar index,
and explicit mode. One contiguous bulk transfer can cover primary+headers after
the stream is repacked. A raw fallback still incurs the768-byte dummy primary
plus1024 bytes of original BF16 fragments from two consumer LDG128 operations
per lane. Average logical bytes become `784 + 1024*f` per unit, where f is the
raw-unit fraction. Even f=0 is2.083% above the primary, before subtracting the
small old table; all-raw costs1808 bytes versus1024 uncompressed. Arbitrary
weights therefore need a conservative whole-matrix fallback as well as exact
unit decoding. This is a correction-execution hypothesis, not byte compression.

One exact selector formulation stores dictionary bytes as `e71|0x80`, uses
contiguous sign/index nibbles, and for four values computes:
`h=prmt(D0,D1,sel&0x7777); s=prmt(0x01010101,0x01010101,sel); h^=s<<7`.
The sign selector bit makes PRMT replicate the constant byte's zero sign bit;
otherwise it returns01. Thus only positive outputs toggle bit7. Four ordinary
PRMTs interleave the original low bytes. This is approximately15 source integer
operations per decode8 instead of12, plus header loading and two live dictionary
words; it requires an independent bit/byte proof and compiled resource review.
The raw-mode branch is uniform within a warp's unit, but can vary across NT.

Keeping the selected geometries and staging16-byte headers increases SST by
1024 bytes. Removing old correction planes offsets that only slightly on the
largest important paths: QKV TOKR8/16/32 net ring-minus-correction changes are
−256/−1280/−8704 bytes; fused pooler TOKR8/16 are−2304/−3328 bytes. These figures
omit separately removed table/barrier bytes and do not establish another
resident CTA. They use the actual `NESC=min(NSTAGE,4)`, not an inactive NESC3
macro. Removing correction consumers requires changing every empty-barrier
arrival count, cbar/plane lifetime and router-stage handback consistently.
Exact original weights in MMA also change FP32 association relative to the old
approximate-weight MMA plus separately accumulated delta correction; bitwise
parent GEMM equality must not be claimed. No precision may be reduced.

## Smaller ready-to-plan alternative: QKV escape balancing

Standalone exact `27a0afe97d0a9b8cd11a58f243b44f2c375d84cec5cb7b742f451da45a239e30`
has `SK2_BALANCE_LAYERS={"qkv_gate"}` at3439, a stable descending escape-count
sort and snake deal at3442, appended row maps at3464, and output remapping at1515.
It comes from the pinned licensed vjeux source. A bounded search of decoder
candidates, tools and saved iteration notes found no decoder version of this
mechanism; the independent critic recalls no such experiment. That is a scoped
history finding. The donor's approximately0.5us comment is not our measurement.

A fixed optional port would retain R24/NT3/NWR1/WSK4/UPW2 and all ring settings,
deal576 intact four-row groups among96 QKV CTAs, repack only immutable QKV
weights, append each CTA's24 original row IDs, and remap the final four-value
store in `vstore` at10923. Keep existing FP32 delta entries and reductions;
do not also import the donor's original-bit correction implementation. Restrict
the output-map code to the QKV compile-time geometry, preserving every
pooler/router/UV/BND path. QKV emits ordinary output used by attention, so the
original row order must be restored before that consumer observes it.

This changes no27MiB weight payload, decode count, activation traffic, MMA count,
or CTA count. The map costs96 bytes per CTA/9KiB total and one shared row-ID load
plus address selection per four-output group (576×M groups). The old observed
QKV escape mean/max was26.20/46 per CTA; those are earlier80a4-lineage weights,
not a current per-four-row histogram. A perfectly balanced table could recover
some map bytes by reducing padded pmax, but neither that balance nor a shorter
critical tail is proven. Total counts also do not balance each escape warp's
stage-modulo workload. Resource and scatter costs can erase a small saving.

Granting the donor comment's0.5us profile-row reduction to both T8/T16 layers
gives a conditional0.130% whole-decoder latency reduction using saved97af
weights/times; granting it to all four small-T loads gives0.275%. These are
scope scenarios with PDL overlap limitations, not forecasts or significance.
If root chooses this small experiment, proof should preserve each original
row's ordered K contributions and output bytes, demonstrate the full fallback,
and record actual pmax/resources before the single private timing gate. It is
not a substitute for the pending broader dictionary feasibility decision.
