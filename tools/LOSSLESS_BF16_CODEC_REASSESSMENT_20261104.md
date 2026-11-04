# Lossless BF16 codec reassessment

November 4, 2026. Exact current decoder:
`97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
**NO GO for another codec implementation on the retained evidence.** This is
not an optimality claim. No candidate, GPU operation, official poll, or paper
search accompanied this source/distribution/profile audit.

## What the current kernel already does

`submission.py:12254` packs each512-weight B-fragment unit into768 bytes:512
raw low bytes and256 bytes of four-bit codes. A low byte preserves all seven
mantissa bits and the exponent's least-significant bit. The code preserves sign
plus a three-bit offset into eight consecutive high-exponent values. A matrix
base is chosen from the actual immutable weights. Out-of-window values retain
exact FP32 corrections; unsuitable matrices take the existing fallback.

`decode8` at9360 reconstructs eight values into four BF16x2 operand registers.
It already uses packed bit operations, two replicated-base additions and four
PRMTs. The SK2 consumer at10525 loads packed weights and activation fragments,
returns its ring credit once loads are complete, and decodes directly before
MMA. There is no temporary global decompressed-weight buffer to eliminate.
Storing predecoded16-bit values would increase payload and repeats an existing
uncompressed alternative, not supply a new compression benefit.

The five prepare/packing/table helpers are AST-identical to80a4; both embedded
`bfp_CUDA_SRC` and `bfp_SK2_SRC` are byte-identical. KPERM changed expert work,
not this dense codec. Thus prior codec and escape-path findings still apply.

## Measured escape distributions and actual byte scope

The directory experiment `4b9f33ffa6bc4eed` recorded actual immutable escape
counts. These are one80a4-lineage preparation's weights, **not the current97af
job's weights**; no full exponent histogram or unique escaped-unit count was
saved. The same unchanged codec makes these relevant observations, not a
guarantee for other weights.

| Matrix region | Weights | Escapes | Escape fraction | Padded table /12-bit payload |
| --- | ---: | ---: | ---: | ---: |
| o | 8,388,608 | 976 | 0.01163% | 0.1953% |
| pooler | 33,554,432 | 4,648 | 0.01385% | 0.1017–0.1302% |
| QKV | 18,874,368 | 2,515 | 0.01333% | 0.1302% |
| unpooler slice | 4,194,304 | 401 | 0.00956% | 0.1628% |
| full unpooler | 33,554,432 | 3,354 | 0.01000% | 0.0936–0.1058% |

The complete table is only about0.1–0.2% of payload. Index narrowing, delta
packing, or removing table padding cannot materially reduce whole weight
traffic. These byte counts do **not** bound correction latency: waits and
serialization can cost more than their bytes. However, the directory experiment
preserved fusion, activated its selected tables, and produced no useful gain;
global-input correction V10 regressed and lost some fusion. Neither result
supports another cosmetic table or sparse-global variant.

## Narrower codes and decoder layout

An11-bit code has704 bytes per512 weights, only8.33% below768 before overhead.
An additional1.5625% escape rate consumes that entire saving with the current
eight-byte correction entries. Twelve-bit escape counts do not determine how
many values fit a four-value exponent window; the needed histogram is absent.
Per-unit bases or variable modes add metadata and decoding/address work.

The present sixteen weights per lane use sixteen low bytes plus an aligned
eight-byte nibble load. Eleven-bit storage needs six code bytes per lane.
Keeping eight-byte lane padding forfeits the saving; actually packing six
bytes requires a different load/bit-plane layout or unpack step. It is not a
free substitution of a narrower integer. Alternative LUT/sign placements also
replace the already simple affine high-byte reconstruction with extraction,
selection or lookup work; no inspected source exposes a redundant whole pass
or repeated dequantization to remove.

Using current97af's actual twelve profiles, benchmark normalized times and
46/16 sliding/full weights gives these **conditional proportional-time
scenarios**, deliberately granting the byte reduction to entire small-dense
kernels, including unaffected work:

| Hypothetical free change across all small-T dense kernels | Whole-decoder latency reduction |
| --- | ---: |
| 12→11 bits, no extra escape/metadata/decode cost | 2.191% |
| 12→10 bits, no extra cost and all weights qualifying | 4.411% |
| Remove0.2% of the entire small-dense kernel time | 0.0523% |

Five percent whole-decoder improvement requires18.8566% less small-dense time
under the same model. The older11-bit note's roughly2.2% generous scenario
remains consistent; it should not be restated as a proven less-than1% bound.
Actual new escape/unpack costs reduce its appeal. These are not hard latency
bounds: PDL rows include waits/overlap, and resource changes can have nonlinear
effects. No such beneficial resource effect is currently demonstrated.

## Distinct mixed-format possibility examined, not proposed

A mathematically distinct layout could store each8×64 tile as12-bit when it has
no escapes and as full BF16 when it has any, eliminating the separate correction
path. This avoids sparse global activation reads, but **increases weight bytes**.
Each expanded tile adds256 bytes. Recorded total escape counts imply at most
4.90–7.09% of tiles would expand if every escape landed in a different tile:
up to1.63–2.36% more weight payload before mode/offset metadata. Actual unique
tile counts were not retained. At most that4.90–7.09% subset skips the main
12-bit decoder; the rest still decodes normally.

Removing correction warps/planes could change scheduling, but the high-impact
QKV and fused-pooler configurations still have roughly167–219KiB dynamic shared
footprints before such changes. Removing threads alone does not establish more
resident CTAs; QKV also retains its96 output CTAs if row geometry is unchanged.
Variable768/1024-byte tile sizes require new stage offsets, TMA transaction sizes,
shared addresses and uniform decode branches. Retaining worst-case fixed slots
instead loses the compression advantage. Fusion and exact FP32 behavior would
need separate source/numerical proofs.

This is a distinct format idea, but there is no measured correction critical
path, dense compiler/counter attribution, or exact resource design supporting
the required broad18.9% reduction. It is not grounds to implement a replacement
for the failed global/directory experiments. Preserve the incumbent and the
NO GO decision rather than manufacture a new variant from the same uncertainty.

All arithmetic, exact hashes, measured table records and updated profile
scenarios are saved in
`experiments/iterations/20261104-lossless-bf16-codec-reassessment.json`.
Prior evidence: `POST_NCU_TRAFFIC_REASSESSMENT.md`,
`DENSE_ESCAPE_GLOBAL_V10_DIAGNOSIS.md`, `DENSE_STAGE_DIRECTORY_RESULT_AUDIT.md`,
and `NEXT_DENSE_ATTENTION_GATE_20261104.md`.
