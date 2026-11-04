# Fixed SC6 experiment on the KPERM incumbent

2026-11-04. One standalone encoding change; no geometry, queue-depth, finalizer,
cost-table or decoder changes. Root authorized CPU implementation after mechanism
review. Root alone decides and performs any GPU check. This plan supersedes the
earlier natural-K SC6 deferral: KPERM is now validated and decodes one scale record
per lane per stage. No speed prediction follows from the byte reduction.

Parent: `muse_moe_w4a16/candidates/native_wgs_kperm3_v9.py`, SHA256
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
Public comparator `d24dd50196cb4158`; private `bcb761dfca264db0`.
Licensed mechanism source: [keell0renz's pinned decoder](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_decode_layer.py),
SHA256 `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
Retain Apache-2.0 LICENSE/NOTICE and existing attributions. Extract only SC6 WGS
packing/decoding, not its global mode flag, skinny SC6, MFIN, BF16 ybuf,
cost coefficients, producer barriers, timers or prepare tuning.

## State and exact format

For each actual local expert bank, qualify **both** W13 and W2 scale tensors:
the existing PRMT predicate must hold and every byte must lie in `0x40..0x7e`.
The upper endpoint excludes E4M3 NaN. Signed, subnormal, smaller finite and any
other ineligible byte select the original encoding/dispatch. No seed, shape,
expected distribution or global mutable `_MA_SC6W` determines qualification.

Store the result in that prepared layout. Apply `OKC_SC6W=1` only to its selected
WGS MAIN extension key, never to generic `sfx` consumed by tiny/MID modules.
Preserve the standalone tiny/use_mul=False layout and its original copies;
enable the new packing only in the larger layout when it will select WGS.
Repeated prepares of eligible/ineligible banks must coexist and alternate safely.
If PRMT is valid but SC6 is not, use the parent's ordinary eight-bit KPERM WGS.
If PRMT itself is invalid, preserve the existing general-scale fallback.

Only WGS entries 12/13 may change. Keep all code nibbles, KPERM order, gate/up
row permutation, W2 rows, global FP32 scales and BF16 post-norm weights exact.
The code portion of each piece remains 8192 bytes. For its 128 lanes, let
`v[t,n] = original_scale_byte[t,n] - 0x40`, `n=0..7`. Selected values are0..62;
the six-bit packing algebra can additionally be checked over0..63.

```
piece[0..5] = [v6&3, (v6>>2)&3, (v6>>4)&3,
               v7&3, (v7>>2)&3, (v7>>4)&3]
W[t].byte[n] = (v[n]<<2) | piece[n]       n=0..3
H[t].byte[n] = (v[4+n]<<2) | piece[4+n]   n=0..1
```

Store plane W as128 little-endian uint32 values (512 bytes), followed by plane H
as128 uint16 values (256 bytes). The source record's eight bytes become six;
the GPU uses one aligned32-bit and one aligned16-bit load per lane per stage.

## Addresses, TMA sizes and unchanged shared allocation

| Quantity | Parent | SC6 |
| --- | ---: | ---: |
| Code/scale bytes per WG piece | 8192/1024 | 8192/768 |
| Contiguous piece stride | 9216 | 8960 |
| Global stage, four WG pieces | 36864 | 35840 |
| Shared ring slot stride | 36864 | 36864 |
| Weight and activation slots | 3/3 | 3/3 |
| Dynamic shared bytes | 231168 | 231168 |

For flat expert/unit/stage index `g`, the producer's source is
`base + g*35840`; destination is `ring + ws*36864`. Issue one bulk copy of35840
bytes and add **35840**, not36864, to its existing joined barrier's expected
transaction count. The activation producer still contributes its exact original
byte count to that same barrier. All participant counts, phase bits and release
points remain unchanged. Both source/destination are16-byte aligned: stage35840,
slot36864 and piece8960 are multiples of16 (also128); parent allocated bases
already satisfy bulk-copy alignment.

Consumer WG `wg` uses `pc = ring + ws*36864 + wg*8960`. Its code addressing is
unchanged relative to `pc`. W is at `pc+8192+4*t`; H at `pc+8704+2*t`, where
`t=wp*32+lane`. Last H byte is8959. Four compact pieces cover `[0,35840)` of the
slot; its final1024 bytes are unused padding. **Do not leave9216 gaps between
compressed pieces, and do not move the activation-ring base.** `WGS_PIECE`
changes conditionally; `UNIT_BYTES` and shared slot/X-ring calculations do not.
Use a separate WGS global/copy-byte constant to avoid changing skinny formats.

## Exact once-per-stage decoder

Use the donor's uint32 modular arithmetic (not Python unlimited-int semantics):

```
X  = W & 0xfcfcfcfc;  Hm = H & 0xfcfc
p6 = (W & 0x00030303) * 0x04104000
p7 = ((H & 0x0303) | (W & 0x03000000)) * 0x10400004
u0 = byte_perm(X,  0,  0x4140)
u1 = byte_perm(X,  0,  0x4342)
u2 = byte_perm(Hm, 0,  0x4140)
u3 = byte_perm(p6, p7, 0x0703)
scale_pairs[j] = u[j] * (4 + p.zero) + 0x60006000
```

The selected bytes in each `u` equal `4*v_even` and `4*v_odd`. With runtime
`p.zero==0`, each half is `16*v+0x6000`, exactly the parent's
`16*(0x40+v)+0x5c00`. Halfwords never overflow for eligible values; p6/p7's
intermediate uint32 wrap is intentional. Preserve the opaque `p.zero` mechanism.
The ordinary branch retains multiplier16 and offset`0x5c005c00`; using those
constants with shifted SC6 planes is wrong. Construct four pairs once before
the four load groups and retain the existing HMUL2/DQ/MMA operand sequence.

No FP4 code, E4M3 byte value, BF16 decoded operand, accumulation order or output
precision changes. Source equality of the arithmetic schedule plus exhaustive
scale reconstruction supports that claim; official checks still establish GPU
correctness. WGMMA instructions, AF banks, GS2/wait1, NTT1 parity, FCI3, all
epilogues, routing, gathers, job/range costs, scratch, final normalization and PDL
remain the parent implementation.

## Bounded verification and measurement

Author owns `native_wgs_sc6_v16.py`, its builder and focused CPU suite. Reuse the
parent proof and compare unchanged functions/preprocessed non-WGS branches.
Do not build a new generic lifetime simulator: no ordering/count participant
changes are proposed. The delta needs these specific checks:

- Exercise actual pack/unpack snippets for every eligible63×63 pair and all
  eight scale positions, independent high-bit/background patterns, both p6/p7
  cross-bit reconstructions and uint32 wrap. Include all64 encodable v6/v7 values
  as an algebraic superset, while confirming v63 is never selected by qualification.
- Recover original scale bytes/parent BF16 pairs at every lane, WG, stage and
  W13/W2 row coordinate; codes remain byte-identical. Independently check compact
  global offsets, padded shared slots, last-byte bounds and all ring slot phases.
- Negative controls must catch multiplier16/oldoffset in SC6, an incorrect
  cross-bit mask,9216 compressed-piece stride,36864 copy expectation and a mode
  cached globally across eligible/ineligible prepares. Test ineligible endpoint,
  sign/subnormal and mixed-bank rejection; tiny/MID/general fallback stays exact.
- Record actual compiled attributes and valid occupancy for both BIG entries at
 576 threads/231168 dynamic bytes. Local bytes and serialization warnings are
  reported facts, not automatic performance rejection; failed API/admission is a
  launch failure. Keep diagnostics bounded and out of device math/timing logic.

Packed weight requests fall27→26.25 MiB per complete job, saving0.75 MiB/job.
No claim of equally reduced DRAM service: repeated requests can hit cache.
Activation traffic, stage/TMA operation counts and shared allocation are unchanged.
The additional shift/mask/IMAD work and second scale-load instruction may outweigh
the saving or change compiler allocation. The NCU DRAM percentage is not a gain
forecast. Inspect M32/128/256 as affected; M1/8/16 remain controls. Preparation
adds a qualifying scan and temporary pack arithmetic but retains only the selected
WGS copies, with no new timed autotuning or duplicate full-bank allocation.

Root's fixed small-gain screen against **both** exact247888 comparator receipts:
affected geometric-mean relative speedup at least1.01; at least two of three
affected cases positive against both; each affected normalized latency ratio at
most1.02; unchanged-control GM normalized latency ratio in `[0.98,1.02]`. Report raw/harness
times separately and retain all official checks/evaluator/source identities.
This is a predeclared consistency screen, not a significance test. Official CI
overlap does not establish equality. A passing standalone result permits review
of isolated decoder integration, not an automatic leaderboard claim.

No new GPU job, source sweep, or implementation beyond the root-assigned author
work is authorized by this document.
