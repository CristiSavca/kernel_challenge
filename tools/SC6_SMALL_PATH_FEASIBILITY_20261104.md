# Six-bit scales for selected small-MoE paths

2026-11-04. **Source-feasible, but defer implementation after the WGS SC6 result;
no skinny speedup established.** The current helper pipeline, WSB tail handling, FINPF,
MMA terms/order and precision can remain unchanged. This is a read-only follow-up
to WGS SC6, not an implementation or authorization to modify it. Root's terminal
WGS analysis reports a correctness pass at2.384843, but failure of its predeclared
performance screen: affected GM speed+0.531% against public and-0.384% against
private, with only M32 positive against both. Actual SC6 activation and valid
96-register/zero-local/one-CTA resources do not supply a performance win. This
weakens the case for spending another check on the same small byte margin with
additional skinny unpack instructions; no automatic extension is recommended.

Exact parent: `native_wgs_kperm3_v9.py`, SHA
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
Donor: [keell0renz decoder890809c](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_decode_layer.py),
SHA `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
Relevant donor code is `_sc6_unit` at7075 and `sc_load` at4389; retain Apache
LICENSE/NOTICE if adapted. Presence of these helpers is not proof that the donor
measured or enabled skinny SC6 in its published configuration.

## Exact selected scope

| Parent path | M | Layout/decode | Distinguishing flags retained |
| --- | --- | --- | --- |
| TINY |1..4|`use_mul=False`, `_POS`, MULDEC0|L2H0, WSBLK2, FINPF1|
| TINY2 |5..8|`use_mul=True`, `_POS_MUL`, MULDEC1|CPF0, WSBLK1, FINPF1|
| MID |9..16|`use_mul=True`, `_POS_MUL`, MULDEC1|SPLIT1, CPF0, WSBLK2, FINPF1, SKSC2/FTSC2 flags|

All three use NW8, TINY1, FAST1=2, NACC4, WS1, WSB1 and WSBL2. Their active
compute function is `skinny_ring_ws`, with eight K64 chunks per warp and four
MMA calls per chunk. Its scale path calls `SCDK` separately for the two rows;
the SKSC2 paired-scale branch present in the other `skinny_ring` functions is
**not used by this WS loop**, even when MID defines SKSC2. Preserve those flags
and all inherited branches rather than attempting a second optimization.

Standalone TINY differs from decoder97af's TINY, which uses MULDEC1 and omits
FINPF. A later decoder port must preserve those decoder flags; neither timing
nor registers transfer automatically. Saved source-equivalent local standalone
resources were149/144/148 registers for TINY/TINY2/MID with zero local bytes.
The separate decoder diagnostic observed144/144/148 and one-CTA capacity with
214784 dynamic bytes. These are historical observations, not new SC6 resources.

## Exact packing and consumer work

Current unit:32768 code bytes +4096 scale bytes, representing one16-row output
block and fullK4096. Its scales are `[warp8][lane32][scale16]`, where the last
axis is `[Kchunk8][row-half2]`. For each byte let `v=b-0x40` after qualifying
actual W13/W2 local-bank scale bytes in`0x40..0x7e` and the existing PRMT domain.

Donor packing makes three1024-byte planes. For plane`w=0..2` and byte`n=0..3`:

```
plane[w][warp][lane][n] = (v[4*w+n]<<2) | ((v[12+n]>>(2*w))&3)
```

Each lane loads one uint32 from each plane at scale-base
`warp*128+lane*4`, plus offsets0/1024/2048. Reconstruct the original sixteen
positions, shifted left by two:

```
r.x = w0 & 0xfcfcfcfc
r.y = w1 & 0xfcfcfcfc
r.z = w2 & 0xfcfcfcfc
r.w = ((w0<<2)&0x0c0c0c0c) | ((w1<<4)&0x30303030) | ((w2<<6)&0xc0c0c0c0)
```

The first twelve values are the plane high six bits; each of the last four
combines three nonoverlapping two-bit pieces. Each reconstructed byte is exactly
`4*v`. In the qualified WS branch use multiplier`4+p.zero` and offset`0x60006000`
in `SCDK`, giving each BF16 half`16*v+0x6000`, identical to the parent's
`16*(0x40+v)+0x5c00`. Keep ordinary load/multiplier/offset for the fallback.
No FP4 code or `_POS` mapping changes, and the existing DQ/MULDEC/MMA sequence
can be byte-identical after scale construction in both layouts.

## Smallest integration avoids layout collateral

Prefer retaining **all shared slot strides and `UNIT_BYTES=36864`**, rather than
copying the donor's global reduction of `UNIT_BYTES` to35840. A separate skinny
payload constant controls only the compressed global stride, transferred bytes
and expected bytes. Each35840-byte payload lands at `ring+slot*36864`; the last
1024 bytes are unused. The final scale load ends at byte35839. All source and
destination unit offsets remain16-byte aligned; no TMA operation is added.

Two actual producer locations need that payload distinction: `ring_issue`
(parent1039, used by `ws_issue` and helper/refill), and FAST1's direct initial
copies (parent3450 area). Change source unit indexing and both bulk-copy and
`expect_tx` byte counts together; retain destination slot strides, barrier
participants/parities, NS3, FAST1 preissue count and WSB refills. If retaining
the non-TMA compilation branch, its copy-loop byte bound must also match the
payload. No consumer may read the unused padding as ordinary scales.

Only prepared entries10/11 (`wAu/wBu`, `_unit_order`) need compressed scales.
Entries0..9 and WGS12/13 remain the exact parent. The decoder helper/tail,
FP32 partial buffers, finalizer scratch aliases, job split count and output
math need no change. Keeping shared strides also preserves the inherited
`max(xa,xb)` allocation, including its otherwise oversized fat reservation;
removing that reservation would be an unrelated resource experiment.

**State/format compatibility is the principal host hazard.** Unlike WGS SC6,
small SC6 must qualify both prepared layouts, including `use_mul=False`; copying
the WGS predicate `use_mul and qualified` would silently leave M1 unchanged.
Store a separate state-owned `sc6_skinny` flag and add a unique compile define
only to the matching NW8/TINY/WS modules. Never use a mutable global mode.

The larger layout is shared by M8/M16 and MAIN. On exact247888, qualified PRMT
selects MAIN WGS, whose device path does not read `wAu/wBu`; that source fact
allows compressed10/11 alongside ordinary WGS12/13. If the selected MAIN or
another active non-WGS fallback can consume10/11, either it must understand the
same encoding or the shared entries must remain ordinary. In particular,
non-PRMT/general-scale fallback keeps original bytes and flags. Rejecting a bank
for one out-of-range byte must not leave a partially packed expert bank. Alternate
eligible/ineligible prepares and both code-position layouts in any later test.

## Cost and decision

Each unit saves1024/36864 = **2.7778% of packed weight requests**. A complete
skinny job has512 A units +256 B units, so requests fall27→26.25 MiB, saving
0.75 MiB/job. MID's existing split-to-eight-token jobs remain unchanged, including
any repeated expert requests. Cache reuse means these are not necessarily
removable DRAM bytes. All activations, unit counts, MMA count, synchronization
counts and output traffic stay the same.

Scale loads change from one source uint4 expression to three scalar uint32 loads
per lane/unit, followed by three masks and shift/mask/OR reconstruction. Actual
SASS may combine those operations. Loaded scale bytes fall16→12 per lane, but
instruction count can rise; do not infer cost from bytes alone. The resulting
four scale registers have the old shape, with extra intermediate liveness and
instructions before32 MMA calls per warp. No extra cross-thread reduction or
shared decoded-scale table is needed. Registers/spills and performance remain
unknown. No occupancy improvement is expected from the intentionally unchanged
shared allocation, and none is claimed from six-bit packing.

The different small-path arithmetic means the WGS result does not prove this
extension would lose, but there is currently no distinct measured cost basis
that makes it the preferred next experiment. Retain the feasibility and defer.
Do not convert WGS's noisy aggregate increase into permission to stack formats.
Required proof is limited to actual packing/scale operands, the two producer
byte paths, format-aware state dispatch and unchanged helper/finalizer code.
Reuse the parent tests; do not build another asynchronous model or sweep NS/NW.
Measure M1/M8/M16 against both exact parent receipts, retaining M32/128/256 as
unchanged controls and raw/normalized results separately. No universal large
gain hurdle is justified. This note makes no implementation or GPU approval.
