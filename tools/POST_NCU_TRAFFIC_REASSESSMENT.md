# Post-NCU traffic reassessment, 2026-11-04

Decision: this pass found no adequately supported large decoder implementation
bet. Do not treat this as optimality or completion of the leaderboard goal.
No candidate, GPU job, or source modification resulted from the audit.

Parent decoder: `80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`.
Its official profile is `experiments/candidates/muse_decode_layer/profiles/`
`80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c.4b216098993e4112.json`.
Counter evidence is the separately recovered `experiments/iterations/`
`20261104-moe-hardware-profile-recovered.json`; it represents two controlled-cold
standalone samples, not decoder PDL or benchmark speedup.

## BF16 projection compression is already present

Exact parent lines 9186–9198 describe the existing 12-bit codec: raw low byte
(7 mantissa bits plus exponent bit 0), sign plus a 3-bit high-exponent offset,
with exact escape corrections. `bfp_pack_compressed` at 11926 and
`bfp_sk2_compress` at 12179 implement it; `bfp_sk2_table` at 12233 stores 8-byte
escape entries plus per-CTA headers/padding. The small-token fused UV/UB2 paths
already consume compressed weights (`_after_attention`, 14862–14974;
`_uv_pooler`, 15268; `_ub2_pack`, 13417). A new 25% BF16 traffic reduction would
double-count this existing mechanism.

A distinct 11-bit code could use the low byte, sign, and a 2-bit high-exponent
offset. A 512-weight SK2 unit falls from 768 to 704 bytes, before any new metadata:
only 8.33% below the current format. With the current 8-byte escape representation,
an *additional* escape rate of 1/64 = 1.5625% consumes the entire 64-byte saving,
before padding and execution. Per-block bases add header traffic; 3-bit codes
also require cross-byte extraction instead of the existing nibble decode.
An exact prepare-time suitability check and fallback are mandatory; normal
weight distributions do not guarantee a profitable encoding.

Deliberately generous traffic-only accounting scales the entire compressed dense
kernel time by 1/12, including unaffected router/fixed costs. Official profile
rows give the following weighted-load reductions:

| Load | Compressed-dense kernel share | Generous 1/12 reduction |
|---|---:|---:|
| decode 1 | 56.60% | 4.72% |
| verify 1 | 39.23% | 3.27% |
| decode 16 | 32.36% | 2.70% |
| decode 32 | 29.25% | 2.44% |

Across the six-load geometric mean, T1/T8 scope gives only 1.35% normalized
latency reduction; extending through T32 gives 2.20%. These are optimistic
proportional-traffic calculations, not hard latency bounds or forecasts.
The exact formula is `1 - product(1 - dense_share/12) ** (1/6)` over affected
loads, with each load's profile times weighted 46/16. Added decode/escape costs
make this a secondary margin rather than a major next bet.

A new SIMT GEMV would be a different implementation, but the current kernels
already stream each 12-bit weight once, use split-K and async staging, and fuse
the small-token boundaries. Old uncompressed GEMV measurements are not evidence
against this newer incumbent. There is no projection counter attribution showing
that replacing tensor-core MMA with scalar FMAs recovers a large exposed cost.

## Expert traffic reduction has little supported margin

Each local expert has 24 MiB of FP4 codes and 3 MiB of scale bytes. The unchanged
NVFP4 helper's `random_weights` (muse_moe_w4a16/nvfp4.py:125–153) generates uniform
code bytes. Generic lossless compression cannot promise a smaller representation
of arbitrary 4-bit codes. Scale bytes are only 11.11% of this traffic even if
their removal were free. A conditional exact 6-bit scale representation saves
only 2.78% overall weight bytes and adds unpack work; it must examine the actual
weights and retain a full-range fallback. The existing PRMT predicate accepts
positive normal E4M3 codes 0x08..0x7E, which itself does not imply 6-bit storage.

Observed NCU reads are 1.008785× and 1.006906× logical J-based weight bytes at
M32/M256; U/J is 16/16 and 26/27. The duplicated M256 job's logical bytes are
only 3.6783% of measured reads. This does not justify a large reuse rewrite.
Conditional unchanged-traffic arithmetic at 4.8 TB/s leaves about 25% of these
profile durations, with no claim that this headroom is attainable or transfers
to the decoder. Low tensor activity does not isolate consumer dequantization.

## Four-stage narrow WGS needs a separate workspace, and remains secondary

The standalone exact ef7 core (`...ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a.py`)
defines FT_T/MMAX/JMAX at 530–540, Smem at 571–617, joined rings/barriers
at 2563–2573, and dynamic-smem sizing at 3714–3720.

For FT_T64 the current rounded metadata is 22272 bytes; three 36 KiB weight slots
and three 32 KiB activation slots give 231168 dynamic bytes. NCU independently
records 231168 dynamic plus 1024 driver-reserved bytes, with 232448 bytes opt-in
dynamic limit. Simply setting FT_T32 and four joined slots grows JMAX 88→136:
rounded metadata becomes 23808 bytes and dynamic use 236800 bytes, exceeding the
limit by 4352 bytes. The raw 208 KiB ring estimate alone is insufficient.

Specializing MMAX32/EMAX256 would reduce metadata enough and guarantee that
FT_T32 does not split a job for M <= 32. It requires a separate scratch/layout
proof, shape guard, and generalized barrier indices (currently BW_E=3/BX_F=6).
NTT5–8 are already compile-time excluded when FT_T <= 32 at 3167–3172.
This could test whether extra outstanding loads help a narrow workload, but
there is no evidence yet that depth limits it. Even a hypothetical 25% MoE
reduction confined to decode 32 is roughly 1.7% aggregate in profile accounting.
It is not a substitute for the missing large improvement; no implementation
or GPU request is recommended from this audit.
