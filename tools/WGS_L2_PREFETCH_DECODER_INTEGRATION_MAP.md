# Conditional L2-prefetch decoder port

2026-11-04. Read-only map; no decoder candidate or GPU work. The standalone
candidate passed correctness but failed its fixed performance screen: affected
normalized latency increased9.03% against public and10.03% against private
parents, with stable controls. The port is closed; retain this map only as source
evidence. No integration is authorized.

Exact sources: decoder97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2,
standalone parent247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0,
prefetch85da23922def6696b206a227af944c13cb0f439c90bf4fa4570c39ca92afe759.

The decoder and standalone parent have **byte-identical** `wgs_wproducer`,
`okc_smem_bytes` and `okc_run` function bodies, confirmed directly from their AST
string literals. Producer hash is b64ac1a3af9622c0a6a9f2a529498dc3ab6e38351a78d27243900e6a34d64dfc;
host launch hash is bedd1bc40b2fb79d0e8edd26a09c62f34b61d8e3826b8c3ee5acf00681e9937f.
Their C++ export strings are also identical. The prior full-source map found
only a historical trailing BIGM comment difference in embedded CUDA.

Two existing AST nodes suffice:

| Node | Scoped change |
| --- | --- |
| `MOE_CUDA_SRC` (line5398) | Apply the exact reviewed distance3 producer insertion and LPFR host admission helper/call. Keep decoder comment convention and every other device/host function exact. |
| `_ma_ext` (line8991) | Unique extension namespace. Preserve flags, full define-tuple cache keys, exports and initial PDL setup. WGS-only verbosity may be enabled for evidence if separately accepted; optimization flags need no change. |

No preparation, packing, state flag, selector, tensor-pointer ABI, scratch or
global weight-format change is needed. `_POS_PACK`, the ordinary9216-byte WGS
pieces,36864-byte stage/slot, all ma_w entries and guards stay exact. Unlike SC6,
there is no new state-dependent encoding, so a state-selector rewrite would be
outside this port. MAIN remains NW16/FAST1=0/EPI4=0/WGS1 with PRMT1/MULDEC1/DQ7
when eligible,576 threads and231168 dynamic shared bytes. General-scale fallback
remains non-WGS. Tiny/MID are unchanged; specifically decoder TINY retains
MULDEC1 and no FINPF1, rather than standalone's MULDEC0/FINPF1 convention.

The first-launch LPFR query uses the exact BIG0/BIG1 pointers after the existing
attribute setters. Valid API results and occupancy>=1 are required; local bytes
and compiler warnings are recorded, not automatic performance kills. The actual
allocation must be measured in the decoder build, not transferred solely from
standalone. Source equality does not establish identical compiler allocation
under different build settings or identical runtime behavior in the whole layer.

The sole timed agent selection is `_after_attention`'s `_moe(T)` at15028. Preserve
it and its per-call `okc_set_pdl(_pdl(T))`, initial kernel dependency wait,
completion trigger and programmatic host launch. Decoder T32 uses PDL while
standalone does not; larger affected T128/256 disable it. Added L2 requests may
compete with overlapping dense/attention traffic, so a standalone gain need not
survive the full layer. The hint remains after WS_JOIN and a real weight copy,
uses only published route metadata and immutable weights, and creates no new
activation-readiness dependency.

UV pooler may process physical2T dense rows, but routing supplies T-row BF16 xm
to MoE. Preserve local IDs0..31/-1, FP32 route weights, lo0/count32 and SM-count
grid. For T>384, retain the existing384-token chunk loop and original-T selector.
No change belongs in routing, UV, SK2B, unpooler, attention or Marlin banks.

Preparation builds agent modules before tuning and propagates compilation errors.
However `_tune` catches agent runtime errors and may select Marlin; official PASS
alone cannot prove prefetch activation. Require clean LPFR records, expected
agent backend coverage and no hidden resource failure. Retained first-launch
records establish compiled resources, not benchmark-phase identity or whether
the hardware honored hints. Do not alter tuner/fallback policy to force a win.

Expected parent profile scope: affected MAIN indices4/5(T32),8/9(T128),10/11(T256);
controls0/1(T1),2/3(T16),6/7(T8). If standalone passes, a focused port review needs
exact embedded-delta/source preservation, decoder-specific non-WGS preprocessing,
unchanged launch/PDL/packing and complete13-test/12-benchmark/12-profile private
evaluation against both97af receipts. No source promotion follows automatically.
