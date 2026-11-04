# Compact WGS → decoder integration map

2026-11-04. Read-only planning by `/root/attention_paper_reset`; no decoder
candidate was created, no source changed, and no GPU or official-job call ran.
Integration is conditional on the compact standalone result and a new exact-source
critic review. A passing standalone check alone does not establish a useful gain.

## Exact inputs and isolation

- Decoder incumbent: `muse_decode_layer/submission.py`, SHA-256
  `80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`.
- Compact standalone: `muse_moe_w4a16/candidates/native_wgs_compact_v10.py`, SHA-256
  `cf09ba584cc1537802636bb905ac08cda2095f83a163358e1c9bb716490130b8`.
  Extracted `CUDA_SRC` SHA-256:
  `789c1a25dbf93483acc0715748110b2cb90560ed7156f4969d813a4b02e0bf82`.
  The final cf09 header differs from reviewed d27; executable code is unchanged.
- Decoder `MOE_CUDA_SRC` currently equals standalone parent ef7b210b's
  `CUDA_SRC` except the comment after `#define OKC_BIGM 32`. There is no separate
  decoder CUDA algorithm to merge. Do not include KPERM, dense global escapes,
  attention changes, or new tuning choices in this integration.

Exactly five existing top-level AST nodes need change:

| Decoder node / current line | Integration operation |
| --- | --- |
| `MOE_CUDA_SRC`, 5357 | Copy the exact compact `CUDA_SRC`; keeping its comment is harmless and permits literal equality. This includes the compact kernels, host scratch offsets, occupancy gate, and WGS-only grid doubling. |
| `MOE_DEFINES`, 8910 | Copy compact `MAIN_DEFINES`: change MAIN `OKC_NW=16` to `8`; keep FAST1=0, EPI4=0, WGS=1. |
| `_ma_ext`, 8943 | Change only the extension module prefix to a unique compact decoder name, e.g. `dl_okc_moe_compact_v13_`. Preserve decoder compiler flags, dispatch suffixes, and PDL wiring. |
| `_wgs_copy`, 9145 | Copy compact packing with the signature adaptation below. |
| `WGS_FIXF`, 12847 | Set `2 * 320 * 2 * 4096 + 2 * 320 * 2 + 136 * 32`. |

All other AST nodes should compare equal to 80a4. In particular `_ma_prepare`,
`_ma_make_scratch`, `_moe`, `_after_attention`, `kernel_fn`, the tuner, dense pins,
attention, router, and residual/norm code remain unchanged. The source header may
describe provenance without changing executable AST.

## Decoder packing and compile details

Standalone `_wgs_copy(codes, sf, rows, positions=None)` calls
`_pack_codes_nat(codes, rows, positions)`. Decoder uses
`_wgs_copy(codes, sf, rows)` and `_pack_codes_nat(codes, rows)`; its latter helper
reads global `_POS_PACK`. Remove only the optional `positions` argument/default
and the corresponding argument to `_pack_codes_nat` when transplanting the
function. Preserve the original decoder helper and `_POS_PACK` assignment.

Decoder `MOE_MULDEC=True` selects `_stored_pos_mul()` for `_POS_PACK`; `_ma_ext`
adds `-DOKC_MULDEC=1`, and DQ7 remains enabled. Do not replace `_POS_PACK` with
`_POS` or import a standalone default position table. The new byte proof must
execute the actual adapted decoder helpers and independently recover all W13/W2
FP4 codes and scale bytes with this encoding. W13 retains its existing
gate/up-interleaved `rowsA`; there is no K permutation.

Preserve all decoder TINY/TINY2/MID/fallback flag tuples. A real difference from
standalone is that decoder `MOE_TINY_DEFINES` lacks `-DOKC_FINPF=1`; copying all
standalone tuples would change T≤4 and invalidate the control. TINY2/MID retain
their existing FINPF flags. PRMT eligibility still comes only from weight scale
bytes; non-PRMT input weights select the unchanged non-WGS fallback.

Other compile settings stay SM90a, O3, fast math, DQ7, MULDEC, NS=3, NSF=3,
FNST=4. Compact GS2/FCI3 are source defaults with a static assertion. Decoder
currently uses ptxas `-O3`; standalone adds `-v` reporting. Copying that reporting
flag is unnecessary: compact host resource output reports actual threads,
dynamic shared memory, occupancy, registers, and local bytes. Preserve full
compiler warnings for the serialization gate.

## Scratch and launch contract

- C++ compact geometry: two compute WGs plus two producer warps (320 threads),
  128 output rows/unit, 32 K128 stages, 9,216 weight bytes/slot and up to 16,384
  activation bytes/slot. The reviewed dynamic shared-memory budget is 98,048 B.
- Python fixup floats increase from 5,246,336 to 5,248,512: only 2,176 extra
  counter words, **8,704 bytes**. Two phases ×320 slots ×two WGs retain the old
  total FP32 fixup payload. Keep `_MA_MMAX=384`, `_MA_EMAX=3072`, `_MA_JMAX=88`,
  `WGS_XROWS=4160`, and guarded allocator unchanged.
- Keep `state["nsm"]` as the physical SM count. Pass it unchanged to `okc_run`;
  compact C++ alone doubles the grid for WGS. Doubling both Python and C++ would
  exceed residency and can deadlock the grid barrier. Non-WGS grid stays old.
- Actual occupancy of both `moe_fused<false/true>` is checked before the first
  launch and must be ≥2; `grid+1<=320` must pass. Do not weaken this gate.

## Decoder-specific gaps in standalone evidence

1. **PDL chain:** standalone PDL is off. Decoder enables it at T≤32, including
   changed T17..32. Preserve the initial all-thread `griddepcontrol.wait`, parent
   trigger placement, and `_after_attention`'s per-call `okc_set_pdl` switch.
   T32 requires a real in-chain test with the new two-CTA admission requirement;
   isolated occupancy and CPU scheduling proofs do not validate CUDA graph
   progress, upstream router visibility, or downstream unpooler visibility.
2. **Inputs and weights:** decoder packs 32 already-local experts and passes
   `lids` (0..31 or -1), `lo=0`, `nloc=32`, BF16 `xm`, FP32 route weights, and
   `post_expert_norm`. Standalone takes a global local_start slice and calls the
   norm `post_norm_w`. Keep existing decoder preparation/calls: transplanting
   standalone `prepare`/`kernel_fn` would break this interface. Decoder `lids`
   allocation and 384-row slicing preserve 16-byte alignment.
3. **Warmup and scratch:** decoder tunes several sizes/backends in prepare,
   reuses guarded scratch, then checks sentinels. Preserve this path and test
   fresh/repeated graphs and shape transitions, including ragged T24/33/40.
   The full decoder oracle also covers downstream amplification of MoE error.
4. **Attribution:** existing synthetic tuning may select a different expert or
   dense backend because timing changes/noise changes. Preserve tuner code but
   inspect returned profiles: target cases must actually run compact WGS, and
   unchanged T1/8/16 controls must be explained. A passing run that silently uses
   Marlin does not validate compact WGS. Do not add new pins merely to hide this.

## Affected loads and measured context

Source: scored decoder job `4b216098993e4112`, stored profile
`experiments/candidates/muse_decode_layer/profiles/80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c.4b216098993e4112.json`,
SHA-256 `848e23f2913e7577300bae4329090a1d63bc420ff114c6292172f722d9f7e722`.
Shares combine sliding/full profile shares using normalized benchmark time ×
case weights 46/16. They are approximate attribution, not exclusive wall time:
PDL durations can include waits/overlap and profiles are separate captures.

| Load | T | Compact path changes? | Approximate MoE share | Stored MoE sliding/full µs |
| --- | ---: | --- | ---: | ---: |
| decode 1 | 1 | No | 29.78% | 22.8 / 22.4 |
| decode 16 | 16 | No | 45.25% | 82.8 / 59.4 |
| decode 32 | 32 | Yes, PDL on | 40.05% | 79.5 / 86.1 |
| verify 1 | 8 | No | 49.33% | 51.3 / 64.7 |
| verify 16 | 128 | Yes, PDL off | 55.27% | 145.2 / 165.1 |
| verify 32 | 256 | Yes, PDL off | 54.22% | 194.6 / 218.0 |

Hypothetically reducing only affected MoE times by 10% /20% /30% gives roughly
2.53% /5.12% /7.79% aggregate decoder latency reductions, or 2.206× /2.266×
/2.332× from the 2.150186× incumbent. These are profile-share scenarios, not
predictions or attainable bounds. This mechanism does not improve three of six
loads; a modest standalone gain cannot establish first place.

## Next permitted experiment

After a useful standalone result and root decision: build one isolated decoder
candidate from 80a4 with exactly the five-node scope; prove literal compact CUDA,
adapted packing, scratch sizing, exact dispatch flags, and all other decoder AST
nodes unchanged. Reuse the style of `tools/test_decoder_kperm3_integration.py`,
but use compact K128/128-row byte coordinates and the current source hashes.
Request independent exact-source review, then one full private decoder check:
13 correctness tests plus12 benchmarks, complete compiler/resource/profile logs.
Require correct execution of the changed path and a useful whole-decoder gain
beyond noise before considering promotion. Do not bundle another mechanism or
repeat an identical queued check.
