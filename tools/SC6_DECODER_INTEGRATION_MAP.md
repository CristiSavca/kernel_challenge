# Conditional SC6 decoder integration map

2026-11-04. Read-only map, no decoder candidate or GPU work. A useful standalone
SC6 result is a prerequisite; this document does not authorize a port.

Pins: decoder `97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`,
standalone parent `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`,
SC6 candidate `8044835232a26e8188471ac5296cd598339e00518641d03f055c129502aa8cc5`.
The earlier `WGS_SHARED_A_DECODER_INTEGRATION_MAP.md` established the parent
embedded CUDA equality after one BIGM trailing-comment normalization, identical
C++ exports, MAIN flags, scratch and the actual public profile scope. SC6 needs
additional preparation/dispatch changes because its stored bytes depend on mode.

## Minimal isolated port and state ownership

Six existing AST nodes need a scoped edit; three small helpers can be added.

| Existing node | Exact port |
| --- | --- |
| `MOE_CUDA_SRC` | Apply reviewed SC6 CUDA delta: default define, compact piece/global-stage constants, three producer byte-count expressions, exact scale reconstruction, bounded SC6R host resource helper. Keep all other embedded CUDA and decoder PDL export untouched. |
| `_ma_ext` | Unique extension namespace; retain complete define tuple as `_MA_EXTS` key, decoder optimization flags and PDL initialization. Optional WGS compiler verbosity requires explicit review; it is not required by packing. |
| `_wgs_copy` | Add optional `sc6=False`; apply the actual `_sc6_wgs_pack` to its final 1024-byte scale record before concatenation. Keep decoder `_pack_codes_nat` and `_POS_PACK` use. |
| `_ma_prepare` | Use state-owned `ma_sc6` for both appended WGS banks12/13. Entries0..11, expert rows, existing `.view(torch.uint8)`, FP32 global scales, `post_expert_norm`, scratch/guards remain identical. |
| `prepare` | Before MoE build threads, compute/store immutable `ma_sfx` from actual W13/W2 bytes and `ma_sc6 = bool(ma_sfx and all_bytes_in_0x40_0x7e)`; initialize state-owned module refs. Build the same MAIN/TINY/TINY2/MID configurations through the new state selector. Preserve `_att7` parallel build, joins/errors and all tuning policy. |
| `_after_attention` | Replace its sole agent `ext = _moe(T)` with the state-owned selector. All routing, PDL setters, launch arguments, chunk loop and Marlin branch stay exact. |

Add the actual scale qualification and `_sc6_wgs_pack` helpers, plus one
`_ma_state_moe(st, M=None)` selector/cache. The selector reproduces current
thresholds4/8/16, TINY2 PRMT guard and `_main_defines(sfx)` fallback, using
`st['ma_sfx']`. Add `-DOKC_SC6W=1` only when the selected tuple contains WGS and
`st['ma_sc6']` is true. Store loaded module references in that state; keep the
existing process-wide `_MA_EXTS` cache keyed by the complete immutable define
tuple. The state holds refs, not duplicate extension binaries or weight banks.

**Do not copy standalone's global callers or add a mutable `_MA_SC6W`.** Decoder
currently uses global `_MA_SFX` and `_MOE*` aliases. Simply setting another global
at prepare time could launch an old state's compact bytes with another state's
kernel after alternating prepares. A state selector is a required integration
difference, even though standalone already has state-owned flags. Existing
legacy `_moe*` wrappers may remain untouched but must be bypassed by both new
build closures and the sole timed agent call. Preparation's four MoE build
closures should capture the same state and representative selector values
None/1/8/16; optional TINY2 guards preserve the current nonempty flag definitions.
No flag enters generic `ma_sfx`, tiny or MID.

Unlike standalone, decoder has one `ma_w` representation for all token counts,
not separate tiny/larger layouts. This is safe only because compact entries12/13
are WGS-only: tiny/MID consume unchanged0..11, and their natural-fat aliases of
12/13 remain dormant under default WGMIN9 with unchanged flags. Explicitly prove
that property in the port; do not duplicate the entire bank for small tokens.
If PRMT is valid but SC6 is not, MAIN uses parent eight-bit WGS. If PRMT is
invalid, retain ordinary entries and the existing warp-MMA fallback. Alternating
all three states must select matching cached modules in preparation and timing.

## Decoder conventions that must survive

`MOE_MULDEC=True` selects `_POS_PACK = _stored_pos_mul()`. Decoder `_wgs_copy`
does not take standalone's `positions` argument; retain its existing packer API
and add only the SC6 boolean. Decode multiplication, global FP32 scales, BF16
post-expert norm and all code/scale ordering remain unchanged. MAIN defines are
the standalone NW16/FAST1=0/EPI4=0/WGS1, with PRMT1/MULDEC1/DQ7 when eligible.
Decoder TINY uses MULDEC1 and omits standalone TINY's FINPF1; preserve those
differences. TINY2/MID tuples match the parent. Do not transplant standalone's
dual layout or use_mul dispatch.

`_after_attention` offers only agent and Marlin. Both tuner synthetic calls and
runtime calls pass through that same function. Marlin banks, `w13q/w13s/w2q/w2s`,
`ma_ok`, fallback policy and tuning choices remain unchanged. A resource failure
inside an agent launch can be caught by `_tune` and select Marlin: an official
PASS alone is not proof that SC6 ran. Require successful SC6R records, eligible
preparation evidence, agent selection/profile coverage and no silent resource
failure before claiming an SC6 result. Preserve existing exception policy; an
out-of-tuner preflight would be a separately reviewed additional change.

The agent receives BF16 `xm[T,4096]`, local IDs0..31/-1, FP32 route weights and
unchanged `ma_scratch`. Its host launch remains lo0/count32 and `state['nsm']`.
For T>384, preserve the existing384-token chunk loop and each slice's physical M.
The source outer selector still uses original T; do not change selection for the
last chunk. Scratch sizes, fixup buffers, guards and shared231168 bytes do not
change. SC6 changes only global weight requests and the corresponding scale
loads/reconstruction.

PDL remains enabled per decoder `_pdl(T)` (T<=32). Keep the existing host setter,
kernel dependency wait/completion trigger and programmatic launch configuration.
UV dense pooler may process physical2T rows, but `_route_split` yields T-row `xm`;
MoE dispatch remains based on T. No UV, router, SK2B, unpooler, dense or attention
changes belong in this port.

## Required evidence if standalone is useful

Parent97af public profile scope is MAIN indices4/5(T32),8/9(T128),10/11(T256);
small controls are0/1(T1),2/3(T16),6/7(T8). MAIN MoE rows range80–206us and
25.6–61.7% of their profiler totals. These sums do not establish removable time
because PDL overlaps and backend retuning exist. A small standalone improvement
can be diluted in the full decoder; no numerical gain is assumed.

The port proof should bind exact embedded SC6 CUDA after the known comment
normalization, unchanged decoder flags/non-WGS preprocessing, actual `_POS_PACK`
packing, all three eligibility states, repeated/alternating state prepares,
state-owned threaded build keys, both backends, M385 chunks and PDL setters.
Then obtain independent source review and one complete private decoder check
(13 tests,12 benchmarks,12 profiles) against both exact97af receipts. No active
submission or leaderboard source changes follow from this map alone.
