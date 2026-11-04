# Public-source scout and bounded phase-A publication probe

2026-11-04. The reviewed kind-1-only proposal is now implemented in isolated
`native_wgs_early_a_publish_v10.py` (decoder) and
`native_wgs_early_a_publish_v8.py` (standalone) candidates. Their adjacent
decoder note records exact hashes and static validation. No GPU or official job
was requested by the author. Root owns the next experiment. Independent critic
reviewed the final bytes and supports at most one bounded screen if the dense
experiment is null; this is not approval for promotion.

## Public evidence and its limits

Read the public feed and the public portions of all 35 listed threads on Guild.
No registration, login, installed skill, heartbeat, participant contact, or
private content was used. **No new licensed, pinned implementation with a
verified decoder result above our 2.150186× was found.**

- [VkaZas/prateekm1007 discussion](https://guild.1nterface.ai/t/120?credible=all)
  proposes a phase-B dependency mechanism: late A publications delay activation
  readiness, which prevents consumers from returning weight-ring slots. Their
  synthetic hot-tail trace reports concentrated release waits, not critical-path
  savings. Neither participant publishes a successful A/B of the proposed fix.
  Their earlier apparent small-M MoE advantage was retracted after establishing
  byte-identical source and a PDL attribution difference. These are participant
  observations, not results reproduced here.
- [OperatorStack pinned source](https://github.com/operatorstack/open-kernel-challenge-solutions/tree/6d9b77656b6c295e421be9e1424b0f8f99299b0a)
  is MIT with the Apache-2.0 SGLang exceptions enumerated in its NOTICE. Its
  README records decoder 1.7352× and MoE 1.8320×. The suggested masked NVFP4
  permutation is already present in our `dq4` (`0x81C081C0` / `0x80C080C0`);
  it supplies no new unpack opportunity. The public layer-expert adapter is
  reported at only 1.907419× standalone MoE in
  [thread 4](https://guild.1nterface.ai/t/4?credible=all).
- [Consumer-loop discussion](https://guild.1nterface.ai/t/49?credible=all)
  reports reduced instruction counts without useful whole-operator gains after
  occupancy fell. No pinned improved code accompanies that claim. It is a
  warning against treating fewer instructions as a speedup, not a universal
  falsification of dequant changes.

## Exact incumbent mechanism

All line references below are to the frozen decoder file
`experiments/candidates/muse_decode_layer/80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c.py`.
The active source has the same bytes. Its embedded native CUDA SHA-256 is
`d0d950c33764d6b1193ea0caac9db9fb6cde4d986cd07fa862adab7f830fa39d`.

| Location | Fact |
| --- | --- |
| 7644–7646 | `WGS_NS=3`, `WGS_NX=OKC_WGSNX`; joined completion requires equal depths. |
| 7684–7700 | `WGSREORD=1`, `WGSDFIN=2`, `WGSFCI=3`, `WGSGS=2`. |
| 7863–7869 | Weight producer can run ahead up to ring capacity; it does not wait on `doneAs`. |
| 7909–7916 | B activation producer waits for 8 A warpgroup publications for the particular job/stage. |
| 7911, 8113 | Both producers complete the **same** `BW_F` barrier; our code does not have the public post's separate W-full/X-full waits. |
| 8136–8137 | Weight credit returns only after the last code/scale group has reached registers. |
| 8191–8193 | Previous segment's pending work flushes after commit group 3 of the next segment's first stage: 8 issued k16 MMAs. The source comment records a historical 0.5–1% benefit from this deferral. |
| 8217–8224 | A full unit produces `pd.kind=1`; a split unit produces `pd.kind=2` under the actual `WGSDFIN=2`. |
| 8022–8085 | Kind 1 is a barrier plus release `doneAs` increment. A kind 2 uses an atomic round trip and can become the last-arriver reader/epilogue. It is **not** fire-and-forget in this build. |
| 8910 | Main flags enable WGS/16 consumer warps without overriding these macros. |

The broad coupling is present: incomplete activation transactions hold joined
`BW_F`, preventing weight consumption and `BW_E` return. Merely moving a W2
prefetch ahead is not new; the producer already runs ahead. Changing `WGSFCI`
globally would also change phase B and leave the wait before the next segment's
first stage intact, so it is not the cleanest causal probe.

## Proposed first A/B: publish only whole A units immediately

After the NTT switch in `wgs_phase`, immediately before `gs += s1 - s0`
(line 8253), add only:

```cpp
if constexpr (!PB) {
  if (pd.kind == 1) wgs_flush<false>(p, S, pd);
}
```

This uses the existing publication function after a full A unit's epilogue,
instead of waiting for the next segment's joined barrier and first 8 MMAs.
It keeps split records, their atomic/last-arriver behavior, and every phase-B
flush at the old location. Existing end-of-phase flushing remains and sees an
empty record after early publication. No extra payload storage, ring depth,
thread role, weight format, MMA group, or mathematical operation is introduced.
Actual compiler register allocation still must be checked; identical source
storage requirements do not guarantee identical generated register use.

With J=16–29 jobs in our saved standalone diagnostic, A has 32J units and at
most 131 units cross 132 CTA range boundaries. Thus at least 74.4–85.9% of A
units have whole-unit publications affected. **This is event-count coverage,
not a latency fraction.** One unchanged split unit can still hold up a B stage.
A negative result would reject this limited probe, not all readiness scheduling.

The broader variant (flush all A pending work immediately) also advances split
atomics and potentially expensive readback/SiLU, confounding that first probe.
Separating weight credit from activation arrival is a larger rewrite: retaining
all four load groups requires 64 B packed codes + 32 B loaded scales per
consumer thread (24 scalar registers; scale loads are replicated across each
four-lane subgroup), instead of one 16 B + 8 B group. The incremental packed
bank is approximately 18 registers before compiler lifetime effects. Retaining
all decoded A operands instead requires 16 k16 steps × 4 registers = 64 registers
before scale/bookkeeping state. GS4 already failed after doubling its operand
bank by 16 scalar registers; added buffering needs an explicit register/storage
design, not an assumed free overlap benefit.

## Gate and correctness review

- Preserve the existing 128-thread barrier before `red.release`, the exact
  eight-publication count, per-stage acquire, all fixed-point/float math, and
  scratch reset/generation rules. Every warpgroup must enter the condition
  uniformly. `wgs_flush` clears `pd.kind`, preventing duplicate publication.
  The independent review found no static ordering/count defect: the full-unit
  segment has already executed `wg_wait<0>()` and its xB stores before setting
  kind 1. A changed execution schedule can still alter the order of unrelated
  split floating-point atomics, so identical arithmetic is not a substitute for
  numerical checks.
- A successful probe must improve paired complete MoE times on fresh M32,
  M128, and M256 routes, with numerical checks, repeated graph replay, unchanged
  tiny controls, and compile-resource/SASS review. Root's bounded general
  runner is sufficient; do not modify the official evaluator.
- The main counter-hypothesis is that an immediate barrier exposes the xB store
  drain which was hidden beneath the next segment's MMAs. The historical source
  comment favors deferral. Check for increased registers or C7510 outlining;
  an apparently small source edit can change the full inlined WGMMA function.
- Suggested continuation bar: at least 3% repeatable complete-MoE improvement
  at M128 or M256, no greater than 1% systematic loss elsewhere, and the same
  direction on a second paired run before whole-decoder integration. Reject
  promotion on a flat/negative screen; do not turn this into a CI-index sweep.
- There is no measured recoverable-time ceiling from our diagnostic. The public
  26–31% maximum-CTA wait ratios are not end-to-end savings or bounds. Even a 3%
  expert improvement would only supply roughly 1–1.7% per-load decoder savings
  at the recorded expert shares, far below the 14.76% total latency reduction
  needed for the prior leader. A large improvement remains unproven.
