# Public source scout, 2026-11-04

Decision: one new licensed implementation contains a distinct mechanism worth
isolated design review: per-expert completion allows the MoE finalizer to start
before unrelated experts finish. This is a research lead, not a verified speedup,
an integrity certification of the entire public file, or a recommendation to
replace the passing decoder. No downloaded code was imported, compiled, or run.
No GPU allocation, login, participant communication, or submission occurred.

## Provenance and reported results

The selected source is [keell0renz's decoder at pinned commit
890809cee421500b00126d084e81b338ef9181ab](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_decode_layer.py).
Its initial source commit is
`34352903efb5f394b6fb2fb0ba7848afc39162f0` (2026-11-01 07:09:13 UTC).
Independent raw fetches at both commits yielded identical 979654-byte files,
SHA256 `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
The local manifest records URLs, bytes, hashes, and the source path's commit
history. Source, README, LESSONS, Apache-2.0 LICENSE, and attribution NOTICE are
preserved under `third_party/public_scout/keell0renz-890809c/`; total selected
upstream content is 1005920 bytes, below the authorized 2 MB budget.

The [pinned README](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/README.md)
reports a historical decoder board score of 2.2659 and exact-file check/submit
scores of 2.2641/2.2532. No raw receipts or official job IDs were found in the
repository tree. These remain author claims. The later README explicitly
disclaims a different current #1 entry as gaming the judge. The preserved source
predates that update and its bytes are unchanged; this does not prove the source
is compliant or connect it to an official scored job. Do not call it #1 code.

The [NOTICE](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/NOTICE)
attributes the vjeux base, alygg77 decoder/MoE/attention work, chenwainuo dense
kernels, ZhimaoL attention branches, Madhavan bulk reductions, and
NathenFernandes MIT-licensed router/projection work. Preserve this attribution
and the applicable licenses if extracting any code.

## Independently reviewed structural lead: per-expert finalization

Public source line references below refer to the preserved file's exact bytes.

* `OKC_MFIN`, lines 3891–3908, replaces the final WGS grid barrier with completion
  counts for each local expert. `wgs_flush` at 5882–5953 publishes one release
  increment per consumer warpgroup after the phase-B epilogue stores. Whole
  units use deferred `kind=4`; a split unit's last contributor publishes after
  completing the epilogue and its 128-thread barrier.
* There are 16 phase-B output units per job and four consumer warpgroups, hence
  64 releases per job. Finalizer lanes 0–7 acquire only the token's local
  experts, target `64 * ceil(local_count / 64)` for ordinary jobs, then synchronize
  the CTA before reading output rows (6524–6533 and 6582–6591). Every local route
  is still consumed; the completion predicate changes when work may start.
* This keeps finalization inside the same persistent kernel. It is distinct from
  rejected V8's separate finalizer launch, which added a launch and kernel-wide
  dependency. Potential benefit is overlap of token finalization with unrelated
  experts' trailing work, plus removal of the grid-wide completion exchange.
  It does not reduce expert weight traffic or prove a scheduling bottleneck.
* The flag layout at 5564–5578 uses two parities for per-expert counts, activation
  readiness and phase-A stage readiness. An atomic counter wraps at `2*G` to
  select launch parity (6388–6414). Runtime callers pass the device SM count
  (`state['nsm']`, set at 11260), including the main paths at 11617, 12502, 13129,
  13335 and 14146. The first instruction before producer/consumer work is
  `griddepcontrol.wait` (6257); this covers prior PDL execution before scratch
  reuse. A clean port must still prove fixed G, initial zero state, resident
  CTAs, all producer/fixup completion, and transitions between tiny/MID/WGS
  builds sharing scratch.

Independent critic review found the count and release/acquire mechanism
internally coherent under those invariants, with no additional statically
demonstrated race. It did identify a blocking failure mode for direct copying:
the finalizer waits silently fall through after 4,000,000 unsuccessful polls
(6530 and 6588). An isolated implementation must use a correct readiness wait
or explicitly fail; elapsed spins must never authorize reading unfinished data.

Whole-file adoption would mix several unrelated changes. The public default WGS
path stores intermediate y rows as BF16 (5824–5827, 5856–5861); our frozen 80a4
stores FP32. The public B4 finalizer at M17–32 keeps FP32 and changes reduction
layout/order. Its producer barriers are separate rather than our joined
barriers; its deferred flush placement is FCI4 rather than our FCI3. SC6 weight
packing and cost-model changes are also present. Retain our math, ybuf precision,
producer pipeline, scheduling costs and finalizer layout in any MFIN-only test.
No whole-file numerical or rules certification is claimed.

Our earlier descriptive phase traces show several microseconds in completion
barrier/finalizer windows, but sums of CTA waits are not removable wall time and
the phase-window heuristic is not a bound. The local NCU result is about 75%
DRAM utilization with a conditional same-traffic ceiling, not evidence that
MFIN can recover the missing ~15% decoder latency. This lead supports one causal
experiment only after a scratch/lifecycle proof; it supports no gain forecast.

## Two secondary deltas, with limited scope

**Exact six-bit scale packing.** `_sc6_ok` at 7066 checks every actual scale byte
lies in 0x40..0x7F; the independent PRMT predicate also applies. `_sc6_wgs` at
7085 packs eight six-bit values into six bytes with reversible bit planes.
The WGS piece falls from 9216 to 8960 bytes and stage from 36864 to 35840 bytes
(5550–5551), exactly 2.7778% less weight traffic. The original encoding remains
the fallback. This is a real math-preserving traffic reduction when the weights
qualify, but unpack instructions can erase it. The author's ~2% MoE statement
in LESSONS has no independent receipt. Even ideal proportional 2.78% MoE gains
on only the three WGS loads are a secondary aggregate margin, not a large bet.

**Warp-parallel job-cost prefix.** `wgs_ranges` at 6152 uses a warp shuffle scan
and distributes four range lookups across lanes, replacing the incumbent's
serial lane work. It also changes the per-unit cost model, so those changes must
be separated. The author's 0.9% M32/0.5% M128 claims are unverified. The scope is
prologue time, not the entire expert latency.

## Route, math and prepare review limits

The inspected runtime paths consume current `data`, return the current top-8 IDs,
translate them using `local_start`, and execute all local routes. The small-T
FP32 router path is not by itself a rules violation: official baseline lines
315–328 also keep normalized router input FP32 at T <= 8. `probe_q` at 4313 is
identity. These checks are narrow findings, not an exhaustive audit of 980 KB
and the embedded alternative native strings.

The final `prepare` adds synthetic MoE A/B timing (16224–16345). Those inputs are
generated independently inside prepare, not taken from `kernel_fn(data, state)`;
no observed input computation is moved out of the timed path by this addition.
However, this A/B combines MFIN and SC6 changes, uses a write flush, and can
consume the preparation budget. Its ratios cannot isolate the mechanism.
Do not copy the timing helper, diagnostics, tuning side effects, or environment
options into a production candidate. No harness/frame/network access or
answer-cache mechanism was found in the inspected active paths; a source scan
and targeted review cannot certify their absence everywhere.

## Bounded search coverage and alternatives

Public GitHub repository searches covered recent repositories matching the
challenge and Muse kernel names, plus public repositories of AlexCuadron,
dolanzhao, caoshiyi, PierreNowak and related surfaced names. Repeated known
rohan/vjeux/lee implementations were excluded. Repositories without an explicit
license were not copied. No bulk clone or participant contact was used.

* [AutoclickerI/muse-spark-open-kernels at 85aed1f](https://github.com/AutoclickerI/muse-spark-open-kernels/tree/85aed1fb1365877b615542dd44980646816d6b26)
  is Apache-2.0 and exposes source/receipts; its author reports decoder 1.889×,
  below the frozen incumbent. It is not a current decoder-score lead.
* [nibague/open-kernel-challenge at cddb565](https://github.com/nibague/open-kernel-challenge/tree/cddb5658562bd225bab4d22dc735d7eddfb886c4)
  is Apache-2.0, but the inspected decoder is a short starter and no superior
  whole-decoder result was established.
* caoshiyi's licensed EAT kernels target a different AMD attention workload;
  trxphxle reports a much lower historical rank. Neither supplies a concrete
  superior Muse decoder mechanism from this bounded pass.

The retained lead is therefore an isolated MFIN design, not a wholesale source
swap or a newly verified leading solution. Root owns any next implementation
decision, allocation and official validation.
