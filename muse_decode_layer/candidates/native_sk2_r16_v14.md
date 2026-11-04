# Fixed SK2 R16 family — November 4, 2026

CPU-verified candidate; official private check `28d3c46e68504ee4` submitted
November4 at11:58:46 UTC and queued at12:05 UTC. No GPU result yet. Frozen source:
`464115b2c871338f8d0f51768b82f6bd9090139aea13588ec668a9de587e44f8`.
Parent is exact scored decoder
`97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
The active submission is unchanged. The concrete decision contract is
`tools/SK2_R16_IMPLEMENTATION_PLAN.md`; this is one fixed hypothesis, not a sweep.

Smaller output-row units reduce per-CTA accumulator footprint and shared storage,
potentially admitting useful independent pipelines. They preserve the lossless
BF16 codec and every model term. No measured bottleneck or speedup is claimed.
The actual parent has zero local allocation on all recorded SK2 specializations.

| Family | R / NT / NWR / WSK / UPW | NS8 / NS16 / NS32 |
| --- | --- | --- |
| Every fused pooler/router, including tuner-only32 |16 /2 /1 /8 /1|4 /3 /2|
| Separate pooler32 |16 /2 /1 /4 /2|unchanged /unchanged /2|
| QKV |16 /2 /1 /4 /2|5 /3 /2|

The shared pooler configuration also changes ordinary non-fused pooler16.
Router reduction is uniformly eight pieces: two K512 stages per piece, eight
warps, full BF16 hi/lo contributions. Both scratch allocations and all partial
strides/counter users use eight pieces. The stage ring reserves the larger of
main and router weight payloads. Existing barriers, release/acquire operations,
PDL placement, UV combination, packing/escape helpers, O/unpooler/BND, attention,
MoE and large-T implementations remain unchanged. FP32 association changes with
the new K partition and correction-plane counts; GPU numerical equivalence is
pending. Compiler flags and tuning policy remain unchanged.

Costs include more activation requests and fused coordination, smaller aggregate
weight lookahead, larger router partial storage and possible QKV tail effects.
No cache-hit assumption, bandwidth floor or occupancy number predicts a gain.
R16 also makes the inherited generic UV mode1 compile-time geometry guard true
for QKV; QKV still calls mode0, but that compiled runtime branch can affect
register allocation. The actual admission gate decides; no rescue specialization
is part of this fixed attempt.

Preparation emits bounded `R16R` host records before any dense self-test/tuner
launch. Fourteen admission calls cover twelve distinct compiled pointers because
pooler32 and QKV32 share the two PK templates; both runtime N/pmax combinations
are checked. Every actual packed pmax participates in the exact host shared-size
formula and CUDA occupancy query. All API calls must succeed, local allocation
must be zero, launch limits must admit the geometry, and primary targets require
at least two CTAs/SM. Ordinary pooler16 and uniform fused32 compatibility require
valid admission (at least one). Their offline selection conditions remain those
of the implementation plan. A backend exception caught internally becomes a
fatal outer preparation failure. The log is capped at8192 bytes; `R16R_STOP`
means incomplete evidence, and resource enforcement continues after that marker.
These are preflight records, not evidence of benchmark selection.

CPU evidence: `experiments/iterations/20261104-sk2-r16-cpu.json`, SHA
`f5d75febd9e083563bd42e4edbd73cd93657ffe282e0bc7cc95bda32fbd54c57`.
Reproduce with:

```
/tmp/okc-kperm-cpu-20261104/bin/python tools/test_sk2_r16_cpu.py
```

The combined check restores exact parent bytes from the deterministic builder,
checks unaffected embedded sources/generated dispatch/compiler flags, executes
actual stream/table helpers (241664 unit mappings,98304 decoded BF16 words,
33408 escape entries), and compiles the unchanged decode8 with CPU intrinsic
models for3964928 BF16 words. Actual extracted host preflight code exercises all
14 records, all six API failures, resource rejection and continued enforcement
after log exhaustion. Actual catch/outer-gate checks reject hidden failures.
Ring checks cover224 interleaved schedules plus every physical M1..TOKR byte
extent, including NS5/NESC4 QKV8 and NS3/NESC3 pooler16. The source-identical
handoff protocol retains its inherited assumptions.

The independent native router helper executes40320 arrival permutations,
41664 calls and43450368 partial stores, including ragged tokens, UV u/v,
mixed TOKR calls and32-bit counter wrap. It checks all router hi/lo K addresses
and fixed piece0..7 addition order. Old four-way counters, stale partial strides,
undersized rings and premature modeled handbacks are rejected negative controls.
CPU models assume publication semantics; they do not prove CUDA compilation,
device races, numerical GPU correctness, actual registers/occupancy or speed.

The independent source critic approved one private check, and root independently
reran the complete CPU suite with exit0. It regenerated the identical full report
hash above; `20261104-sk2-r16-root-cpu.json` is the separate root summary.
The combined report's router evidence pins the final464115 source. The standalone
router-native report preserves an earlier094ddc92 source review; do not treat
that standalone artifact as the final-source proof.

The one private whole-decoder check is now submitted. Require all13 tests/12 timed cases/12 profiles,
complete clean resource evidence and intended geometry/fusion selection. No
nearby-parameter rescue or duplicate retry follows a failed fixed attempt.
Provisional promotion requires at least2% aggregate over both exact97af baselines,
positive affected-load aggregate, no affected case more than3% slower after
baseline normalization, and control-load geometric drift within3%. Diagnostic
e444 timings are not a comparator. Any promotion remains contingent on the
full recorded decision contract and a separately verified scored result.
