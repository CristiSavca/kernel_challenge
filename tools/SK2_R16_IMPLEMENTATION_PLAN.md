# Fixed SK2 R16 implementation plan — November 4, 2026

Root authorizes reversible CPU implementation of one isolated candidate from
the reviewed feasibility study `bc4990dab4d8480359b2fc79b0b67ce53bf3555396c94fff8ffecaa117c11466`.
The independent feasibility verdict is `f0aac72108ae55e4e56ffe9c22e9224f57ebb179add46f2a9c22f1a213c79c25`.
This is the concrete scope and decision contract, not a new parameter search.
Independent review of this plan may correct defects during CPU work; final frozen
source and root verification are required before the one official private check.
No Modal allocation or scored submission is authorized by this document alone.

## Objective and fixed scope

Test whether smaller output-row units provide useful independent SK2 pipelines
while preserving weight bytes, precision, full computation and existing fusion.
Parent is exact decoder97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2.
Writer: attention_paper_reset. Candidate:
`muse_decode_layer/candidates/native_sk2_r16_v14.py`.
Do not edit the public submission or official evaluator files.

Implement the whole fixed A/B/C family as one row-granularity hypothesis:

| Family | R / NT / NWR | WSK / UPW | NS by TOKR8 /16 /32 |
| --- | --- | --- | --- |
| A: every fused pooler/router specialization |16 /2 /1|8 /1|4 /3 /2|
| B: separate pooler TOKR32 |16 /2 /1|4 /2|unchanged /unchanged /2|
| C: QKV |16 /2 /1|4 /2|5 /3 /2|

Keep existing configuration indices and all unrelated configuration values.
Names may change consistently for the modified entries; the fused-pooler sentinel
and every generated dispatch must agree. Preserve the original compression,
escape-table representation, decode8, MMA primitives, precision, compiler option
lists, tuning algorithms/search policy, PIN_PICK, PDL placement, attention, MoE,
O/UV generation, unpooler/BND and large-T implementations. Geometry-dependent
packing/metadata changes are necessary; numerical weight values must be identical.
Use a distinct extension cache namespace. No quota/register-cap changes or fallback
to a different algorithm. New instrumentation is limited to bounded host resource
admission facts for exact new specializations; no GPU timing instrumentation.

## Required source changes and invariants

1. In device ring layout, host launch allocation and Python preparation geometry,
   reserve `align1024(XBY + max(WBY, WBYR))` when FUSE is true. Non-fused layouts
   retain `align1024(XBY+WBY)`. Fixed NS replaces cap-derived NS only for changed
   configurations. Match packing descriptors and generated SM90 dispatch, including
   ordinary, fused, UV, ragged PK and tuner-only TOKR32 paths.
2. Every FUSE specialization uses eight router K pieces: NSTR2, rt=cta>>3,
   rq=cta&7, global stage=rq*2+j. Preserve 32 router row tiles, 16 K512 stages,
   eight K64 warps/stage and all BF16 hi/lo terms. Keep COMP/CL1/UPW1/UPS8/NWR1
   invariants. Router weights and their stream stride are unchanged.
3. Both ordinary and UV pooler allocations use FP32 `part[32][8][32][8]`.
   Writers/readers use the matching eight-piece strides. Last arrival is
   `(old&7)==7`; sum slots0..7 in explicit fixed order. All shared-counter users,
   including prepare/tuner calls, use the same eight-way convention. Never reset
   the counters in timed calls. Preserve release/acquire and wrap semantics.
4. Preserve every escape warp's observation and handback of every stage, including
   router suffixes; exact transaction sizes remain XBYr+WBY or XBYr+WBYR.
   Keep cbar and named barriers before ring reuse. Non-last contributors do not
   spin. Preserve PDL wait/trigger positions and the UV statistics lifetime.
5. UV mode2 uses physical M=2T and paired rows. Reduce u and v separately across
   all eight pieces before the original combination and rounding. Preserve all
   model operations. FP32 association changes are explicit, not bit-equivalence.

The feasibility report records substantial costs: more activation requests,
more fused-stage coordination, smaller total weight lookahead and possible QKV
tail effects. They are accepted experimental risks, not predicted improvements.

## Source and CPU verification

Review the entire diff against frozen97af, not just generated filenames. Verify
unaffected embedded device sources and launch/tuning paths remain unchanged.
Enumerate generated target templates and their packing configuration; cover each
reachable PK variant and FUSE TOKR8/16/32. Check actual patched expressions for
SST, transaction extents, partial indices and both allocations. Include negative
controls that detect an old four-way counter user, stale stride or undersized ring.

Use actual extracted/compiled snippets where practical to check lane/row/K coverage,
lossless packed-weight reconstruction and hi/lo router contributions, all eight
arrival permutations (40320), counter wrap, repeated/mixed-TOKR calls, UV paired
rows and ragged M. Check shared-ring and reduction extents at actual metadata
bounds. CPU schedules assume publication ordering and do not prove GPU memory
semantics. Retain source hashes and concise proof artifacts. Root reruns the
combined verification; independent critic reviews the frozen complete source.

## Compiled resource and official correctness gate

Use exact compiled pointers, actual threads and actual prepared pmax/dynamic
shared size. Query all reachable new PK specializations and retain bounded facts
before any assertion. Preflight failures must not be swallowed by tuner exception
handlers or leave an unreported resource failure looking like success. Keep host
queries out of graph replay; do not inspect harness or phase identity.

All queried APIs must succeed, local allocation must be zero, and every executed
specialization must have valid admission. Intended targets A8/A16/B32/C8/C16/C32
require at least two blocks/SM. Primary register-only ceilings are72/80/128/96/128/128
respectively; actual CUDA occupancy at the real shared size is authoritative.
Uniform FUSE32 compatibility may admit one block with zero local allocation, but
only if offline timed profiles show it was not selected. If selected with one,
reject the mechanism gate. Its numerical and lifecycle correctness is mandatory.
The shared pooler configuration also changes ordinary FUSE=false TOKR16; include
its exact PK0/PK1 pointers in preflight. This is a compatibility path with valid
admission and zero local allocation required. Selection in place of the planned
fused path fails the fusion/activation gate; it cannot silently substitute for A16.
Existing BND admission remains a separate correctness requirement.

After source/CPU/critic gates, root may submit this exact candidate to one official
private check. Require all13 correctness tests,12 timed cases and12 profiles,
clean checks/no flags and successful full preparation. Verify intended changed
templates are selected and the original UV/router fusion remains present.
Missing resources, ambiguous activation or tuner fallback do not establish a pass.
Correctness failure, resource failure, compilation failure or missing evidence
ends this fixed attempt. No nearby-parameter or quota rescue, duplicate retry,
or automatic extra diagnostic follows. Preserve the result and reconsider scope.

## Measured decision

Compare the candidate to exact97af scored7a8c4941741645ee and privateb2a773e2d7194de8,
with matching evaluator and shape metadata. Use paired-baseline speedup ratios;
report raw and harness times separately. Eight T<=32 cases are affected; four
T128/256 cases are controls. Read profiles for selected geometry/fusion and
autotuner differences; durations may contain PDL waits and overlap.
Reproduce each run's load speedup from its weighted baseline and candidate times
(sliding/full weights46/16), then the six-load geometric aggregate. Affected
aggregate means four-load geometric mean; control drift means the geometric
latency ratio of the two control loads. Neither is an unweighted case average.

Provisional promotion requires at least2% aggregate speedup over both incumbents,
positive aggregate improvement on affected cases, no affected case with more than
3% normalized latency regression, and control-case geometric drift within3% of
each comparator. Otherwise reject promotion or mark the comparison inconclusive
when controls/provenance do not support attribution. Report confidence intervals
and do not describe overlapping intervals as proof of statistical significance.
This threshold does not forecast a gain or redefine the #1 objective.

A passing useful result may support root's separate decision to score that exact
source under the user's existing submission authorization. Verify the scored
result and live identity before changing the working incumbent. The official
best remains97af until a measured replacement actually qualifies. The diagnostic
e444 score is never a baseline or promotion signal for this candidate.
