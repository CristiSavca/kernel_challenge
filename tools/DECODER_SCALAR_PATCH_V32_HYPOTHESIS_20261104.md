# V32 fixed compiler hypothesis — 2026-11-04

V31 diagnostic job `5f48e0da4dcd4595` distinguishes V29's nonactivation from a
numerical failure: all20 SK2 selftests passed, disabled=[], no retained launch
exceptions. Retained COMP resources show localSizeBytes=NT×32 (32/64/96/128),
while routers show0. The repaired array has exactly that size. Most COMP
variants lost the existing tuner. The correlation supports one representation
rewrite; it is not proof of dynamic spill traffic or the sole slowdown cause.

The isolated implementation is
`muse_decode_layer/candidates/decoder_scalar_patch_v32/submission.py`.
It retains V29's exact codec/geometry/operand order and V31's bounded observers,
but COMP uses value-return decode plus statically instantiated scalar repairs
into two uint4 values. No repaired-array address escapes. The raw sidecar and
all non-COMP computation remain. Full source restoration and actual-source
CPU word/address/MMA-argument comparisons precede one official check.

Parent public math/source is97af; V29 is the computation predecessor, not the
performance baseline. Required baselines are public7a8c4941741645ee and
privateb2a773e2d7194de8. Freeze the root's gates: whole six-load ratio>=1.025;
affected four-load GM>=1.04; every case normalized latency<=1.03; two control-load
latency GM[.97,1.03]; clean13/12/12, exact compressed geometry activation,
zero local bytes on every captured COMP pointer and a successful capacity>=1
resource match for each active compressed first-seven-field signature.
Load aggregation uses the official46:16 weights. Marker sk2_exact_patch_v32
retains V29's length so profiler prefixes remain comparable.

Failure of compiler allocation, activation, numerical acceptance or the frozen
performance screen closes this exact attempt. No precision, shape tuning,
admission, disabled-key, timing-policy or forced-selection changes are included.
Speed, resource allocation and actual benefit are unmeasured until the GPU run.
