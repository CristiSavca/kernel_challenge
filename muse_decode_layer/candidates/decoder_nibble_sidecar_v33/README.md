# V33: matrix-window coding with missing-nibble completion

Fixed computation predecessor: V32 `ea4470fc…`. Public97af remains untouched.
V31/V32 selftests pass; V32 removed observed local allocation but still lost
selection/performance. This is a different encoding of escaped units: no scalar
word repair and no header-dependent choice before loading primary weights.

Normal units retain the original matrix base, sign/window codes and decoder.
An escaped512-weight unit stores its original low8 bits plus literal low4 bits
of the high byte in the same768-byte primary slot. A256-byte sidecar supplies
the missing high4 bits, including sign, in the same nibble order. Header0 is
normal; bit31 plus index31 selects a sidecar unit. Both modes reconstruct the
original BF16 bits, including signed zero, subnormals, infinity and NaN payloads.
Matrices costing at least raw BF16 use the inherited alternative backends.

Primary LDS is unconditional and independent of the header. After primary,
header and activation loads, the original early handback remains. Raw mode
reads one immutable global uint2 per lane, merges nibbles and uses four PRMTs
per eight values. No repaired weight array address escapes. MMA input order,
geometry, ring footprint, barriers, PDL, fused epilogues, selection rules and
compile flags remain. All17 emitted geometry tails are16B-aligned, so the
772-byte/unit eligibility arithmetic includes the entire header cost.

This trades operations for extra work elsewhere. Compared with the old raw
fallback it reads primary LDS in raw mode and performs nibble reconstruction;
many more units now request sidecar data. Root's retained V32 matrix model at
`experiments/iterations/20261104-decoder-nibble-sidecar-v33-byte-model.json`
gives4.878–6.851% raw units,2.147–2.805% extra logical bytes versus bare primary,
and1.443–1.917% more than V32. These are requested-byte counts, not latency or
DRAM forecasts. There is no extra incumbent kernel/tuner alternative in this
bounded experiment; public/source rollback remains available.

The ten-edit builder reverses exactly to V32. The focused test reuses V29's
NumPy adapter to execute actual preparation and stream helpers, then compiles
actual value decoders and the consumer branch with checked CPU raw reads.
Coverage includes all BF16 bit patterns, every position with exceptional bits,
normal/raw modes, zero sidecar use, nonzero indices,31-bit arithmetic bounds,
raw economic cutoff/full entropy fallback,2,808 stream mappings and four
deliberate source mutations. MMA/non-COMP code and handbacks are source-identical
to V32. No CUDA compiler, register allocation or memory-order proof is claimed.

Root's pre-result gates remain: both exact97af parents; clean13 tests/12 benches/
12 profiles; all expected compressed geometries active; every captured COMP
local allocation0 and successful resource coverage for every active signature;
whole weighted six-load speed ratio>=1.025; affected four-load GM>=1.04; every
case latency ratio<=1.03; control two-load latency GM[.97,1.03]. Load weights
remain46:16. Missing activation/resources or failed numerical/performance gates
close this fixed attempt. No forced selection, parameter sweep or automatic
promotion. Marker `sk2_nibble_sidecar_v33`, logs `V33P`/`V33R`.

CPU command: `.venv-modal/bin/python muse_decode_layer/candidates/decoder_nibble_sidecar_v33/test_cpu.py`.
Import `test_cpu.py:run()` for a read-only result. Root owns the one official
private check after frozen source review and independent CPU rerun.
