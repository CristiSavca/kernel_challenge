# V29 terminal result — close the fixed attempt

**The official run passed numerics, but the intended compressed kernel never appeared in the eight affected timed profiles.** This is an activation failure and an end-to-end regression, not a measured rejection of the register-repair kernel’s speed. No retry, rescue or public promotion follows.

Job `185fee4d182f41dd` used exact frozen source `bbbc37f80aa3a5263dd0a66611bae62b2376c8b77ed67c14ba8ffd8db78e8a2c`. The raw receipt, extracted profiles and local source match; setup/13 tests/12 benchmarks/12 profiles passed with clean flags and the exact parent evaluator/image. I independently reproduced the official six-load 46:16 score and every paired ratio: **1.8985960692**, CI [1.8701171281, 1.9270750102].

| Measure | Versus public97af | Versus private97af |
|---|---:|---:|
| Whole score ratio | 0.870443 | 0.872254 |
| Affected four-load speed GM | 0.818498 | 0.820504 |
| Control two-load latency GM | 1.015814 | 1.014454 |

All eight affected cases are slower than both parents and exceed the fixed 3% per-case cap. The controls satisfy their GM and individual caps. Reported aggregate confidence intervals are disjoint below both parents; this still does not identify a compiler or kernel cause.

| Weighted load | Normalized latency vs public | vs private |
|---|---:|---:|
| decode 1 | 1.3247 | 1.3384 |
| decode 16 | 1.2193 | 1.2025 |
| decode 32 | 1.1319 | 1.1223 |
| verify 1 | 1.2186 | 1.2215 |
| verify 16 | 1.0109 | 1.0137 |
| verify 32 | 1.0208 | 1.0152 |

Expected compressed-instance counts were 3, 3, 4, 4, 4, 4, 3, 3; observed counts are zero in every affected profile. Only the renamed **COMP=false router** appears at decode32 sliding/full. The other dense paths expose `_gemm_splitk` and cuBLAS kernels with separate boundary/routing work. Consequently the official numerical pass also does not establish isolated GPU numerical correctness of the new compressed repair path.

The four retained preparation lines show that codec byte eligibility was plausible:

| Matrix | Units | Single repairs | Raw units | Raw unit rate | Bytes above primary |
|---|---:|---:|---:|---:|---:|
| o | 16,384 | 910 | 21 | 0.1282% | 0.6917% |
| pooler | 65,536 | 4,278 | 164 | 0.2502% | 0.8545% |
| qkv_gate | 36,864 | 2,473 | 92 | 0.2496% | 0.8536% |
| unpooler | 65,536 | 3,128 | 80 | 0.1221% | 0.6836% |

Every retained matrix is below the whole-raw requested-byte cutoff. The lines do not identify their prepare generation or benchmark case, so they are not a complete per-case fallback census. The 327-byte saved log is the exact prefix of the 2,283-byte receipt tail; the remainder contains profiler start/stop messages. No COMP resource attributes, per-path disabled reasons, selftest errors or tuner timings were retained.

The same generated extension supplies the observed renamed router, which supports successful module compilation/loading. It does not prove that compressed specializations launched successfully. The unchanged `bfp__selftest` silently disables paths on a launch exception or excessive relative error; nested tuning may also choose alternatives. This evidence cannot distinguish those cases. Claiming that register repair, sidecar traffic, reduced occupancy or spills caused the slowdown would be unsupported.

The stronger lesson is a selection-contract failure: exact CPU bit reconstruction did not establish that the new backend remained eligible and selected on the GPU. Our fixed activation requirement caught that failure. A future separately reviewed dense architecture needs retained prepared/eligible/selected-state evidence before its timed result can answer an architectural question; this does not reopen V29.

The next distinct experiment is the already separate V30 whole-K MoE design, which removes Stream-K partial accumulation and cross-phase readiness waits while preserving the parent weight format. Its real cold two-CTA gate, numerical checks and paired speed thresholds must stand independently. V29 contributes no expected V30 gain. Public97af remains unchanged.

The [machine-readable critic](../experiments/iterations/20261104-decoder-exact-patch-v29-result-critic.json) retains exact pins, all twelve paired cases, packing counts and uncertainty.
