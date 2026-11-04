# Fixed distance3 bulk L2 weight prefetch

2026-11-04. Isolated candidate from exact scored KPERM parent247888; no SC6,
encoding, geometry, arithmetic or pipeline changes. Root accepted fixed plan
`tools/WGS_BULK_L2_PREFETCH_FEASIBILITY_20261104.md` (57bad98b) after critic review.
Independent frozen-source review and root CPU reproduction passed. Official
private check `ea811fae5f6648bf` passed correctness but failed the fixed performance
screen. This attempt and its conditional decoder port are closed. Active
submissions remain unchanged.

Source SHA256: `85da23922def6696b206a227af944c13cb0f439c90bf4fa4570c39ca92afe759`.
Parent: `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`;
comparator public `d24dd50196cb4158`, private `bcb761dfca264db0`.

The hypothesis is that a plain bulk L2 hint may start fetching a weight stage
before a shared ring slot is free for its real copy. Lane0 of the weight producer
issues the hint immediately after its current real TMA, only if the third future
stage belongs to the same phase range. Its address follows the actual rotated
phase-A order and ordinary phase-B order, including job/expert changes. The last
three stages of each phase issue no hints; empty/short ranges issue none.

The real copy source/destination/36864-byte count, joined mbarrier bytes, waits,
ring parity, activation producer and compute are unchanged. No hint contributes
to data readiness. Fixed576 threads,231168 dynamic shared bytes, GS2/wait1,
three weight/X slots, NTT1 dual accumulation, FCI3, costs, all packing and Python
dispatch remain the parent. Only the weight-producer block, reused first-launch
LPFR resource helper, unique extension name and WGS compiler verbosity change.
The helper records both BIG specializations and requires successful APIs and
occupancy>=1; register/local bytes are evidence rather than automatic rejection.

This adds nearly one hint per weight stage, approaching50% more bulk requests
than the combined weight+activation copies. Those are requested cache bytes,
not necessarily duplicate DRAM bytes. The hint can be ignored, arrive late,
cause eviction or compete with demand traffic. New address work may affect
compiler allocation. Existing DRAM utilization does not establish a latency
bottleneck, and no speedup is predicted.

CPU report `experiments/iterations/20261104-wgs-l2-prefetch-cpu.json` is PASS,
SHA `79ba61d2084fa4a36c09ca77358531cb783b87c11b0331a2af0763b26f9e1b4c`.
Verifier `tools/test_wgs_l2_prefetch_cpu.py` is frozen at
`4583f2e913d83678cb6412ccd478580e85de2d083d8e4f6a854d54a546ac92f3`;
builder `tools/build_wgs_l2_prefetch_candidate.py` is `e1cb4519`.

- Full CUDA restoration to the parent after removing the declared delta;
  other Python nodes and16 non-WGS preprocessed builds match exactly.
- Native execution uses the actual producer and split functions, replacing
  CUDA operations with trace shims and byte pointers with integer addresses.
  All real copies, waits and arrivals equal the parent trace. Every hint follows
  its current real TMA and names the phase's third future actual weight address.
- Cases cover both phases, empty/short/ragged ranges, reordered wrap, large
  job boundaries, three expert mappings, shared-ring parity wraps and inactive
  producer lanes. Full-stage addresses stay aligned and within packed banks.
  The harness uses `jE[256]`; directed cases beginning at65519 extend integer
  address testing beyond the real FT64 JMAX88 routing domain. They are not legal
  routed workloads. The verifier's "greatest legal job-array region" comment
  refers only to its enlarged CPU fixture and must not be read as GPU coverage.
- Six source mutations fail: missing tail guard, wrong rotated wrap, current
  rather than future expert, changed expected bytes, changed real copy bytes,
  and wrong prefetch distance. Reused native host mocks cover API errors, zero
  occupancy, allowed local allocation, failure propagation/retry and caching.

Reproduce with `python3 tools/test_wgs_l2_prefetch_cpu.py --expected-sha
85da23922def6696b206a227af944c13cb0f439c90bf4fa4570c39ca92afe759`.
The finite CPU trace proves neither GPU asynchrony nor execution/effectiveness
of a cache hint. CUDA compilation, official numerical checks and performance
remain pending. No GPU call was made by the author.

Root's fixed screen uses both exact247888 comparator receipts: MAIN M32/128/256
GM relative speed>=1.01, at least two of three positive against both, and each
normalized latency<=1.02. Control M1/8/16 GM normalized latency must be within
[0.98,1.02], with all five tests/six benchmarks clean. No distance, cache-policy
or SC6 combination sweep follows a failed result. A useful standalone outcome
requires a separately reviewed decoder integration and complete decoder check.

## Terminal outcome

Official result observed13:57 UTC:5 tests,6 benchmarks and6 profiles passed;
clean checks and no flags. Score2.2783223373, CI[2.2441475022,2.3124971723].
Affected-case normalized latency GM increased9.0334% against public247888 and
10.0348% against private247888. All three affected cases slowed against both;
zero joint positives, and every affected case exceeded the2% slowdown cap.
Controls stayed close: latency GM0.9955393 and0.9999446. Aggregate score ratios
were0.9598226 and0.9533384. Both baseline score intervals are above this interval.

Both LPFR entries report576 threads,231168 dynamic shared bytes,96 registers,
zero local/static bytes, one-CTA capacity and successful APIs. Both compiled
entries report zero stack/spill stores/spill loads. No C75 diagnostic is visible;
its complete absence is not established. Pinned source, compile flags and logs
support MAIN activation by inference, not direct per-case module-pointer tracing.
The result does not measure whether hardware honored the hints or isolate cache
pollution, request contention or address overhead as the cause.

Root and independent critic`c69406d2` agree to reject. Retain performance,
resource, root-result and result-critic JSON under
`experiments/iterations/20261104-wgs-l2-prefetch-*`. No retry, distance/policy
sweep, SC6 combination, decoder port or scored promotion follows this result.
