# WGS final route-scalar owner reuse v19

November 4, 2026. Isolated candidate from scored standalone MoE
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
Candidate SHA256:
`e849120b4f93b8eac82ce05f127dc405bb90312fa9d516bcc3a13e43754b9863`.
Active submissions are unchanged. No GPU execution has been performed by the author.

Root accepted fixed plan `tools/WGS_FINAL_SCALAR_OWNER_FEASIBILITY_20261104.md`
at `8ecccef4` after an independent source review. The later plan wording correction
only clarifies that M128 at grid132 does not assign multiple tokens to a CTA;
the implementation and mapping are unchanged.

The existing WGSFINSS route-sum owners, threads 0–7, now compute each live route's
`topk_weight * rsqrtf(tsum / 4096 + 1e-8)` and publish it in the same shared slots
that previously held `tsum`. The 512 output threads load that scalar. The entire
FP32 reduction, route order, output arithmetic and BF16 casts remain the parent.
Same barriers, shared capacity, jobs, TMA traffic, weight encoding, ring depths,
register-A/MMA schedule, routing and PDL. Tiny/MID and non-WGS fallbacks are unchanged.

The source removes 511 repeated scalar expressions per live route. This is not a
latency forecast. It moves arithmetic onto the eight owners before the existing
barrier and could change compiler allocation; GPU checks must resolve the tradeoff.

The only additional changes are a unique extension namespace, WGS-only compiler
verbosity and the reused bounded first-launch resource helper. Marker `FSR v=1`
reports both compiled BIG pointers with actual threads, dynamic/shared/local bytes,
registers, maximum dynamic shared memory, occupancy and API statuses. Parent
geometry expects 576 threads and 231168 dynamic bytes. Set/attribute/occupancy API
errors or occupancy below 1 fail before launch; nonzero local allocation or a
serialization warning is reported, not an automatic rejection. Queries cover both
pointers during a module's first launch and do not establish that both executed.

Targeted verifier: `tools/test_wgs_final_scalar_cpu.py`. It restores every parent
byte after reversing the declared delta and checks 16 preprocessed non-WGS builds.
Actual owner/consumer snippets execute through CPU shims over all 256 route masks,
three token iterations at grid132 and a rotated-mask chunk remainder. It checks
shared-slot ownership/publication, exact FP32 coefficient bits under the shim,
output bits and unchanged scalar-load/store counts. Separate enumeration covers
M0/1/8/16/32/128/132/133/256/257/384/385/769 and all 4096 columns. Five negative
controls cover missing guard, wrong slot, changed sum order, stale consumer math
and missing barrier. Reused host mocks check failure propagation, retry and caching.

These are source/finite CPU checks. CPU `1/sqrt` and BF16-conversion shims do not
prove CUDA rsqrt, FMA, BF16 behavior or memory ordering. Earlier squares/shuffles
are preserved by source identity, not simulated. No new global-zero-job support
or GPU correctness/performance is claimed.

Author CPU verification **passed** with verifier SHA
`0d0cc7ed0adfd4a27a3ba82b086ae6ec59a8f48b8b6d56e1fa0dd3095494a31d`.
Report: `experiments/iterations/20261104-wgs-final-scalar-cpu.json`, SHA
`040c7cc7292f35bc9aca7ac754c2ed2cb8fea0dce2369e2daf3b71ad1ca73526`.
It contains 1,024 token cases, 524,288 consumer-thread pairs and 2,097,152
matching FP32 coefficient bit comparisons under the CPU shim. The resource
helper passes ten mock modes, including failure/retry and permitted local memory.

Reproduce:

```sh
python3 tools/test_wgs_final_scalar_cpu.py --expected-sha e849120b4f93b8eac82ce05f127dc405bb90312fa9d516bcc3a13e43754b9863
```

Predeclared performance screen, against **both** public parent
`d24dd50196cb4158` and private parent `bcb761dfca264db0`:

- Complete official correctness/profile coverage: 5 tests, 6 benchmarks, 6 profiles.
- MAIN M32/128/256 geometric-mean relative speed at least 1.005 against each parent.
- At least two of those three cases faster against both parents.
- Each affected case's normalized latency ratio at most 1.02 against each parent.
- Unchanged M1/8/16 control GM normalized latency ratio within [0.98, 1.02].

This is a consistency screen, not statistical significance. Report raw/harness
times separately from SGLang-relative normalized ratios. One private check follows
frozen-source review and root CPU reproduction; no depth/geometry sweep or failed
attempt retry. Decoder integration requires a useful checked result. Public
promotion additionally requires actual aggregate improvement.

## Official terminal result — November 4, 14:53 UTC

Private check `bf28693eb41f41e7` passed 5 tests, 6 benchmarks and 6 profiles
with clean checks. Score **2.3698692914**, CI [2.3343212520, 2.4054173308].
Affected M32/128/256 GM speed ratios are **0.9947040823** versus public and
**0.9856520708** versus private; zero affected cases improve against both.
Private M256 additionally exceeds the 2% normalized-latency cap. Unchanged
controls remain within 2%. The fixed screen fails; no repeat, decoder port
or public promotion. Active sources remain unchanged.

Both first-launch FSR entries report 576 threads, 231168 dynamic shared bytes,
96 registers, zero local/static bytes, occupancy capacity 1 and API success.
Retained compiler output reports zero stack/spills; no C75 is visible, but full
absence is not established. No SASS/source-PC evidence establishes arithmetic
instruction removal. The overlapping intervals do not prove a causal slowdown.
Root evidence: `experiments/iterations/20261104-wgs-final-scalar-root-result.json`.
