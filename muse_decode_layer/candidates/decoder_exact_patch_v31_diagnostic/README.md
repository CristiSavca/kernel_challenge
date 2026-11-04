# V31: identify why V29 compressed SK2 was not selected

This diagnostic derives only from frozen V29
`bbbc37f80aa3a5263dd0a66611bae62b2376c8b77ed67c14ba8ffd8db78e8a2c`.
V29 job `185fee4d182f41dd` passed official correctness, but its eight affected
profiles showed no COMP=true V29 kernel. Its aggregate score therefore does
not measure the proposed compressed path. Weight preparation printed normal
compressed statistics; raw sidecar units were about 0.12–0.25% of units.

Independent source inspection found no decisive defect in the 16-byte bulk
copy alignment, packed descriptor stride or uint8 sidecar ABI. Compiling the
actual host geometry expressions offline for 17 emitted configurations found
all WBY divisible by16, XBY/SST divisible by1024, and dynamic shared plus a
2048-byte allowance below232448. Those checks do not establish GPU admission
or numerical correctness.

V31 changes only observation and extension cache namespace. The CUDA device
code, packing, geometry, selftest bounds, timing calls/order, tuner decisions,
fallbacks and runtime dispatch remain unchanged. Nothing forces COMP=true.

`V31P` JSON records are produced only by existing prepare procedures:

- `selftest`: each attempted SK2 key, relative error, unchanged tolerance,
  pass/fail and bounded exception text; then the SK2 disabled-key list.
- `choice`: existing inner BFP tuner candidates/times and selected choice.
- `outer`: existing outer T<=32 projection timings, BFP/fused exceptions,
  and selected backend. `us` order is Triton, Torch, BFP, fused BFP.

Budgets are separate: selftest4096B, inner2048B, outer3072B. Earlier stages
cannot consume the outer stage's allowance. Each group emits an explicit STOP
on overflow. Nonfinite diagnostic values become JSON null without affecting
the original numerical decision.

`V31R` reports attributes and occupancy for the first observed normal launch
of each compiled SK2 template/PK pointer. Cap-only calls do not consume it.
The template tuple is MT,TOKR,NT,NWR,WSK,UPW,NSTAGE,COMP,HL,NACC,CL,PK,FUSE,BND,
output-size. Runtime sample `x` is M,N,K,pmax; `attr` is registers,local,static,
max-dynamic; `api` is attributes,occupancy status. Reports are observations,
not new admission checks or proof that this specialization won the tuner.
Clustered kernels' block occupancy is not achieved cluster occupancy. Host
records have a4096B cap, a race-free atomic first-report flag, and a mutex
around the bounded logger. Together all new logs are bounded by13312B; with
the previous2283-character log, retained-tail headroom is limited. Missing
prefixes or STOP markers mean incomplete coverage.

Falsifiable outcomes:

1. SK2 launch exceptions identify a rejected API/resource/launch path; retain
   the exact exception and compiled-pointer attributes before a specific fix.
2. Finite or nonfinite selftest failure identifies a numerical-disable path;
   do not describe it as a timing loss or relax the unchanged tolerance.
3. Selftest passes but inner/outer timing selects another backend: this is an
   honest observed tuner loss under the existing prepare procedure.
4. Successful eligibility plus selected COMP=true profiles finally permits
   examining the compressed path's timings. This diagnostic alone is never
   an automatic public promotion.
5. Incomplete or inconsistent markers leave the reason unknown; no identical
   retry, forced selection or blanket timeout increase follows.

CPU evidence checks full byte restoration to V29, complete device-literal
identity after removing the host observer, actual selftest and outer selection
blocks under directed success/error/NaN mocks, resource status/error reporting,
and independent log budgets. It does not emulate GPU execution. Root owns
one official private diagnostic after independent review; no scored submission
or changes to public incumbents are part of this package.
