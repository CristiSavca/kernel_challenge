# V31 diagnostic source review — 2026-11-04

**GO for one private diagnostic check**, source
`0d552bf386fdbcae6837d97371ae74aa778709b234d993b4e341e655fe9216f7`.
No scored submission, forced activation, parameter change or promotion is approved.

The starting evidence is V29 job `185fee4d182f41dd`: official correctness passed,
but all eight affected profiles lacked COMP=true SK2. Its 1.898596 score therefore
does not measure the proposed exact-register-repair kernels. Normal compression
statistics and the COMP=false router establish preparation and some extension
execution, not compressed-kernel eligibility. The retained V29 log cannot distinguish
numerical selftest disable, caught launch failure, inner tuning loss and outer
backend selection. A focused read of sidecar ownership, the untyped ptab ABI,
descriptor/unit layout and stage alignment found no decisive source defect.

V31 has twelve reversible edits against exact V29
`bbbc37f80aa3a5263dd0a66611bae62b2376c8b77ed67c14ba8ffd8db78e8a2c`.
I reviewed the full builder, delta, host observer, Python observation sites and
focused verifier. Removing the host observer restores the complete SK2 CUDA
literal; reversing all edits restores the entire parent file. All device math,
packing, compile flags, geometry, barriers, PDL, selftest bounds, timing calls,
candidate order, disabled-key logic and selection algorithms are retained.
The extension cache namespace changes. Actual measured choices may differ across
runs; this is not a claim of identical timing samples or selected backends.

The selftest observer records an already computed error and unchanged decision,
including caught RuntimeError text. Nonfinite diagnostic values serialize to null
without changing the original decision. Inner choices are printed after their
existing timing work. Outer observations occur after the existing best-pick update;
the original BFP/FUSE exception-to-1e30 behavior and timer sequence remain intact.
No new input-derived GPU computation, explicit CUDA synchronization or admission gate is added.

The first-normal-launch observer queries the exact existing template/PK pointer
after existing attribute setup and the cap-only return. Its atomic first-report
flag and mutex protect new mutable host state. API statuses are retained, failures
do not reject or rescue a kernel, and the bounded logger stops querying after its
cap. First observed M/N/K/pmax and template attributes are not benchmark-phase or
weight-state identities. Occupancy is block capacity; CL>1 records do not establish
cluster admission, and neither resource attributes nor capacity prove achieved
occupancy, spill traffic or a performance bottleneck.

The separate caps are selftest 4096, inner 2048, outer 3072 and host 4096 bytes:
13,312 added bytes maximum. V29's existing log was 2,283 characters, so approximately
15,595 leaves limited room in the 16,000-character retained tail. STOP, missing
prefixes, missing stage records or inconsistent identities mean partial evidence;
absence of a record never proves a path passed or was not attempted. Budgets are
process/module lifetime observations and do not reset to provide exhaustive
coverage of every preparation.

Independent CPU execution of `test_cpu.py:run()` produced the exact author report
SHA `31b7e2e09a923c5daf548d1c37183bcd29e32957ef1b0c3ec84444289843327f`.
It covers complete source restoration, four actual selftest modes
(pass/numerical failure/RuntimeError/NaN), four actual outer-selection blocks with
identical timer calls/picks, separate log budgets, and a native host-helper mock
for successful/failed CUDA APIs and no queries after exhaustion. The source was
hashed before and after. No CUDA compiler/device was used; these checks do not
establish runtime eligibility or numerical correctness.

Read the eventual result causally: a selftest exception is launch/API failure
evidence; an error above the unchanged tolerance is numerical-disable evidence;
passing selftests followed by lower measured alternatives is tuner-loss evidence.
Multiple failures can coexist. Preserve exact first failure and marker coverage.
Do not weaken tolerances or turn a missing/partial log into a speed claim. Root
owns the one official diagnostic and its independent terminal audit.
