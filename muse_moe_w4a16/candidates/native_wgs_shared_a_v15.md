# Shared-A WGMMA v15 — fixed experiment, November 4, 2026

Candidate: `native_wgs_shared_a_v15.py`, SHA256
`93c2d0fbfe1d37792389402b335d34f8e00328e84ce25ad8ea732b9726d08a38`.
Parent: scored standalone MoE source
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
This isolated candidate does not replace either active submission.

Official private check `100bb9bbce8f4539` was submitted November4 at12:46 UTC
and acknowledged as queued. There is no GPU correctness or timing result yet.

The root accepted the fixed shared-A plan before implementation, following
independent arithmetic/lifetime review and two independently reproduced CPU
models. The plan is `tools/WGS_SHARED_A_PLAN.md`; the later store-service
sensitivity paragraph records the explicitly accepted cost risk. This is one
discrete experiment, not a parameter search.

## Hypothesis and fixed changes

The same512 consumer threads decode the same exact NVFP4/BF16 operands into
two WG-private shared A banks, then issue shared/shared WGMMA instead of
register/shared WGMMA. Removing the asynchronous A-register lifetime may improve
compiler allocation/scheduling. Parent serialization is **unknown**; previous
C7511 observations were on failed WAIT2/REG112 variants, not this parent.

Keep GS2, wait1, FCI3, FT64,576 threads, output/K work units, jobs, KPERM packing,
scale construction, zero-D positions, NTT1 even/odd accumulators, epilogues,
fixups, route/gather work, PDL and finalizer. Non-WGS preprocessed builds must
remain identical. Raw weight/X rings each shrink from3 to2 slots; the appended
32768-byte A region makes total dynamic shared194304 bytes rather than231168.
No register cap, launch-bound change, quota redistribution or fallback is added.

Each unchanged uint32 BF16 pair is stored in the canonical no-swizzle K-major
m64k16 layout. A per-WG pre-overwrite barrier follows prior wait1 retirement;
per-thread proxy fences and a separate publication barrier precede WGMMA.
The existing producer completion/empty-barrier protocol remains intact with
depth-derived indices. Both BIG kernel specializations receive host attribute
and occupancy queries once per WGS module; API success and at least one admitted
CTA are required. Registers/local bytes are reported, not automatic rejection
criteria. Resource failure propagates before launch and cannot mark the module
initialized. WGS compilation verbosity is retained for diagnostic evidence.

## Cost and practical scope

The new path adds96MiB shared stores and96MiB logical tensor A reads per complete
job, plus49152 collective WG barriers. Weight bytes and tensor arithmetic do not
decrease; raw/X lookahead drops from3 to2. At the explicit ideal128B/SM/cycle
model over132 SMs and a saved1980MHz endpoint clock sample, stores require
3.009µs/job of service, or72.22–87.26µs for24–29 jobs. This can overlap other work;
it is not an additive latency prediction or a kernel clock measurement.

The intended standalone changes affect M32/128/256; M1/8/16 are unchanged
controls. Conditional profile scenarios suggest a roughly20% improvement in
the affected decoder MoE rows would matter to whole-decoder performance. They
are prioritization scenarios, not predicted gains or overlap-adjusted bounds.

## Validation and its limits

Reproducer: `tools/build_wgs_shared_a_candidate.py` builds exactly from the
immutable parent snapshot. Combined verification:

```
/tmp/okc-kperm-cpu-20261104/bin/python tools/test_wgs_shared_a_cpu.py \
  --expected-sha 93c2d0fbfe1d37792389402b335d34f8e00328e84ce25ad8ea732b9726d08a38
```

The source proof restores all parent CUDA bytes outside the explicitly changed
regions, preserves41 Python nodes, and compares16 non-WGS preprocessed builds.
It checks all16 SS instruction wrappers,16384 BF16 coordinates, both A-bank
descriptors, all65536 possible BF16 patterns in each pair position, and an
actual-source native trace of1,462,272 MMA calls over NTT1..8, both PB phases,
starts0..5, segments1..16 and seven representative threads. Layout coverage
separately includes every lane/warp/WG. The parent and candidate operand/order
traces match; CUDA arithmetic instructions are stubbed, not emulated.

Actual preparation helpers are executed with a bounded NumPy adapter and
their full W13/W2 packed bytes are independently decoded in both bit layouts.
Host CUDA API mocks test success, each error, zero occupancy, local allocation
without rejection, first-call failure propagation, retry and cached success.
Native negative controls reject wrong register transport, wrong descriptor
steps and one A bank. The independent critic's separate source-bound model
interleaves joined W/X completions, four WGs and all16 warp handbacks, including
phase changes, partial ranges, drains and malformed-lifetime controls.

These checks do not prove CUDA compilation, GPU numerical correctness, hardware
memory ordering, asynchronous overlap or speed. Empty per-CTA ranges are
modeled; arbitrary global zero-local-job routing is not established by these
tests. The inherited range implementation has no explicit global nj0 guard;
the supplied problem generator uses the busiest rank and excludes zero-local
reference outputs. No unrelated routing change is included.

## Predetermined decision

Independent full source review approved one check (source-critic SHA4f53a826).
Root's separate combined CPU run passed and exactly reproduced author report
SHA57400109290853f0b52b52bebd2ffcd5814ce068134df90e2ef32bf5e8b244f0.
The separate root report is `20261104-wgs-shared-a-root-cpu.json`.
One private official check is now queued; only root polls that existing handle.
Require all5 tests/6 benchmarks/6 profiles and clean evaluator checks/flags.
Compare against exact-parent public `d24dd50196cb4158` and private
`bcb761dfca264db0`: affected-case geometric speedup must be at least3% over both;
no affected case may regress over3% normalized latency; control geometric drift
must stay within3%. Compiler/attribute records explain the result but cannot
establish the parent's unknown serialization state.

Failure or no useful measured gain closes this fixed experiment. Only a useful
standalone result permits a separately reviewed decoder97af integration and
whole-decoder private check. No nearby bank/depth/quota rescue is authorized.
