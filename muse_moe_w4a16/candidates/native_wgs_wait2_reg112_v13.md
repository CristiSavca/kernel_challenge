# WAIT2 with one full producer warpgroup and32/112 register quotas

Status: independent exact-source review and the root's CPU rerun passed.
Root submitted one official private check, `42527fa2c64d4b42`, at10:05 UTC.
The service received it; compilation, correctness and timing remain pending.
Resume that receipt without another upload. The implementation agent made no
GPU allocation or submission, and both public incumbents remain unchanged.

Frozen source SHA256:
`31127a37ce35616b555e2ea5a5df673719aa03c422f9edc1a2be75320b3b9f65`.
CUDA text SHA256:
`492145aa8318592ba6cddabe72d69f1f1269e26a7900ffbba32c9caf651f7001`.
Base is frozen WAIT2 `97ffccc30a90ce8400cddef800d932889822eaf876fbdedaec18b1013e8f66b6`,
not the failed compact/NW4 candidates. Scored incumbent247888 remains unchanged.

The terminal WAIT2 check`f685c2533ebc4214` passed but did not improve speed.
Its retained log contains C7511 in both BIG entry points: WGMMA serialization
due to insufficient pipeline registers, despite96 registers and zero reported
stack/spill/local bytes. Parent compile diagnostics are unavailable; this is not
proof that WAIT2 introduced serialization or that this explains all timing loss.
The exact plan and independent implementation GO are preserved in
`tools/WGS_WAIT2_REGISTER_PLAN.md` and
`experiments/iterations/20261104-wgs-wait2-register-plan-critic.json`.

## Selected implementation

WGS alone launches640 threads:512 consumers and one complete128-thread producer
warpgroup. Every producer thread executes dec32 before branching. Warps16/17
run the original weight/activation producers; idle warps18/19 join barrier2 once
and return. The explicit consumer `else` starts with inc112 and encloses the
entire unchanged consumer body through finalization. This establishes source
dominance without introducing an outlined helper or a common live math merge.

Barrier2 grows to640; barrier7 stays544, compute barriers512 and consumer-WG
barriers128. FT64,256row/K256 stages, three weight/x slots,231,168 dynamic shared
bytes, one-CTA/SM grid, packing, scratch, counters and every numerical operation
remain unchanged. `wgs_seg` is byte-identical to WAIT2, including AF4/wait2,
ci1 release, NTT1 parity, FCI3 and terminal wait0. A comment outside that function
identifies its retained576-thread comment as historical parent context.

Before either ordinary or PDL launch, both compiled specializations must report
exactly96 initial registers, local0 and occupancy≥1. With actual initial96,
640×96 =512×112+128×32 =61,440 registers. A lower initial allocation is rejected;
the source does not borrow unused SM registers or select another quota. The
existing WGS-only verbose compile output remains enabled, with a new resource
marker and unique extension namespace.

## CPU evidence and limits

`python3 tools/test_wgs_reg112_cpu.py` checks exact CUDA scope restoration and
Python AST preservation,16 unchanged non-WGS preprocessed builds, the actual
extracted role prefix for all640 IDs under six repeated job labels (3,840 cases),
and3,969 combinations of actual host admission predicates. Job labels do not
simulate routing; role dispatch has no job-dependent input. The test executes
the unchanged actual WAIT2 segment using its existing deferred-pointer stubs:
1,462,272 MMA calls,731,136 groups,10,752 flushes and39,168 x releases match the
retained parent trace. Evidence is
`experiments/iterations/20261104-wgs-reg112-source-cpu.json`.

The separate plan model covers all120 WG request orders and rejects partial
producer participation, insufficient initial88 allocation and missing idle
joins. The independent pinned NVIDIA-header model establishes total initial
allocation/admission; it does not establish actual role placement in hardware
register partitions. These CPU checks do not compile CUDA, emulate tensor math,
prove dynamic register hints, simulate CUDA scheduling or measure performance.

## Bounded official decision

Root owns any official private check after exact-source review. Compile/resource
failure, dropped hints, any retained WGMMA serialization including C7511/C7512,
or visible spills kills this chosen32/112 attempt. Missing/truncated compiler
output cannot establish warning absence. Initial96/local0/occupancy1 are
necessary screens, not proof that consumers actually used112 registers. Retained
PTX/SASS can strengthen mechanism evidence if official artifacts expose it;
the candidate must not spawn inspection tools.

If these screens and all official correctness tests pass, require geometric mean
`incumbent_speedup / candidate_speedup <=0.95` across M32/M128/M256, no case ratio
above1.03 and stable tiny/MID controls. Separately report harness/raw time ratios
and variation between GPU runs. Compare failed WAIT2 as a secondary control.
No quota sweep, automatic retry, scoring or decoder integration is authorized
by a CPU pass. Any useful standalone result needs separately reviewed whole
decoder verification before promotion.
