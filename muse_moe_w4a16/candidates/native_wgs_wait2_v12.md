# WGS wait2 with four register-A banks

November 4, 2026. One isolated implementation from scored standalone KPERM.
Final independent source review and the root's CPU rerun passed. Root submitted
one official private check, `a0dd88ebc8964b3f`. At 09:32 UTC it returned an
infrastructure error with no measurements; the service refunded it and explicitly
permitted resubmission. Root submitted one recovery, `f685c2533ebc4214`, from
the exact same immutable source. It is terminal PASS, but rejected for performance.
All five tests and six timed cases passed without flags; speedup is2.331989071
versus the scored parent's2.373691044. No active submission was changed.

The affected M32/M128/M256 cases lose geometric-mean SGLang-relative speedup:
4.54% against the public parent and5.41% against its private check. Equivalently,
inverse-speedup latency ratios rise4.76%/5.72%; these are not direct cross-run
median_ns changes. All three raw medians are slower against both parents.
Unchanged controls improve approximately1.11%/0.66%. The prespecified gain
gate fails. Do not score, integrate or repeat this measured source.

The retained compiler output explicitly reports **C7511** for both BIG variants:
WGMMA is serialized due to insufficient register resources. Both use96 registers,
zero stack/spill/local bytes, and admit one CTA at576threads/231168dynamic bytes.
This is positive serialization evidence despite zero spills. Parent diagnostics
are incomplete, so it does not establish that serialization was newly introduced.
See `tools/analyze_wgs_wait2_result.py` and
`experiments/iterations/20261104-wgs-wait2-official-analysis.json`.
A separate register-redistribution design is under review; it has no measured gain.

| Artifact | SHA-256 |
| --- | --- |
| Parent `native_wgs_kperm3_v9.py` | `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0` |
| Candidate `native_wgs_wait2_v12.py` | `97ffccc30a90ce8400cddef800d932889822eaf876fbdedaec18b1013e8f66b6` |
| Candidate CUDA string | `d6c91c05cae6e5e9d4bd6455062678b4ee86f96affe6c9a4c475f2c0c3984658` |
| Verifier `tools/test_wgs_wait2_cpu.py` | `56b0a0b4cec85f17e0ee9e5f19d4540616e71ae536ce0ed13e68c0195ab8c6b7` |

The current adaptation and inherited KPERM provenance are separated in the
header. Original licenses and upstream notices remain intact.

## Change and expected mechanism

WGS commits the same two k16 MMAs per group, then permits two groups to remain
pending instead of one. Four constant-index AF banks (`ci & 3`) avoid reuse
across the eight-group stage boundary. The previous activation slot is released
after ci1's wait2, when both trailing groups of the previous stage have retired.
Every segment still drains with wait0 before generic accumulator access, final
x release, partial stores or epilogues. NTT1 retains its original even/odd
accumulator partition and first-write flags; FCI3 remains unchanged.

The opportunity is more overlap between operand construction and outstanding
MMAs. It is not an increase in tensor instruction count or a reduction in bytes.
All 256-row/K256 stages, FT64 jobs, three TMA rings, 576-thread launches, one
CTA/SM, packing, routing, global scratch, math and finalization remain inherited.
All small/MID/fallback flags remain identical. There is no automatic sweep.

WGS-only host diagnostics report both compiled specializations' registers,
local/static bytes and occupancy. Both must admit at least one CTA at the actual
thread/shared allocation before ordinary or PDL launch. Only WGS compilation
is verbose so ptxas diagnostics can be reviewed; missing/truncated logs cannot
establish absence of serialization. The extension namespace is unique.

## Verified on CPU

`experiments/iterations/20261104-wgs-wait2-source-cpu.json` records the completed
author run. The verifier normalizes only the intended WGS segment edits and
then requires exact parent source equality. After removing the new host marker,
all CUDA outside that segment is byte-identical. Every top-level Python AST node
except CUDA_SRC and the explicitly checked extension name/verbosity is unchanged.
Sixteen preprocessed non-WGS configurations match their parent byte-for-byte.

Native C++ runs the actual extracted segment through its final wait0/x release,
with instrumented CPU stubs retaining operand addresses until modeled group
completion. Across NTT1–8, both phases, lengths1–16, six initial ring positions/
generations and seven representative thread positions, it checks1,462,272 MMA
calls,731,136 commits,10,752 pending-work flushes and39,168 x releases. Complete
operand/descriptor/accumulator/zero-D traces match the parent. The two local
accumulator arrays are initialized only in the CPU harness because tensor
arithmetic itself is stubbed; this is not a GPU numerical proof.

The separate plan model covers13,872 cases and3,662,208 groups with conservative
two-group retention, all segment tails and zero-length ranges. Deliberately
unsafe two-bank, reset-three-bank and early-x-release controls fail. Evidence:
`experiments/iterations/20261104-wgs-wait2-lifetimes.json`.

## Resource risk and decision gate

Hopper's actual576-thread register-admission cap is96 registers/thread, already
used by the parent. Aggregate division suggesting112 is misleading. Four source
banks declare32 AF words; the modeled live peak is24 versus16 in the parent,
but compiler allocation may differ. Wider NTT accumulators and FCI3 deferred
work add pressure. The previous NTT1-only GS4 binary's96-register/local0 result
does not establish that this new source fits efficiently. C7512 serialization,
outlining and spills are unresolved risks.

The observed8.11/9.58% aggregate `stall_wait` counters describe fixed-latency
execution dependencies and lack source-PC attribution. They are not measured
WGMMA-wait time or a predicted speedup. Full rationale, primary documentation
and critic-reviewed scope are in `tools/WGS_WAIT2_PLAN.md`.

Require official numerical correctness and actual resources before any claim.
The bounded performance gate is at least5% lower normalized geometric-mean
latency at M32/M128/M256, no affected case over3% worse, and stable tiny/MID
controls. A failure does not trigger a depth/bank sweep. A useful standalone
result still needs separate integration/review and a full decoder check; this
candidate has no established #1 claim. Root alone submits and polls.
