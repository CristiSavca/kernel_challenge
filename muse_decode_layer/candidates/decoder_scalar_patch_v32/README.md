# V32: keep exact repaired BF16 words in register values

Parent computation is frozen V29 `bbbc37f8…`; observations are frozen V31
`0d552bf3…`. Public decoder97af and its source/weights remain untouched.

V31 job5f48e0da4dcd4595 established20 passing SK2 selftests, no disabled SK2
keys and no retained exceptions. The compressed paths mostly lost existing
inner/outer tuning. Every retained COMP resource record allocated NT×32 bytes
of local storage, exactly the size of `b[NT][8]`; router local allocation was0.
This identifies a concrete compiler hypothesis. It does not prove that local
traffic caused all of the slowdown, or that eliminating it is sufficient.

V32 replaces only the COMP decoded-weight representation. Two uint4 value
objects per NT hold eight BF16x2 words. Decode returns a value; repair uses
eight statically instantiated scalar word positions and returns values.
No repaired COMP storage is passed by pointer/reference. Raw sidecar loading
uses the same address and two uint4 reads. The original non-COMP b-array path
remains inside its compile-time branch; COMP cannot execute or address it.
The existing j/mt/nt/uu order feeds exactly the same words to each MMA.

Weight codec, descriptor layout, raw fallback, geometry, shared memory,
ring waits/handbacks, launch bounds, PDL, reductions and fused epilogues are
unchanged. The inherited non-COMP low-half/router code remains. V31's bounded
selftest, inner/outer tuning and first-template resource observers remain,
with V32P/V32R labels. The same-length device marker is sk2_exact_patch_v32;
COMP is template field7 (zero based). Cache namespace is unique.

`build.py` verifies parent and observation-parent hashes, replays twelve frozen
diagnostic edits, applies ten representation/identity edits, and reverses all22
to the exact parent. `test_cpu.py` compiles actual old/new integer helpers and
the new consumer raw/MMA argument blocks using CPU PRMT/LOP3/load shims. It checks
every BF16 payload, every repair position with directed exceptional bit patterns,
all32 lanes, raw unit boundaries and MMA operand ordering. Its18,350,080 word
comparisons are not an exhaustive Cartesian product of all possible tuples.
Five wrong-owner/half/word/raw-stride/MMA-pair source mutations must fail.
The prior V29 packing/ring proof is inherited by exact source preservation.
This does not prove CUDA register allocation, device ordering or performance.

The pre-result screen is fixed against BOTH exact97af public/private parents:
13 tests,12 benchmarks and12 profiles must be clean; all expected compressed
first-seven-field geometry signatures must activate in the eight affected
profiles. Captured COMP pointers must have zero local bytes; each active
compressed signature needs a matching successful API/capacity>=1 resource row.
Whole official six-load speed ratio must be>=1.025, affected four-load GM>=1.04,
every case normalized latency<=1.03, and the two control-load latency GM must
lie in[.97,1.03]. Each load retains the official46:16 sliding/full weighting.
These are screening thresholds, not statistical significance or a #1 forecast.

Nonzero local storage, missing activation/coverage, correctness errors or a
missed speed gate close this fixed attempt. No tile/distance/register quota
sweep, forced selection or tolerance relaxation is included. Root alone owns
the one official private check after independent source review and CPU rerun.
