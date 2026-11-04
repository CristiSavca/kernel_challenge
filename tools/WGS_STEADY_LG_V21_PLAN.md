# One fixed steady-loop experiment, November 4

Candidate `muse_moe_w4a16/candidates/native_wgs_steady_lg_v21.py`, SHA256
`4fcf108fc860865b0284031075c9e35461b761eb11f5cd4681917e44165871da`, changes only
the steady `lg` loop's unroll hint and extension namespace relative to scored
parent `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
The FIRST body remains unrolled. Arithmetic, AF-bank parity, waits, releases,
dispatch, weight storage, all NTT variants and non-WGS paths are unchanged.

Hypothesis: less duplicated steady-stage machine code can improve the MAIN
path enough to offset loop/address overhead. This is unproved: prior NCU
counters did not measure instruction-fetch stalls, and the full 340 KB binary
is not the executing hot set. Nominal body copies fall from eight to five per
phase/NTT; that 37.5% is not a latency prediction or total-code reduction.
Even a hypothetical 10% MAIN latency reduction affects only three of six standalone
loads (about 5.4% aggregate speedup if controls are identical). It cannot by
itself establish the 18.11% decoder speedup needed to pass the current leader.

Root reproduced the source-bound CPU report exactly: 13,440 cases and
5,871,216 model events, two detected negative controls. This proves neither
CUDA correctness nor compiler compliance with the pragma. Independent critic
review approved one official private check after inspecting the actual diff.

## Pre-result decision gate

Use public parent `d24dd50196cb4158` and private parent `bcb761dfca264db0`.
Require exact uploaded source, evaluator identity, clean setup/check flags,
all five correctness tests, six benchmarks and six profiles. For each case,
normalize time to the same run's SGLang baseline, as in prior comparisons.

- MAIN cases M32/M128/M256: geometric mean speed ratio at least 1.01 against
  each parent; at least two of three cases faster against both parents; every
  MAIN normalized latency ratio at most 1.02 against each parent.
- Unchanged controls M1/M8/M16: geometric mean normalized latency ratio in
  [0.98, 1.02] against each parent.
- Any numerical failure, rejection, missing coverage, or failed performance
  gate closes this fixed attempt. No automatic retest or unroll-factor sweep.

A passing screen is evidence for a paired confirmation and decoder integration,
not proof of causality, statistical significance, first place or prize prospects.
Official comparisons use fresh weights/nodes and retain that limitation.
Do not promote an unvalidated decoder port or replace a passing incumbent.

## Execution choice

At 17:32 UTC no official or Modal job was running, MoE had 29 private checks
left, and the kernel track remained open until November 5 08:00 UTC. Use one
official private check now. A new optional owned-SASS collector is deferred:
it is not required to test numerical correctness and net performance, and would
delay this fixed experiment. No new Modal allocation or financial adjustment.
Existing Modal paired benchmarking remains available for a useful positive result.

Latest checked standing: overall/decoder 401; own decoder score 2.1812, leader
2.5762. Public scores and sources remain unchanged. The previous interrupted
continuations made no GPU progress; source/CPU review artifacts survived and
were revalidated before this next action. Goal remains active and unachieved.
