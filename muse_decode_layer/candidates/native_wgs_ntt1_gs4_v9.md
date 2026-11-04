# Four-MMA issue groups for single-tile WGS jobs

Status: **rejected for promotion and decoder integration**. The standalone
counterpart compiled and passed 80/80 local checks, but failed the performance
criterion. The decoder counterpart remains GPU-untested. No official check or
scored submission was spent on either candidate; no further GS4 run is planned.

| Source | SHA-256 |
|---|---|
| Decoder `native_wgs_ntt1_gs4_v9.py` | `acfbdc2e5aead53ee4a9f8bf52a289f0942dd20a5e793de2fb39593f8cafd3f5` |
| Standalone MoE `native_wgs_ntt1_gs4_v7.py` | `7f24e2857e1b419fb528e8031c7c7cbb1cb6233e9fbae46b7efdcad88c390847` |

Parents are exact frozen decoder `80a4eaa3...` and standalone MoE `ef7b210b...`.
Active submissions and the completed diagnostic driver remain unchanged.
Both candidates retain full upstream notices and use unique extension names.

## Completed comparison and independent decision

Root ran the standalone counterpart against frozen `ef7b210b...` on one H200:
`experiments/modal/20261104T045617Z-6354a177/result.json` (SHA-256
`af277c1d5a96af9830f223f222c52b84d09311893f31f7b2bdd5d1ac83e6ef36`).
All 80 eager/graph checks passed. The run took 379.77 seconds and terminated
successfully; prepare including weight cloning took 187.12/185.19 seconds for
baseline/candidate. The 128 MiB read-sweep selfcheck passed. It verifies the
observable loads, not cold DRAM traffic.

Six fresh paired graph replays per case produced these median ratios
(baseline time divided by candidate time; greater than one is faster):

| Case | Paired ratio | Candidate faster pairs | Separate profile, baseline → candidate |
| --- | ---: | ---: | ---: |
| M32 | 0.981728× | 2/6 | 113.761 → 113.791 µs |
| M128 | 0.967743× | 1/6 | 183.359 → 193.831 µs |
| M256 | 0.952977× | 0/6 | 224.050 → 238.184 µs |
| M16, unchanged path | 1.002119× | 4/6 | 106.615 → 105.921 µs |
| M32 after transitions | 0.991159× | 2/6 | 106.778 → 108.894 µs |

The intended all-NTT1 replay was exercised: seed 7302 had 16/16 NTT1 jobs,
39 local routes, and 156.960 → 153.632 µs (1.02166× for this single pair).
It did not approach the proposed 15% gain; both M32 groups regressed in paired
median. M256 lost every pair. Separate single profiles use another fresh input
and support rejection, but do not establish a precise cause or enter medians.

Compiler resources match: both main variants report REG=96, STACK=0, LOCAL=0;
the other variants retain 144/148/149 registers and zero stack/local bytes.
These fields do not establish absence of WGMMA serialization or explain the
regression. There is no measured spill-based explanation. Standalone routing
also remains distinct from decoder routing.

Independent critic `leaderboard_critic` rejected this variant. Preserve the
frozen incumbents and pursue the separately reviewed dense-chain selector.
Machine-readable decision: `experiments/iterations/20261104-moe-ntt1-gs4.json`.

## Falsifiable mechanism

The diagnostic's actual route histograms show NTT=1 jobs (one tile of at most
eight routed tokens) constitute 89–100% of M32 jobs, 71–81% at M128 and 45–59%
at M256. The native `wgs_seg` uses two k16 MMAs per asynchronous commit group
for every NTT. Its source records that four-MMA groups caused register-driven
serialization in the wider N16–32 variants. A single-tile job uses only four
`d` and four `d_odd` accumulator values, leaving more room than wider jobs.

The only computation change is `GS = NTT == 1 ? 4 : OKC_WGSGS`. It doubles
the operand staging array from 16 to 32 registers in that specialization and
reduces commit/wait groups per 256-K stage from eight to four. Wider jobs keep
GS=2. No weight packing, HBM request volume, expert scheduling, routing,
precision or main/tiny/MID dispatch is changed. This tests exposed issue/fence/
wait cost in the dominant small-job class; it is not a parameter sweep.

Halving synchronization groups does not imply halving kernel time. There is
no measured gain forecast. Require a material gain on the known all-NTT1
M32 input (seed 7302), preferably at least 15%, and no meaningful regression
on M128/M256 before whole-decoder testing. If allocation spills or WGMMA
serialization increase, or the paired gain is noise, reject this variant.

## Arithmetic and asynchronous lifetime proof

For GS=2 or GS=4, the nested loops execute `4 * (4/GS) * GS = 16` k16 MMAs
per 256-K stage, visiting K steps 0–15 exactly once. NTT=1 initializes `d`
at the first group's first MMA and `d_odd` at the second group's first MMA,
then alternates groups between them and combines the arrays after `wait<0>`.
GS=4 changes FP32 association but retains every term and the final BF16
boundary. Reference tolerances still require GPU validation.

The two operand buffers alternate by group parity. `wait_group<1>` after
each commit retires the earlier group before that buffer is reused. Both
eight and four groups per stage are even, so the next stage starts in buffer
0 while the previous stage's last group uses buffer 1. The first new commit
and wait retire the previous stage's last group before its activation slot
is released. The final wait retires all outstanding groups before accumulator
use or split-unit fixup.

The deferred epilogue index is rescaled from group 3 to group 1 for GS=4,
preserving the boundary after eight **issued** k16 MMAs. The retired count at
`wait_group<1>` changes from six to four; no correctness dependency on that
elapsed delay was found. A static assertion checks the issued-K boundary.
Independent review and a CPU sequence model confirmed coverage, unique
initialization, buffer lifetimes and slot retirement.

## Historical bounded validation plan

Use the standalone counterpart first. Suggested case order is M32, M128,
M256, M16, M32, using its adjacent case JSON. M16 is an unchanged-path control
and the last M32 checks scratch reuse after transitions. With the common
runner's `--seed 7202`, the first timed input is seed 7302, known to have all
NTT=1 jobs in the unchanged MoE reference. Subsequent replay seeds differ:
record their actual local counts and job-width histograms outside timing.
Standalone routes remain different from decoder routes.

```bash
.venv-modal/bin/python tools/modal_h200.py benchmark \
  --problem muse_moe_w4a16 \
  --candidate muse_moe_w4a16/candidates/native_wgs_ntt1_gs4_v7.py \
  --incumbent experiments/candidates/muse_moe_w4a16/ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a.py \
  --cases muse_moe_w4a16/candidates/native_wgs_ntt1_gs4_v7.cases.json \
  --seed 7202 --repeats 6 --resources --profile --timeout 900 --dry-run
```

The command is retained to document the completed experiment; do not rerun
this rejected variant. Root used an official-like read flush for interpretable
cache comparisons; the earlier diagnostic's `zero_` write flush did not
verify cold DRAM. Graph overhead remains in local timings.

`cuobjdump --dump-resource-usage` on directly owned extension paths reports
register counts and stack/local bytes. These are useful resource indicators,
not exact dynamic spill traffic. C7512 is a ptxas compilation diagnostic,
normally absent from the final `.so` or PTX. Successful compilation or a
missing warning in swallowed build output does not prove absence of spills
or serialized WGMMA. Save available compiler output and compare resource/
generated-code evidence for the actual main kernel.

NVIDIA documents the binary resource fields in its
[CUDA Binary Utilities guide](https://docs.nvidia.com/cuda/cuda-binary-utilities/#resource-usage-information).
No binary is modified or submitted; all GPU execution remains compiled
from the readable source in each candidate.
