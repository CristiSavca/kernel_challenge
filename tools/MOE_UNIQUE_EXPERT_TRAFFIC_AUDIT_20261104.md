# Unique-expert versus FT64-job weight traffic

**Do not implement another wider-tile/reuse rewrite on the retained evidence.** The bandwidth denominator did count jobs, so near-nominal traffic alone was insufficient to dismiss reuse. Recounting the actual histograms still finds a small opportunity, with significant wider-tile costs. This is not a claim that FT128 was experimentally disproven.

Pinned source is public MoE `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`. The adjacent evidence is `experiments/iterations/20261104-moe-unique-expert-traffic-audit.json`; it binds the original histograms, current hardware run, exact decoder profile and prior experiments. No generated routing samples, GPU calls or candidate changes were used.

Each full expert pass requests 27 MiB: 18 MiB W13 and 9 MiB W2. Define `U=count(c>0)`, `J64=sum(ceil(c/64))`. Complete reuse removes at most the logical requests `27 MiB*(J64-U)`; FT128 removes only `27 MiB*(J64-J128)`. Requests are not measured removable HBM bytes.

| Actual retained samples | All-reuse request saving | Mean |
|---|---:|---:|
| M32, three earlier + six current replays | 0% | 0% |
| M128, three earlier routes | 0–4.17% | 2.67% |
| M256, three earlier routes | 0–11.54% | 7.29% |
| M256, six current 247888 replays | 0–10.71% | 6.00% |

Two M32 histograms occur in both artifacts; there are 19 distinct histograms, not 21 independent samples. The current profiled M256 route is U26/J27. Its 770,291,712 read bytes are **1.007692× job bytes but 1.046449× unique-expert bytes**. Removing the single repeated 27 MiB pass equals 3.675% of measured reads, without proving those particular bytes reached DRAM. All FT64 reuse is zero for M≤64. No actual decoder histogram or M384 histogram was found in this bounded evidence; standalone routing cannot substitute for decoder routing.

Using the exact 97af public profile/benchmark, its 46:16 layer weighting and six-load GM, transfer the largest observed M128/M256 fractions and optimistically reduce the **entire MoE time** by those fractions with zero overhead. The modeled whole-score gain is **1.465%**; FT128 alone on those same routes gives **1.093%**. The current NCU M256-only scenario gives 0.340%, or 0.555% using all six current replay fractions' mean. These are conditional proportional-work scenarios, not predictions or hard latency ceilings. Wider MMA issue/decode changes might matter separately, but no evidence attributes their cost. They do not support closing the saved 18.1% score gap, much less doubling score.

Current R256 geometry differs from the older compact-layout FT128 audit. Native extraction of the actual metadata struct gives:

| FT/K/slots | Dynamic shared | Accumulator words/thread |
|---|---:|---:|
| 64/256/3, parent | 231,168 B | 32 |
| 128/256/3 | 328,704 B, exceeds 232,448 B opt-in limit | 64 |
| 128/256/2 | 226,304 B, shared-only fit | 64 |
| 128/128/3 | 175,104 B, shared-only fit | 64 |
| 256/256/2 | 356,992 B, exceeds limit | 128 |

FT128 also retains 16 A operand words before other live state under the unchanged 576-thread allocation ceiling of 96 registers. Two slots sacrifice lookahead; K128 doubles stage events to `2*J128/J64` while padded activation work remains unchanged. NTT9–16, cost tables, readiness and doubled fixup payload require real changes. Source word inventories are not compiler allocation measurements.

Keeping one stage while processing multiple jobs requires all their accumulators or extra partial storage. Keeping a whole output unit's weights for sequential complete jobs takes 576 KiB, exceeding shared capacity. A naive K-outer/global-partial implementation for 128 tokens adds 180 MiB of partial writes/reads across both phases versus 27 MiB of saved weight requests. This illustrative accounting is not a proposed design.

Prior SS-A, compact/NW4 residency and V30 whole-K experiments did not implement cross-job reuse; their failures cannot disprove it. Prior FT128 was an analytical rejection against a different geometry. Arbitrarily hot valid routes can have much larger duplicate fractions, so the missing decoder J/U distribution is the specific evidence limitation. No new run or implementation is requested by this audit.
