# V265 measures the score implications of changing user speed and GPU throughput

On V64's four-load curve, a uniform **1.801× improvement in both aggregate user speed and GPU throughput** would reach a modeled 3× score. The 4× crossover is 2.293×. Holding user speed fixed raises those throughput requirements to 2.125× and 2.833×. Holding GPU throughput fixed puts the modeled ceiling at 2.038× even with arbitrarily fast per-user responses. The calculation transfers V64's measured curve and calibrates each stock anchor to the judge. Across all five scored receipts, the uncalibrated local interpolation reconstructs the official geometric mean to within 0.9%. The official paired per-repetition ladders are unavailable. [Reproduction and input hashes](audit.json) · [Published scoring contract](../../RULES.md) · [Interpolation implementation](../../../tools/okc_node_loadgen.py)

| V64 measured rung | Local score elasticity to GPU throughput | To aggregate user speed | To both at this rung |
| --- | ---: | ---: | ---: |
| 1 user | 0.203 | 0.198 | 0.401 |
| 4 users | 0.238 | 0.098 | 0.336 |
| 16 users | 0.126 | 0.040 | 0.166 |
| 32 users | 0.433 | 0.055 | 0.488 |

A 1% improvement on both axes at the 32-user rung gives approximately 0.49% more score near V64. The official 16-user stock anchor already lies between V64's 16- and 32-user points, which gives the latter extra influence on the interpolation. These local derivatives apply near the observed curve. They change when rungs move across an anchor. The implementation measures current tensor geometry independently of these evaluation-only comparisons. [Official anchors and measured rung rates](../../runs/abd217baf0ed49fd.json) · [Sensitivity code](audit.py)

## Hardware controls

V48b and V64 ran the scored ladder on the same physical node, image, model and workload specification. At one user, first-turn p90 fell from 8.310 to 5.419 seconds as median decode throughput fell from 236.5 to 203.2 tokens/s/user. At 32 users, first-turn p90 fell from 168.949 to 110.000 seconds and median decode throughput rose from 38.6 to 43.3. Raw GPU throughput increased 2.24% and 38.03% at the two loads respectively. The patches and flags differ, so these timings measure the complete serving configuration. [V48b run](../../runs/8ebed2bced8942ff.json) · [V64 run](../../runs/abd217baf0ed49fd.json)

At 16 users, V62 and V68 ran private checks on the same node. Normalizing both to their respective stock throughput gives a 1.224× gain. V68 includes sequence-parallel prompt processing and Humming expert changes. First-turn p90 fell from 69.865 to 50.737 seconds and median decode speed rose from 45.6 to 48.0. The closer V64/V68 private comparison uses the same flags on different nodes. V68 gained 0.94% raw throughput, 0.83% after division by paired stock throughput, and 0.85% on the published rounded single-point indicator. V64's separate three-repetition scored rung reports 1.63% spread. Those statistics cannot distinguish an individual W13, W2, gather, reduction or draft latency; the spread is descriptive. [V62](../../runs/220305c15ded4f6b.json) · [V64](../../runs/ac89fddc4ee147ab.json) · [V68](../../runs/04b01f958aa5487c.json)

The private V41 and V44 one-user client traces contain 39 matching requested and returned completion lengths. The per-request median ratio of time after the first token was 0.996. For the 21 pairs also matching prompt and cached lengths it was 0.998. Summed separately by repetition, the three streaming ratios ranged from 0.895 to 1.148. Both sessions used different machines, and equal token counts can contain different generated text. These observations explain why a small single-run delta gives weak evidence about a kernel. [Paired client timings](audit.json)

## Eight-rank component timing

The four archived full-node server logs contain eight stage reports, all from one 16,384-token V43 forced-Marlin prefill. They report 261.308–261.310 s total and 203.847–204.017 s labeled expert time across ranks. The diagnostic synchronizes CUDA on each lap and excludes stream capture. The first forward includes first-use compilation and setup. The node archive contains no Nsight Systems report or SQLite export. The available measurements leave the exposed captured Humming W13/W2, reduction and draft replay durations unresolved. [V43 instrumentation](../v43_dense_prefill_profile/patch.diff) · [All eight reports](audit.json)

Run `python3 -B audit.py` in this directory to reproduce the receipt. This investigation used the existing hardware records and added no serving patch or external run.

The follow-up eight-rank test exercises full and striped reductions at the 16,384-token native shape. A separate NCCL ring source example identifies how column packing can change the sum order even with aligned rows. [Collective comparison and mitigation](reduction_order.md)
