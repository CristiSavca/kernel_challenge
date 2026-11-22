# V195 score headroom

**The scored V64 run needs a 2.8331× improvement in geometric-mean throughput at stock interactivity to reach 4×.** At fixed interactivity and fixed work, a uniform improvement of that size removes 64.70% of elapsed time. The four loads need a product of throughput ratios of 256, so individual loads can contribute different gains. On this stock curve even perfect interactivity with V64’s raw throughput peak could score at most 2.056×. At least one new raw rung must reach 39,984 tokens/s/GPU, or 1.946× V64’s 20,547.4 peak. [Official score definition](../../RULES.md) · [V64 judge result](../../runs/abd217baf0ed49fd.json) · [Original raw-throughput bound](../v100_paged_tree_verify/frontier_budget.json)

| Users | V64 official ratio | Time saved if this anchor individually reaches 4×, fixed user speed | Whole-trace speedup with first TTFT free | Whole-trace speedup with all post-first-token time free |
|---:|---:|---:|---:|---:|
| 1 | 1.1700× | 70.75% | 1.496× | 2.041× |
| 4 | 1.3160× | 67.10% | 1.988× | 1.678× |
| 16 | 1.5919× | 60.20% | 2.827× | 1.450× |
| 32 | 1.6213× | 59.47% | 3.321× | 1.360× |

The last two columns replay 767 requests from seven repetitions of our own eight-H200 V41 run. Each user retains launch delay, interturn gaps, later-turn TTFT and all intervals outside the selected phase. Each repetition ends at the *new* last-finishing user. Throughput retains every counted prompt and output token. The script reproduces all seven measured wall times within 6 ms and all four raw rates within 0.06 tokens/s/GPU. These client intervals include queueing. Faster execution in one server phase can also shorten waits in another. The V64 judge publishes aggregated TTFT and rate summaries, which do not uniquely determine the same decomposition for V64. [Archived client trace](../../../experiments/modal_node/20261113T204343Z-bench/results__20261113T204343Z-bench__v41__load.json) · [Rerating code](score_headroom.py)

**Speeding both visible phases is one route to 4×.** With the measured V41 phase shares transferred to V64’s raw rungs, a cold-only change can mathematically cross 4× but requires **8.985×** faster first-turn TTFT. Even making every post-first-token output interval instantaneous gives a peak of **27,935 tokens/s/GPU** and a **2.795×** ceiling at any user speed. Thus an optimization confined to that interval cannot reach even 3× in this model. A change confined to later-turn TTFT gives a 1.581× modeled score. [Computed limits](score_headroom.json)

| Faster phases, holding other measured intervals fixed | Crossover for a 4× modeled score |
|---|---:|
| First-turn TTFT only | 8.985× faster, or 88.87% of that interval removed |
| First-turn TTFT and post-first-token output, same factor | 2.455× faster each |
| Post-first-token output 2× faster | First-turn TTFT 2.771× faster |
| First-turn TTFT 2× faster | Post-first-token output 4.496× faster |
| Both raw rung axes improve uniformly | 2.293× faster throughput and per-user rate |

The scenario maps the archived changes in throughput **and** aggregate per-user speed onto V64, then uses the local harness’s clamped linear-throughput interpolation in log user speed. V64’s rounded raw rungs give 1.4244× in that harness versus 1.4119× reported. Each anchor is calibrated to its official baseline before varying the phases. The judge’s unpublished paired-repetition detail and future node placement preclude an exact raw-time crossover. These conditional crossovers serve as engineering targets. [Local interpolation and metric definitions](../../../tools/okc_node_loadgen.py) · [Per-load factors, calibration and unadjusted estimates](score_headroom.json)

**V195 has no measured four-load speedup or identifiable candidate-specific hard ceiling.** Its new affine verify path retains all full-attention K/V payload, and its fused expert path retains both projections. A separate native H200 attention implementation saved 103.8 µs per full layer at 16 users, equivalent to 1.66 ms across 16 layers if transferable. V68’s separate 16-user serving result implies an illustrative 83.3 ms round from median user decode and mean advancement. V191 removes one expert kernel launch per eligible layer and 496 MiB of logical reads/writes across 62 layers at 32 users under uniform routing. Cache traffic, launch cost and overlap on V195 remain unmeasured. Neither figure supplies the cold-time reduction above. A matched hardware check should report raw throughput, aggregate per-user interactivity and first-turn TTFT together. [Attention hardware context](../v192_affine_verify_tiles/hardware_context.json) · [Expert geometry and measured coverage](../v191_decode_gate_up_epilogue/README.md) · [V68 hardware result](../../runs/04b01f958aa5487c.json)
