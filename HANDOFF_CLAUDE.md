# Open Kernel Challenge — Claude handoff

**November 19, 8:11 p.m. Eastern.** The final integrated V312 is available for a private hardware check if one is refunded. V204 holds the measured 1.5487× score. V312’s long-prompt workspace and decode blocks are smaller in local replays. Its serving speed remains open. The one remaining scored slot is reserved for a measured winner. [Candidate and evidence](serving_v0523/candidates/v312_bounded_prefill_and_decode/README.md) · [Final-slot comparison](experiments/iterations/20261120-serving-final-slot.json)

**November 19, 7:50 p.m. Eastern. V204 scored 1.5487× across 1, 4, 16 and 32 users.** It passed correctness, all four latency points and the memory check at +1.9 GB/GPU. The previous four-load best was 1.4119×. V139 passed the four-user check at 10,150.6 tokens/s/GPU and 56.7 end-to-end tokens/s/user, below V204’s 12,973.1 and 75.3. [Official V204 result](serving_v0523/runs/ad275bf67d9a4a0a.json) · [V139 check](serving_v0523/runs/b013f83007d544d3.json)

The user reserved the final scored slot for a measured winner. V129/1, V141/32 and V154c/16 used all three final private slots and were queued. The existing collector saves completed receipts. The read-only comparison script verifies each candidate's exact patch, flags and job, then reports raw throughput, user speed and first/later TTFT against V204. It also gives the separate V204 private comparator for the 16-user check. The cutoff is 3:00 a.m. Eastern on November 20. [Final-slot comparison](experiments/iterations/20261120-serving-final-slot.json) · [Refresh script](experiments/iterations/20261120-serving-final-slot.py)

**November 19, 18:43 UTC. V204 is running its four-load scored job. V139’s four-user check is queued.** V64 remains the best scored result at 1.4119×. Replaying 767 historical request traces on V64's calibrated curve puts a 20% cut in cold first-token time and a one-third cut in streaming time at 2.0088×. Halving both gives 3.1870×. These scenarios hold later-turn first-token times and client gaps fixed. V204’s single-load advantage transferred uniformly to all loads would give 1.5207× in a separate scenario. [Targets and assumptions](experiments/iterations/20261119-serving-score-targets.json) · [Live score receipt](experiments/iterations/20261119-serving-score-state.json)

V308 ports V306’s GPU-controlled long-prompt expert waves to the passing V204 patch. A shared dispatcher preserves V204’s optimized W2 and its original fallback on both ordinary and wave paths. The CPU replay covered 180 regimes, 53,280 BF16 output words and 1,539 W2 calls. The separate mixed scheduler fixture passed 96 rounds. V307 independently repairs the JIT exception that prevented V120b’s 32-user test from producing any timings. Both candidates still need physical serving checks. No new submission or change to the November 20 workers was made in this iteration. [V308 patch](serving_v0523/candidates/v308_mixed_compact_prompt_waves/patch.diff) · [V308 replay](serving_v0523/candidates/v308_mixed_compact_prompt_waves/joined_replay.json) · [V307 repair](serving_v0523/candidates/v307_paged_allocator_compatibility/README.md)

**November 19, 18:29 UTC. V204 passed its 16-user check and the judge accepted the identical patch and flags for a four-load scored run.** V204 delivered 18,338.6 tokens/s/GPU and 1.489× at the stock point. V64's comparable check delivered 17,349.9 and 1.412×. V64's official four-load score of 1.4119× remains the team best. V93 failed correctness and V120b crashed. V139 remains queued. The serving cutoff is 08:00 UTC November 20. [V204 private result](serving_v0523/runs/7fcdf9abf39b486a.json) · [Scored acceptance](serving_v0523/runs/v204_nov19_scored.submit.txt)

V306 is a separate long-prompt candidate. It removes the host route-count read from eligible fused expert layers. At 16K tokens the same 384 MiB BF16 arena fits 24,576 routed rows, up from 16,384. The next wave stayed idle in all 512 synthetic rank draws. The C++ launcher and 18 Hopper kernel variants compiled, and CPU tests checked 21.7 million packed words. V306 has no physical serving result. V303 evaluates bulk cache transfers, V304 evaluates decode operand conversion, and V305 rules out early commitment from a partially reduced target head. [V306 and evidence](serving_v0523/candidates/v306_compact_prompt_waves/README.md) · [Iteration](experiments/iterations/20261119-serving-v303-v306.json)

**November 19, 08:15 UTC. V302 combines small-row expert verification with bounded paged attention. V64 holds the best official four-load score at 1.4119×.** On both small expert projections, the Hopper warpgroup issues four ordered K16 slices before waiting, halving the full waits while preserving the original FP32 accumulator chains. The archived one- and four-user runs were in this expert size bin in all 102 scheduler reports. At one request, sliding attention reduces its grid from 132 to 68 thread blocks per eligible layer. The combined export uses 409,489 of 409,600 bytes. Local validation covered 56 expert dispatch cases, 68,955 unchanged attention intersections and a dense FP64 comparison at 184,320 output components. The integrated tree has no eight-H200 timing or generated-token result. [V302 patch, flags and evidence](serving_v0523/candidates/v302_decode_verify_partitions/README.md) · [Independent expert instruction audit](serving_v0523/candidates/v299_four_slice_decode/independent_chain.md) · [Attention geometry and numerical tests](serving_v0523/candidates/v301_bounded_verify_partitions/README.md)

V298 is a separate cold-prompt candidate on V68. For an eligible 16K checkpoint covering all three KV pools, the original path can submit 98,304 small CUDA copies for the same 288 MiB that V298 transfers with three mapped GPU kernels. The traces do not identify the exposed backup time. The latest judge read showed V93/16, V204/16, V120b/32 and V139/4 queued, with today's three private checks used and one scored slot available. The existing November 20 private selection retains its reservation. [Cold-prompt candidate](serving_v0523/candidates/v298_mapped_page_first_backup/README.md) · [One-user component calibration](serving_v0523/candidates/v300_verify_component_calibration/README.md) · [Official judge](serving_v0523/candidates/v302_decode_verify_partitions/judge_observation.json) · [Iteration](experiments/iterations/20261119-serving-v298-v302.json)

**November 19, 07:48 UTC. V297 widens both fused expert projections during target verification. V64 remains the team's best measured score at 1.4119×.** Each eligible kernel processes twice as many output channels per tile, while retaining the compact scaled gate/SiLU output and the original FP32 reduction order. On the archived 256-row routing case, the two projections together go from 2,112 tile jobs to 1,056 and request 198 MiB less activation data per layer and rank before cache effects. All four wide Hopper kernels compiled with zero register spills. Three earlier images with the expected matrix instructions still spilled and now fail the startup check. The 405,695-byte export builds on V295 and preserves mixed attention, rank-major decode and prompt route publication. V296 remains the separate direct-normalization candidate. [V297 patch and results](serving_v0523/candidates/v297_wide_fused_decode/README.md) · [Machine-code admission](serving_v0523/candidates/v297_wide_fused_decode/independent_spill_guard.json)

At 07:48 UTC V93/16, V204/16, V120b/32 and V139/4 were queued. No new submission was made. Token agreement, full-graph H200 memory, time to first token and throughput remain unmeasured for V297. The archived scheduler reached an eligible size in 11 of 21 reports at 16 users and 18 of 24 at 32 users. Under a V64 curve scenario holding the lower two rungs fixed, doubling both high-rung throughput and user speed gives about 2.37× overall. [Official judge](serving_v0523/candidates/v297_wide_fused_decode/judge_observation.json) · [Four-load score sensitivity](serving_v0523/candidates/v297_wide_fused_decode/high_load_score.json) · [Iteration](experiments/iterations/20261119-serving-v297.json)

**November 19, 07:23 UTC. V296 publishes exact route metadata and normalized expert inputs at one peer barrier. V64 remains the team's best official score at 1.4119×.** Over a fully eligible 16K prompt on eight GPUs, V296 removes 186 separate route collectives and 62 copy launches. The copies represent 1.94 GiB of logical traffic per rank. Two independent multiprocess replays each checked 7,813,888 BF16 activation words and 861,584 score/ID pairs. Replacing a required fallback copy with an alias produced 17,088 errors. The final export uses 407,046 of 409,600 bytes. Eight-H200 token agreement, time to first token, peak memory and throughput remain unmeasured. [V296 patch and evidence](serving_v0523/candidates/v296_normalized_peer_routes/README.md) · [Per-load work](serving_v0523/candidates/v295_published_prompt_routes/work.json)

V93/16, V204/16, V120b/32 and V139/4 remained queued at 07:23 UTC. Today's one scored slot was available and all three private slots were used. V68's comparable 16-user gain over V64 is 0.85% after normalizing to each node's stock point, below V64's 1.63% repetition spread. None of the 18 finished private jobs returned within the eight hours between the final reset and closing. The existing V229/V198 conditional private selection remains in place. [Official queue](serving_v0523/candidates/v296_normalized_peer_routes/judge_observation.json) · [Submission calibration](serving_v0523/candidates/v292_calibrated_selection/README.md) · [Iteration](experiments/iterations/20261119-serving-v292-v296.json)

**November 19, 07:01 UTC. V291 combines ranked decode and mixed prompt attention. V64 holds the official four-load score at 1.4119×.** Each GPU plans the current expert rows while it gathers pooled columns. Long prompts use alternating peer input buffers. When a full prompt shares a forward with running decoders, the certified path preserves direct sliding-window attention and can split dense prompt attention from short paged verification. On the archived eight-H200 V41 schedule, 58 of 72 full-prompt joins at 16 users and 115 of 145 at 32 users fit the full-attention prompt pack. Across the eight ranks of a reconstructed 16K routing fixture, the largest logical incoming peer payload was 74.70 MiB versus 112 MiB for the gathered input. H200 token agreement, time to first token, memory and throughput remain unverified for V291. [V291 and final patch](serving_v0523/candidates/v291_mixed_ranked_attention/README.md) · [Phase assumptions and timing demand](serving_v0523/candidates/v291_mixed_ranked_attention/work.json) · [Route counts by sender and receiver](serving_v0523/candidates/v290_mixed_ranked_peer_serving/routed_payload.json)

The final patch is 396,795 bytes. All 40,320 flag orders and 17 server configurations passed. The host mixed-attention comparison covered 133,264,128 visible FP8 K/V bytes, and the paged verifier checked 409,600 BF16 scalars against FP64 causal attention. The 07:01 UTC judge query still showed V93/16, V204/16, V120b/32 and V139/4 queued, with no November 19 private slots left and one scored slot available. The existing November 20 conditional V229/V198 private selection retains its reservation. [Judge receipt](serving_v0523/candidates/v291_mixed_ranked_attention/judge_observation.json) · [Combined iteration](experiments/iterations/20261119-serving-v286-v291.json) · [V288 paired historical H200 timings](serving_v0523/candidates/v288_paired_cold_and_verify_intervals/README.md) · [V287 scheduler ablation](serving_v0523/candidates/v287_prompt_completion_decode/README.md)

**November 19, 06:34 UTC. V284 is the new decode candidate. V64 still holds the official four-load score at 1.4119×.** V283 makes the FFN boundary consume the gathered unpooler columns directly. V284 also applies the full pre-expert RMSNorm to gathered pooler columns and writes each normalized result directly to its routed expert rows. Against V229, an eligible TP8 pass with uniform routing removes 124 GPU graph nodes at eight target rows and 185 at 32, 128 or 256 rows. At the largest modeled load, it removes 984 MiB of logical payload reads and writes per rank. Real H200 throughput, output agreement, time to first token and peak allocation remain open. The compiled Hopper input kernel uses 32 registers with zero spills. The exact source passes the full export rules and all 720 flag orders. [V284 implementation](serving_v0523/candidates/v284_rank_major_routed_pool/README.md) · [V283 unpooler](serving_v0523/candidates/v283_rank_major_unpool_boundary/README.md) · [Combined work and comparison](experiments/iterations/20261119-serving-v281-v285.json)

V93/16, V204/16, V120b/32 and V139/4 were queued at 06:34 UTC. All three November 19 private slots were used, with one scored slot available. The November 20 conditional private standby retains its V229/V198 plan and now recognizes a separately accepted scored run. The replacement passed 151 offline tests, acquired the sole process lock and preserved all 31 submission receipts. V284 has no reservation. [Judge observation](serving_v0523/candidates/v284_rank_major_routed_pool/judge_observation.json) · [Running private standby](serving_v0523/runs/nov20_v229_quota_split_runner.launch.json) · [4× frontier calculation](serving_v0523/candidates/v283_rank_major_unpool_boundary/score_target.json)

**November 19, 05:50 UTC. V278 is the new decode candidate. V279 combines the corrected prompt expert kernels. V64 holds the measured four-load score of 1.4119×.** V278 widens both decode expert projections and overlaps route planning with the pooler and input normalization. On the archived 256-row routing case, jobs fall from 1,056 to 528 per projection and the two kernels request 198 MiB less activation payload per layer and rank. Four high-load kernels compiled without spills. The integrated callback and fallback passed 24 host model comparisons. V68’s matched private control is 17,512.9 tokens/s/GPU at 48.0 tokens/s/user. No H200 throughput, agreement, memory or TTFT is available for V278. [V278 documentation](serving_v0523/candidates/v278_overlapped_wide_decode/README.md) · [Official decode patch](serving_v0523/candidates/v278_overlapped_wide_decode/patch.diff)

**V270 and V273 are withdrawn.** A cluster could overwrite a slower neighbor’s previous epilogue. Both corrected projections in V279 acknowledge the peer’s post-epilogue fence before the next multicast. On the reconstructed rank-3 16K prompt, gate/up source boxes fall from 4,356 to 1,089 MiB per layer. The six prompt source images were checked and 11 host model-entry cases passed. The patch is 407,230 of 409,600 bytes. Four older private runs were queued and all three November 19 checks were used at 05:50 UTC. The existing November 20 conditional V229/V198 worker retains its prior plan. [V279 evidence and limits](serving_v0523/candidates/v279_multicast_expert_serving/README.md) · [Fence audit](serving_v0523/candidates/v277_multicast_prompt_gate_up/independent_fence.md) · [Official queue](serving_v0523/candidates/v279_multicast_expert_serving/judge_observation.json)

Cache replay changes how to interpret these candidates. Under a 50 MiB L2 model most avoided activation transfers already hit cache. On the V278 decode fixture, the change in activation misses ranges from 0.47 MiB more to 0.07 MiB fewer per layer and rank. The W13 V279 fixture has identical modeled HBM reads across the tested issue orders. The compiled decode kernels still issue 5.8–6.7% fewer inner-loop instruction sites per equal unit of output. V280’s four-expert scheduling experiment is held after it increased modeled total cache misses by 18–122 MiB at 16K. Its shared-memory alignment was corrected and checked in the actual SM90a binary. [V278 cache study](serving_v0523/candidates/v278_overlapped_wide_decode/cache_sensitivity.json) · [V279 cache study](serving_v0523/candidates/v279_multicast_expert_serving/gate_cache.json) · [V280 mixed-cache replay](serving_v0523/candidates/v280_expert_width_order/shared_cache.json) · [Iteration](experiments/iterations/20261119-serving-v274-v280.json)

**November 19, 05:26 UTC. V273 is ready for a future private comparison. V64 still holds the measured four-load score of 1.4119×.** The new expert scheduler removes two block-wide setup barriers in captured W13 and W2. The new long-prompt W2 uses a 256-row tile, shared activation transfers and occupancy-sized clusters, with the existing 176-row tile for unsuitable local row counts. At 16K uniform routing the modeled W2 weight requests drop from 929.8 to 716.6 MiB per GPU per layer. The exact combined patch regenerates all nine offline Hopper sources and passes the native model and distributed host replays. It keeps full-width prompt QKV and the native attention-output sum. The official export uses 409,051 of 409,600 bytes. [V273 package and evidence](serving_v0523/candidates/v273_prompt_decode_expert_serving/README.md)

V272 separately overlaps routing-plan construction with pooled input normalization on the private-tested V68 base. V271 found that even costless rebalancing under perpetual forced rank skew, together with twice-as-fast later phases, yields only a modeled 2.556×. No new H200 score, allocation or submission. V93/16, V204/16, V120b/32 and V139/4 remain queued, with the next private quota on November 20 at 00:00 UTC. The previously armed V229/V198 conditional plan stays in effect. [Iteration receipt](experiments/iterations/20261119-serving-v269-v273.json)

**November 19, 05:02 UTC. V268 is the new candidate for a private hardware comparison. The measured four-load score is still V64’s 1.4119×.** V268 reads the original token rows directly in the eight-row W13 expert kernel. On an archived one-request route distribution this removes 160 KiB of staging traffic per GPU per layer, or 9.69 MiB in a 62-layer verification pass. It also combines prompt route compaction, larger-batch decode staging, prompt QKV overlap and terminal residual fusion. It uses V248’s full-address draft KV storage. The official 407,883-byte export passes public source and flag checks. H200 output agreement, TTFT, memory and speed are open. [V268](serving_v0523/candidates/v268_native_sum_expert_serving/README.md) · [W13 calculation](serving_v0523/candidates/v263_token_major_expert_decode/work.json) · [Iteration](experiments/iterations/20261119-serving-v263-v268.json)

V264 and V267 split the attention output reduction into two messages. On Gloo at native width, 25,679 BF16 sums differed, including 159 at TP8 and 8,200 prompt rows. A constructed near-tie changed the greedy choice. V268 restores the original full-width output projection and reduction. On 48 adverse cases it reproduced the 1,556 discrepancies in the earlier kernel and matched all 6,291,456 output words and all 40 near-tie decisions with the correction. At 05:01 UTC, four earlier private jobs were queued and all three November 19 checks were used. No jobs or external GPU allocations were added. V265’s model estimates that increasing both throughput and per-user speed uniformly by 1.801× from V64 would reach a 3× score, and 2.293× would reach 4×. [Numerical investigation](serving_v0523/candidates/v267_pipelined_expert_serving/attention_precision.md) · [Adverse replay](serving_v0523/candidates/v268_native_sum_expert_serving/adverse_attention.json) · [Score model](serving_v0523/candidates/v265_native_serving_profile/README.md) · [Judge state](serving_v0523/candidates/v268_native_sum_expert_serving/judge_observation.json)

**November 19, 04:40 UTC. V261 fuses small-batch expert-route planning and W13 input staging during ordinary target verification. V64 holds the measured four-load score of 1.4119×.** On an eligible 62-layer pass, V261 issues 62 staging launches in place of 124. It preserves each expert, its route weight and the order of the staged BF16 rows. At the four-user verification geometry the eight-warp kernel compiled with zero register spills. The initial four-warp build spilled 28 bytes each on loads and stores. The actual kernel bodies matched over 10.26 million BF16 output words in the interpreter. An independent CPU replay of the normal grouped entry passed 180 paired cases. The official export is 404,326 bytes. H200 token agreement, peak memory, TTFT and throughput remain unmeasured. [V261 implementation and checks](serving_v0523/candidates/v261_fused_decode_route_staging/README.md) · [Source and rules](serving_v0523/candidates/v261_fused_decode_route_staging/validation.json) · [Iteration](experiments/iterations/20261119-serving-v257-v262.json)

V258 replaces a global prompt route sort with stable tile counts and compact placement. V260 overlaps QKV projection with a 128 MiB peer exchange on a full prompt. On the archived 32-user schedule its assumed peak-rate overlap amounts to 2.78 of 304.43 seconds. At the same load a separate 4× budget study estimates a further 62.76 ms would have to come out of the V64 decode pass if all four loads improved equally. The five dense projections and routed-expert control together account for 56.59 ms across separate single-H200 microbenchmarks. The most informative missing measurement is the captured eight-rank target expert and reduction time alongside the draft pass. At 04:35 UTC four earlier private jobs were queued and all three November 19 private slots were used. No new submission or external GPU allocation was made. The V229/V198 conditional standby remains active. [V258 route packing](serving_v0523/candidates/v258_compact_prompt_route_plan/README.md) · [V260 QKV](serving_v0523/candidates/v260_pipelined_prompt_qkv/README.md) · [4× timing budget](serving_v0523/candidates/v262_decode_phase_budget/README.md) · [Queue](serving_v0523/candidates/v260_pipelined_prompt_qkv/judge_observation.json)

**November 19, 04:17 UTC. V256 integrates sliding-window draft storage and bounded FA4 address lookup with V248’s mixed serving pipeline. V64 remains the official four-load incumbent at 1.4119×.** The five draft layers would use 6.116 GiB less device KV payload per GPU at V68’s measured slot counts. V256 also fixes a partial-write recovery path. The unadapted integration reproduced 12 failures. The repaired writer passed all 16 cases across changing physical slots. A separate 54-call scheduler replay checked 1,473,120 BF16 words through the following tick. For 64 consecutive draft pages, the host-copy fixture issued two descriptors in place of 128. H200 token agreement, peak memory, TTFT and speed remain open. [V256 patch and evidence](serving_v0523/candidates/v256_window_backed_mixed_serving/README.md) · [Recovery regression](serving_v0523/candidates/v256_window_backed_mixed_serving/paired_store.json) · [Iteration](experiments/iterations/20261119-serving-v249-v256.json)

The focused control is V255 on V68. V254 preserves both native router GEMMs and folds their FP32 addition into the routing kernel. V252's alternative summation order can affect near-tied experts on a CPU fixture and remains experimental. At 04:05 UTC V93/16, V204/16, V120b/32 and V139/4 were queued, with all three November 19 private checks used. The V229/V198 conditional standby retains its existing gates. This iteration used zero external GPU allocations and zero submissions. On V41 at 32 users the first-token wave occupied 212.78 of 304.43 seconds, which limits the benefit available entirely after that wave to 1.431×. [Queue](serving_v0523/candidates/v255_private_draft_window_metadata/judge_observation.json) · [4× time budget](serving_v0523/candidates/v251_hicache_residency/README.md)

**November 19, 03:33 UTC. V248 reuses the GPU tile schedule across ordinary decode verification layers. V64 remains the official four-load incumbent at 1.4119×.** At 32 live requests, V247 ran 73,784 local batch scans across 62 layers. V248 issues two planner launches covering 14 thread blocks and refills the tables at each verification. All 72 Hopper builds assembled without spills. In 36 numerical comparisons every inspected intermediate and all 1,589,760 live BF16 outputs matched V247. Hardware output agreement, throughput, peak memory and TTFT are outstanding. The patch is 408,604 bytes. The existing private queue and conditional November 20 selection are recorded with V247. [V248 implementation and reach](serving_v0523/candidates/v248_shared_verify_raster/README.md) · [Numerical results](serving_v0523/candidates/v248_shared_verify_raster/kernel_replay.json) · [Capture and failed-refresh tests](serving_v0523/candidates/v248_shared_verify_raster/independent_graph.json) · [Iteration](experiments/iterations/20261119-serving-v248.json)

**November 19, 03:21 UTC. V247 combines a single expert-routing collective with direct writes into mixed prompt attention storage. V64 holds the measured four-load score of 1.4119×.** A full eligible 62-layer prompt pass uses 62 routing collectives in place of 186. At 16 and 32 users, 58 and 115 single-prompt joins on the archived V41 cold wave could eliminate 7.25 and 14.375 GiB of logical FP8 copying per GPU. These figures do not predict H200 speed. The 401,616-byte export passed both component orders, all 40,320 flag orders and 17 runtime configurations. Four prior checks were queued, with all three private checks used for November 19. The existing November 20 conditional selection remains active. [Implementation and tradeoffs](serving_v0523/candidates/v247_coalesced_serving_decode/README.md) · [Numerical checks](serving_v0523/candidates/v247_coalesced_serving_decode/independent_host.md) · [Iteration](experiments/iterations/20261119-serving-v242-v247.json) · [Judge state](serving_v0523/candidates/v247_coalesced_serving_decode/judge_observation.json)

**November 19, 02:57 UTC. V241 joins long-prompt dense FA3 with paged verification for running requests. The best official four-load score is still V64's 1.4119×.** The combined candidate also shares partition planning across adjacent query heads and fuses FP32 vocabulary scaling into the greedy winner reduction. At 32 users the ordinary full-model merge grid falls from 253,952 to 63,488 thread blocks. In the eight-H200 V41 cold wave, 58 of 72 joined arrivals at 16 users and 115 of 145 at 32 could fit the prompt pack within its 32 MiB bound. The pinned FA3 host selector enables bulk KV loads for the long prompt in the split. Device throughput remains unmeasured. [V241 implementation and limits](serving_v0523/candidates/v241_split_serving_decode/README.md) · [FA3 dispatch](serving_v0523/candidates/v241_split_serving_decode/fa3_dispatch.json) · [Iteration](experiments/iterations/20261119-serving-v236-v241.json)

Seventeen mixed model and graph comparisons exercised both numerical kernel bodies. They checked 409,600 tail output scalars against FP64 causal attention, with maximum relative L2 of 0.002244. All 12 component orders and 40,320 flag orders passed. The two-context official patch is 397,454 bytes. Four checks were queued and three of three November 19 private checks were used. The existing November 20 conditional V229/V198 standby remains active. V241 consumed zero checks or scored runs. [Integrated mixed numerical check](serving_v0523/candidates/v241_split_serving_decode/mixed_merge_numerical.json) · [Source and flags](serving_v0523/candidates/v241_split_serving_decode/validation.json) · [Judge observation](serving_v0523/candidates/v241_split_serving_decode/judge_observation.json)

**November 19, 02:30 UTC. V235 now shares full prompt chunks with running decoders and retains direct attention for certified sliding-window layers. V64 still holds the official four-load score of 1.4119×.** On the historical eight-H200 schedule, six extra graph sizes could fit the entire running cohort alongside 72 full chunks at 16 users and 145 at 32. V235 also sends prompt rows directly to the GPUs that own their routed experts and writes exact FP8 query values during normalization. The combined public patch is 407,103 bytes. All 12 valid component orders and 40,320 flag orders passed. V235 has no H200 result. [V235 and physical limitations](serving_v0523/candidates/v235_peer_mixed_sliding_prefill/README.md) · [Cold-wave replay](experiments/iterations/20261119-serving-cold-wave.json) · [Iteration](experiments/iterations/20261119-serving-v230-v235.json)

The candidate keeps the ordinary full-attention path while packing every visible sliding-window key and current row for a linear, rank-agreed decode tail. A separate V235/V233 replay checked that rejected slots are reusable on the next ordinary call. The newest four private jobs were queued, and all three November 19 private checks were used. The November 20 closing standby still selects V229 on a qualifying V204/16 result with V198 as fallback. The new iteration issued zero jobs. [Mixed attention and reuse](serving_v0523/candidates/v235_peer_mixed_sliding_prefill/independent_causal_attention.md) · [Private jobs](serving_v0523/candidates/v235_peer_mixed_sliding_prefill/judge_observation.json) · [Armed standby](serving_v0523/runs/nov20_v229_runner.launch.json)

**November 19, 02:00 UTC. V229 is implemented and armed for a conditional 16-user H200 check. V64 holds the measured four-load score of 1.4119×.** V229 shards two replicated target projections across eight GPUs and combines the draft candidate exchanges into a single gather. At 128 verification rows this removes 6.78 GiB of nominal projection weight reads per rank across 62 layers. Each rank contributes 7,168 bytes to the draft gather at 16 users, down from 21,504. Archived timings put the idealized verification-period gain from the two target projections alone at about 1.043× at that load. V229 has no measured GPU speed, agreement, peak memory or TTFT result. [V229 patch and validation](serving_v0523/candidates/v229_sharded_decode_serving/README.md) · [Timing assumptions](serving_v0523/candidates/v228_sharded_verify_projections/work.json) · [Iteration](experiments/iterations/20261119-serving-v225-v229.json)

Independent four- and eight-rank replay matched 218,726,400 inspected words against V224. The integrated selector reproduced the sampled tokens and proposal probabilities at all four batch sizes, and CPU compilation preserved all 65,536 BF16 encodings. The November 20 04:00–08:00 UTC standby now selects V229 on a passing V204/16 above both existing V68 thresholds. V198 remains the fallback and V120b retains priority. The last judge observation had V93/16, V204/16, V120b/32 and V139/4 queued with 3/3 private checks used. The handover kept all 31 submission receipts intact and used no quota. [Worker receipt](serving_v0523/runs/nov20_v229_runner.launch.json) · [Private check conditions](serving_v0523/runs/history/nov20_v229_staging/nov20_v229.plan.json) · [Queue observation](serving_v0523/candidates/v229_sharded_decode_serving/judge_observation.json)

**November 19, 01:36 UTC. V224 is implemented and locally verified. V64 holds the official four-load score of 1.4119×.** V224 sizes the target-to-draft prefill outputs to the rows each TP rank owns. This reduces six named buffers from 1,536 MiB to 192 MiB per GPU at the largest captured shape and avoids 640 MiB of logical copy traffic per rank per full chunk. It also batches several ragged prompts into one native FA3 call and reduces prompt expert input packing from five launches to one per eligible layer. V219’s decode paths are included. [V224](serving_v0523/candidates/v224_packed_prefill_serving/README.md) · [Official export](serving_v0523/candidates/v224_packed_prefill_serving/patch.diff) · [Iteration](experiments/iterations/20261119-serving-v221-v224.json)

The combined patch is 362,196 bytes. All 720 flag orders and all six component application orders passed. Source-based replays on the exact tree matched 81,512,960 physical KV bytes and 25,170,432 BF16 expert outputs. Four- and eight-rank checks covered 144 graph/rank cases. No H200 speed, generated-token agreement, TTFT or peak-memory result is available for V224. Four prior private checks remain queued and the November 19 private quota is exhausted. The November 20 V215/V198 standby is active under its existing gates. [Judge observation](serving_v0523/candidates/v224_packed_prefill_serving/judge_observation.json) · [Standby](serving_v0523/runs/nov20_v215_runner.launch.json)

**November 19, 01:18 UTC. V219 is frozen for H200 testing. V64 still holds our measured four-load score of 1.4119×.** The ordinary target verification graph now splits the 40,960-input draft projection across output channels. Each GPU needs 80 MiB of FC weights per pass, down from 640 MiB, and the worker issues one gather and zero per-step CPU votes. The same target pass can use Hopper bulk loads on consecutive 64-key tiles in the 46 sliding-window layers. All 22 offline Hopper builds passed. Four- and eight-rank Gloo tests compared 215,040 BF16 values, and a full-width CPU test matched all 65,536 FC outputs and all 65,536 normalized values. [V219](serving_v0523/candidates/v219_verify_graph_sliding_tiles/README.md) · [Full-width test](serving_v0523/candidates/v219_verify_graph_sliding_tiles/full_width.json) · [Iteration](experiments/iterations/20261119-serving-v216-v220.json)

All four official checks were still queued when V219 was validated. The November 19 private quota was 3/3 with 0/1 scored used. V219 has no reservation. Keep the existing V215/V198 private standby. Whole-model memory, accuracy, TTFT and throughput remain open. V216 found zero repeated expert weights at one user, and V220 found zero extra joins from already earned per-prompt credit. [Judge state](serving_v0523/candidates/v219_verify_graph_sliding_tiles/judge_observation.json) · [Standby](serving_v0523/runs/nov20_v215_runner.launch.json) · [Weight analysis](serving_v0523/candidates/v216_decode_weight_multicast_feasibility/README.md) · [Scheduler replay](serving_v0523/candidates/v220_causal_prompt_bucket_credit/README.md)

**November 19, 00:50 UTC. V215 is ready for an H200 private check. V64 holds the measured four-load score of 1.4119×.** The new decode kernels skip padded expert rows and retain the original FP32 reduction order. On a recorded rank-local routing histogram with 256 verification tokens, issued matrix rows fall from 792 to 456 and intermediate shared-memory traffic from 123.75 to 57.19 MiB per layer and rank. Packed expert weight traffic stays at 891 MiB. A 20 MiB peer draft weight bank per GPU makes 58 of the 99 possible mixed batches eligible in the archived 16-user replay. All six kernels compiled for SM90a. The integrated four- and eight-rank Gloo replay passed 88 paired comparisons. H200 speed, accuracy, first-token latency and peak memory remain unmeasured. [V215](serving_v0523/candidates/v215_decode_reduction_paired_context/README.md) · [Arithmetic on the measured routing histogram](serving_v0523/candidates/v213_live_decode_expert_reduction/physical.json) · [Combined regression](serving_v0523/candidates/v215_decode_reduction_paired_context/joined.json) · [Iteration](experiments/iterations/20261119-serving-v212-v215.json)

The existing November 20 final 16-user private standby can choose V215 if V204/16 passes and beats V68 by more than 1.63% in both raw and stock-normalized throughput. V198 remains its fallback under the existing gates. A completed V120b/32 gives priority back to its original branch. Activation held all 31 submission records intact and consumed zero check or score quota. V93/16, V204/16, V120b/32 and V139/4 were queued at the latest observation. [Standby and quota receipt](serving_v0523/runs/nov20_v215_runner.launch.json) · [Judge status](serving_v0523/candidates/v215_decode_reduction_paired_context/judge_observation.json)

**November 19, 00:21 UTC. V211 is frozen for H200 testing. V64 holds the measured four-load score of 1.4119×.** V211 selects narrower Hopper operations for the final tile of each prompt expert, folds the down-projection scale into normalization, and enables owner-projected context and sharded greedy verification on the real mixed scheduler. All 75 eligible batches in the archived 16-user run processed 16,384 prompt rows, where the routing model predicts 5.64% fewer tensor-core rows in the two expert projections. The integrated scheduler replay passed 84 paired comparisons with real four- and eight-rank Gloo collectives. H200 throughput, memory, TTFT and agreement are unmeasured. [V211](serving_v0523/candidates/v211_prompt_expert_tiles_mixed_serving/README.md) · [Static checks](serving_v0523/candidates/v211_prompt_expert_tiles_mixed_serving/validation.json) · [Iteration](experiments/iterations/20261119-serving-v207-v211.json)

**Correction to the V202–V206 reports.** The queued V204/16 candidate runs the combined target and pipelined expert W2 but does not enable owner projection or sharded verification logits at the scheduler entry. Its earlier tests reached an alternate worker helper. V209 connects both endpoints to the actual call. At the last November 19 observation V93/16, V204/16, V120b/32 and V139/4 remained queued, with 3/3 private and 0/1 scored runs used. No V211 submission was made. [Actual scheduler source and regression](serving_v0523/candidates/v209_mixed_prompt_context_entry/README.md) · [Judge identities](serving_v0523/runs/nov19_initial_private_jobs.json)

**2026-11-19 00:03 UTC. The judge accepted three November 19 private checks.** V120b/32 `a73535cda6bf4cac` (queued) · V204/16 `7fcdf9abf39b486a` (queued) · V139/4 `b013f83007d544d3` (queued). The 16-user control is V68. The official four-load score remains V64’s 1.4119×. This uses the three private checks and zero scored runs. V206’s steady decode changes passed local correctness and compilation checks and have no judge timing. [Official job identities and quota](serving_v0523/runs/nov19_initial_private_jobs.json) · [V206](serving_v0523/candidates/v206_steady_decode_gate_up_pipeline/README.md)

**November 18, 23:55 UTC. V206 covers both steady decode expert projections.** In the new gate/up projection, Hopper now drains its matrix pipeline 64 times per warpgroup for a full K=4,096 output tile, down from 96. V205's down-projection takes 32 drains, down from 48 in V204. All three native tile widths retained 96 registers and zero spills. A joined CPU replay matched 2,359,296 final BF16 SiLU outputs. V206 is frozen and its H200 speed is unknown. [V206 implementation](serving_v0523/candidates/v206_steady_decode_gate_up_pipeline/README.md) · [Compiled instructions](serving_v0523/candidates/v206_steady_decode_gate_up_pipeline/sm90a.json) · [Iteration](experiments/iterations/20261118-serving-v206.json)

V64 still holds the official four-load score of 1.4119×. V93 was queued at 23:50 UTC. V204 is armed for the November 19 16-user private check if V93 stays pending and its quota and parent evidence pass. A passed V93 retains V98 and a failed V93 retains V95 under their existing gates. V205 and V206 have no judge submissions. [Private worker](serving_v0523/runs/nov19_v204_runner.launch.json) · [Four-load result](serving_v0523/runs/abd217baf0ed49fd.json)

**November 18, 23:43 UTC. V205 extends V204 with a steady decode expert down-projection. V64 still holds the official four-load score of 1.4119×.** The three grouped row tiles issue the same 32 Hopper matrix operations and reduce full waits from 12 to 8 per four-stage loop, retaining 96 registers with zero spills. All 13,824 FP32 outputs matched in the host replay. V205 has no H200 measurement. [V205 patch and validation](serving_v0523/candidates/v205_steady_decode_weight_pipeline/README.md) · [Compiled synchronization review](serving_v0523/candidates/v205_steady_decode_weight_pipeline/independent_release.md) · [Official V64 result](serving_v0523/runs/abd217baf0ed49fd.json)

V204 is armed for the November 19 16-user private check if V93 is still pending when the window opens at 00:03 UTC and the official comparison and quota gates pass. V93 was queued at activation. A confirmed V93 pass or failure retains the earlier V98 or V95 choice. If V204 takes the slot, the V173 and V180 branches lack their V168 parent and V176 lacks V98 or V95. All nine peer workers retained their locks, and all 28 attempt receipts matched across the change. Activation consumed zero check or score quota. [Private worker receipt](serving_v0523/runs/nov19_v204_runner.launch.json) · [V204's connected replay](serving_v0523/candidates/v204_mixed_prompt_weight_pipeline/joined.json) · [Iteration](experiments/iterations/20261118-serving-v205.json)

**November 18, 23:35 UTC. V204 combines mixed verification with a pipelined Hopper expert down-projection. V64 still holds the official four-load score of 1.4119×.** The W2 kernel retains the original 32 tensor-core instructions and cuts the full waits in the compiled four-stage loop from 16 to 8. The loader gains work and shared memory increases by 8 KiB per CTA. Separately, the target can process prompts and a running DFLASH cohort in one pass, compute the full sharded vocabulary for every verification position, and exchange a score and token ID per row and rank. At the largest archived 32-user opportunity the rank-major gather would fall from 191.92 MiB per GPU to a 0.77 MiB prompt gather and 15.5 KiB of tail winners. The schedule and tensor sizes are modeled. V204 has no measured H200 speedup. [V204 source and evidence](serving_v0523/candidates/v204_mixed_prompt_weight_pipeline/README.md) · [V199 machine code](serving_v0523/candidates/v199_prompt_expert_overlap/sm90a.json) · [V203 numerical replay](serving_v0523/candidates/v203_mixed_sharded_verify_head/independent_gloo.md)

The combined tree reconstructs in both donor orders and its 291,087-byte official export passes the public static rules. A connected replay sent 10,240 finalized BF16 expert values through a synthetic vocabulary projection and real Gloo verification, checking 933,750 logit values under the full and sharded paths. V201 is a separate owner-directed prompt collective with 408 real Gloo rank-cases and a Hopper reducer. V200 found that cross-block draft speculation still needs the target's true bonus and hidden features. [Connected replay](serving_v0523/candidates/v204_mixed_prompt_weight_pipeline/joined.json) · [V201](serving_v0523/candidates/v201_owner_partial_exchange/README.md) · [V200](serving_v0523/candidates/v200_cross_block_draft_feasibility/README.md) · [Iteration](experiments/iterations/20261118-serving-v199-v204.json)

**November 18, 22:54 UTC. V198 combines a cold-prompt expert fusion with segmented verification attention. V64 still holds the measured four-load score of 1.4119×.** At 16,384 live expert routes of width 4,096, V198’s long-prompt kernel removes the intermediate scale gather, scale multiply and SiLU launches, along with 1 GiB of logical reads and writes per eligible layer per rank. The original three small decode expert kernels still select their native configurations. In the allocator replay with fresh pages, later 32-user verification batches had 96.42% of history tiles eligible for TMA reads, up from about 9.0% under whole-partition eligibility. A heavily fragmented free list gave zero coverage for both. Every live history tile is still evaluated in logical order. The real eight-H200 effect on time, memory and model agreement remains open. [V198 patch and tests](serving_v0523/candidates/v198_segmented_verify_prompt_experts/README.md) · [Allocator assumptions](serving_v0523/candidates/v196_segmented_verify_tma/independent_serving_allocator.md)

The integrated tree passed both donor reconstruction orders, 120 CLI permutations, 288 grouped decode fixtures and the attention interpreter. Concrete BF16 attention output was replayed through the decoder-layer fixture and reproduced independently with a checked CPU matrix contraction. V194 is a separate scheduler candidate that can co-batch continuing prompts and running decoders when both the present and final prompt still fit their original capture buckets. A 14,000-sequence replay retained every row in order and the original prompt forward count at fixed capacity. A rerating of 767 archived client requests puts the modeled 4× crossover near 2.46× faster cold first-token latency and 2.46× faster post-first-token output together. V198 has no measured four-load factor. [Joined output](serving_v0523/candidates/v198_segmented_verify_prompt_experts/cross_phase.json) · [V194](serving_v0523/candidates/v194_mixed_decode_chunk_budget/README.md) · [Score headroom](serving_v0523/candidates/v195_fused_expert_affine_verify/score_headroom.md)

V93/16 was still queued at 22:54 UTC. The daily quota showed 3/3 private checks used and 0/1 scored runs used. V198 replaces V189 in the November 20 conditional 16-user slot. The slot opens from 04:00 to 08:00 UTC only after V129/1 and V141/32 have consumed the first two private checks, while V120b/32 stays pending and one private slot remains. All nine peer workers and 28 previous attempt records were verified. Activation used no quota. [Standby receipt](serving_v0523/runs/nov20_v198_runner.launch.json) · [Iteration](experiments/iterations/20261118-serving-v191-v198.json)

**November 18, 22:16 UTC. V190 combines a target decode kernel with a smaller draft page-table build. V64 still holds the best measured four-load score at 1.4119×.** For the three Hopper expert down-projection tile sizes selected during grouped target verification, V189 overlaps unpacking the next packed NVFP4 operand with an outstanding tensor-core multiply. Each compiled loop has 12 wait instructions, down from 16, and retains the same 32 ordered matrix instructions, 96 registers and zero spills. V190 also applies V188’s tile-aligned draft page-table suffix. At 16 users, 208 archived first-block shapes would require 441,578 live page IDs in total, down from 17,237,738. All five draft layers still attend to the same physical keys in the same split ranges. The change covers address preparation once per eligible draft pass. [V190 implementation](serving_v0523/candidates/v190_grouped_decode_draft_window/README.md) · [V189 Hopper instructions](serving_v0523/candidates/v189_decode_grouped_mma/sm90a.json) · [V188 trace and split replay](serving_v0523/candidates/v188_draft_window_metadata/independent_geometry.md)

V190 retains the V187 long-prompt endpoint. On an eligible 16,384-token prompt at TP8 it exchanges 122 MiB per GPU for the projected draft KV, versus 224 MiB for the prior normalized-feature gather. The integrated tree passed 200 native FA4 wrapper calls, 144 grouped expert dispatch/finalization cases, 864 arithmetic tests and seven inherited CPU integration suites. Its 158,066-byte official export passed the public static rules and strict full/delta reconstruction. V183 separately improved the grouped prompt W2 schedule. V184’s second target verification and V186’s hot-expert row migration did not justify a hardware check. Full H200 output agreement, peak memory, TTFT and serving throughput for V190 remain open. [Patch and manifest](serving_v0523/candidates/v190_grouped_decode_draft_window/validation.json) · [Prompt projection checks](serving_v0523/candidates/v185_paired_prompt_draft_kv/README.md) · [Iteration](experiments/iterations/20261118-serving-v183-v190.json)

V93/16 was still queued at 22:16 UTC. November 18 had used all three private checks. V68 remains the private 16-user reference at 17,512.9 tokens/s/GPU, 48.0 median decode tokens/s/user and +1.7 GB/GPU over stock. Reaching 4× would require another 2.833× over V64 across the scored loads. A single V189/16 private standby is armed for November 20 between 04:00 and 08:00 UTC if V129/1 and V141/32 take the first two slots, V120b/32 is still queued or running, and the judge reports exactly one check left. The November 19 experiments keep their places. V190 is available for comparison after the hardware evidence warrants it. [V64 official score](serving_v0523/runs/abd217baf0ed49fd.json) · [V68 private control](serving_v0523/runs/04b01f958aa5487c.json) · [V189 conditional reservation](serving_v0523/runs/nov20_v189_runner.launch.json)

**November 18, 21:29 UTC. V182 feeds a fused FP8 query and physical K/V writes directly into paged target verification. V64 still holds the best measured four-load score at 1.4119×.** An eligible layer goes from five preparation launches to one. Across 62 eligible layers that would remove 248 launches per verification pass. A V180/V182 replay matched 1,179,648 BF16 result components in 96 paired calls. A separate numerical run exercised the real query preparation, paged launcher and partition/merge bodies over fragmented caches and full or sliding windows. Its 65,536 BF16 outputs had maximum relative L2 error 0.002152 against float64 dense attention. Two archived H200 attention tests at the same verification shape suggest 0.428–0.429 ms per pass across the 62 layers. Against V68’s illustrative 83.33 ms decode cycle, a full transfer would yield about 1.005× for this fusion alone. CUDA graph timing, exact responses, TTFT and peak memory need H200 measurement. [V182 source and tests](serving_v0523/candidates/v182_paged_verify_fused_qk/README.md) · [V64 scored run](serving_v0523/runs/abd217baf0ed49fd.json)

V181 combines direct NVLink reads of routed prompt rows with a paired gate/up projection. A distributed host replay traced 1,180,800 BF16 output components through both optimizations and the original sparse return. At 16,384 local expert routes the fused activation avoids 512 MiB of logical intermediate reads and writes per rank and eligible layer. V180 isolates direct FP8 prompt K/V on the H200-tested V68 decoder. In the archived 16-user trace, 60 of 75 long prompt forwards were single-sequence calls. V68’s private result was 17,512.9 tokens/s/GPU with 1.7 GB/GPU above stock. [V181 combined replay](serving_v0523/candidates/v181_peer_inputs_paired_gate_up/README.md) · [V180 source and trace](serving_v0523/candidates/v180_paged_verify_direct_prompt/README.md) · [V68 private comparator](serving_v0523/runs/04b01f958aa5487c.json)

The November 18 private quota is exhausted and V93/16 was still queued at 21:21 UTC. V180/16 is armed only for a disjoint third private slot on November 20 if V168 has an attributable hardware failure, both controls fail, and V129/1 then V163/16 occupy the first two slots. Activation preserved all eight earlier waiters and 28 existing receipts and consumed zero quota. V181 and V182 have no hardware reservation. Reaching 4× requires another 2.833× over the best measured result. [Standby receipt](serving_v0523/runs/nov20_v180_runner.launch.json) · [Iteration](experiments/iterations/20261118-serving-v177-v182.json)

**November 18, 20:52 UTC. V176 combines a new Hopper decode kernel with three prompt and verification changes. V64 still holds our best official four-load score at 1.4119×.** The V174 kernel handles short and long expert lists together, reducing the eligible gate/up and down projection pair from four launches to two. The integrated tree also reduces eligible FA3 schedule preparations from 62 to two and cuts work on long prompts through grouped expert tiling and token-owner projection. Independent replay matched 2,515,968 BF16 expert values. A prompt-to-verify test exercised 324 prompt forwards and 27 verifications including failure recovery. The frozen 195,416-byte patch passed the published static restrictions and CLI checks. Physical correctness, speed, TTFT and peak memory are open. V64's scored memory was 1.9 GB above stock against a 2.0 GB limit. [V176 candidate](serving_v0523/candidates/v176_integrated_prompt_decode/README.md) · [V174 machine code](serving_v0523/candidates/v174_multirow_expert_tma/sm90.json) · [V175 rejected microbatch split](serving_v0523/candidates/v175_decode_collective_pipeline/README.md)

V68's private 16-user check reached 17,512.9 tokens/s/GPU versus V64's 17,349.9 on a different node. Each node's stock-normalized indicative ratio was 1.424× and 1.412× respectively. The 4× target still requires another 2.833× across the scored loads. The November 18 private quota is exhausted and V93/16 was queued at 20:46 UTC. A conditional V176/16 worker is armed for the last private slot on November 20, after confirmed V129/1 and passing V163/16, both failed controls, and evidence that November 19 selected V98 or V95 instead of V168. It requires fresh quota of exactly one private check. All seven preexisting waiters retained their locks and all 28 attempt records were identical at activation. [Conditional run receipt](serving_v0523/runs/nov20_v176_runner.launch.json) · [Measured score and trace headroom](serving_v0523/candidates/v176_integrated_prompt_decode/independent_frontier.md) · [Iteration](experiments/iterations/20261118-serving-v174-v176.json)

**November 18, 20:21 UTC. V173 targets medium-batch decode on the V168 stack. V64 remains the best official result at 1.4119× across four loads.** V173 gives experts with one to eight live verification rows a Hopper producer warp to stage weights and inputs. Busier experts retain the existing 16/32-row kernels and their packed-weight reuse. The new kernels compile for SM90 and SM90a without spills. A 33-case numerical replay matched 1,115,520 BF16 values. At 16 requests uniform routing puts 95.35% of routes on the new path. A separate 256-row device histogram puts just 59 of 354 local routes there. Actual H200 speed and the cost of two extra launches per eligible layer remain open. [Candidate and exact flags](serving_v0523/candidates/v173_medium_expert_tma/README.md) · [Measured and modeled geometry](serving_v0523/candidates/v173_medium_expert_tma/independent_headroom.md)

V172 reduces FA3 scheduler preparation from 62 to two launches in eligible decode verification. V171 reduces the two long-prompt all-gather outputs from 1,536 MiB per rank to 256.125 MiB with a single 16,384-token sequence. V169 reduces modeled grouped expert row operations by 18.9% for a uniform 12,288-token prompt chunk and 1.8% for a full 16,384-token chunk. V170 found that the five cold-prompt chunks already run FA3 with one split. Its alternative would increase layer-weighted tile visits. The four implementation candidates await physical measurement. [Verify scheduling](serving_v0523/candidates/v172_streamed_experts_verify_schedule/README.md) · [Prompt endpoints](serving_v0523/candidates/v171_prompt_endpoint_context/README.md) · [Grouped expert tiles](serving_v0523/candidates/v169_grouped_expert_tile_cohorts/README.md) · [Attention study](serving_v0523/candidates/v170_long_prefill_attention_feasibility/README.md)

V93/16 was queued at 20:29 UTC. All three November 18 private checks were consumed. The November 19 private runner opens at 00:03 UTC and now selects **V168/16** if V93 is still pending at selection. The private V64 reference achieved 17,349.9 tokens/s/GPU at 16 users. Arming the runner consumed no quota. V173/16 is armed for the otherwise unused third private slot on November 20, 00:03–07:59 UTC. It requires a passing, nontruncated V168/16 comparator, V129/1 and V163/16 in the first two slots, both displaced controls failed and exactly one check remaining. The new worker retained all six other waiters and spent no quota on activation. [V173 standby](serving_v0523/runs/nov20_v173_runner.launch.json). [Armed runner](serving_v0523/runs/nov19_integrated_verify_runner.launch.json) · [Iteration evidence](experiments/iterations/20261118-serving-v169-v173.json)

**November 18, 19:51 UTC. V168 combines V165’s low-load decode kernel with V167’s direct return of expert outputs on long prompts. Our best official four-load result is V64 at 1.4119×.** On Hopper, a producer warp stages the next expert weights, scales and active input rows while eight consumer warps compute. The launch uses measured device occupancy to decide whether more expert jobs can run concurrently. With uniform top-eight routing and four resident thread blocks per SM, the model’s longest block completes 1.00 full projection sweeps on average for an eight-row verification pass, down from 1.98. The source compiled for SM90 and SM90a. H200 throughput is unmeasured. [V168 integrated patch](serving_v0523/candidates/v168_streamed_experts_sparse_return/README.md) · [V165 scheduling and validation](serving_v0523/candidates/v165_streamed_expert_stages/README.md) · [Official V64 score](serving_v0523/runs/abd217baf0ed49fd.json)

V167 sends each expert rank’s final BF16 partial only to its token owner, which accumulates contributions in FP32 and rounds once. At 16,384 rows, uniform routing gives 148.3 MiB sent per GPU per eligible layer across input and return. V64’s ring-equivalent volume is 336 MiB. The physical NCCL algorithm and bandwidth are unknown. V168 preserves the original window budget after compacting expert inputs. With 10,844 owned tokens and 16,384 live routes it needs one pair of grouped projections instead of two. The merged path matched 2,943,360 BF16 owner values in 1,344 distributed CPU replays. H200 agreement, exact output lengths, peak memory, TTFT and four-load throughput remain open. [Sparse transport](serving_v0523/candidates/v167_sparse_expert_return/README.md) · [Combined replay](serving_v0523/candidates/v168_streamed_experts_sparse_return/prompt.json) · [Independent window sweep](serving_v0523/candidates/v168_streamed_experts_sparse_return/independent_budget.json)

In the archived eight-H200 V41 client trace, 64.6% and 69.9% of wall time at 16 and 32 users had every outstanding request waiting for its first turn’s first token. These are client-visible intervals. V166’s alternate speculative selector reduced synthetic expected commits from 4.00 to 3.82 when the target was sharpened to reproduce V64’s one-user mean. Keep V166 offline. V93/16 remained queued at 19:50 UTC, with all three November 18 private checks consumed and one scored run still available. The November 19 and 20 allocations stand. [Trace measurements](experiments/iterations/20261118-serving-client-overlap.json) · [Selector study](serving_v0523/candidates/v166_bellman_draft_chain/quality.json) · [Checks, hashes and score](experiments/iterations/20261118-serving-v165-v168.json)

**November 18. V164 combines a new Hopper decode kernel with sparse communication for long prompts. The best official four-load serving score is still V64 at 1.4119×.** In V161, Hopper’s Tensor Memory Accelerator loads each expert’s packed weights and scales. The compiled 128-feature loop shrank from 987 to 728 static instructions for gate/up and from 681 to 426 for down projection. The tensor-core operations and rounding order match the parent. V163 sends each prompt activation only to ranks whose experts need the row. For a 16,384-row chunk with uniform top-eight routing, expected remote activation traffic falls from 112 to 74.1 MiB per rank per eligible layer. H200 timing, output agreement, TTFT and peak memory are open. [V164 and exact flags](serving_v0523/candidates/v164_tma_experts_sparse_prompt/README.md) · [Compiled Hopper evidence](serving_v0523/candidates/v161_tma_expert_channels/sm90.json) · [Prompt exchange on V64](serving_v0523/candidates/v163_sparse_prompt_exchange/README.md)

The combined source passed 864 CPU replay cases and 672 distributed CPU rank cases. Independent checks compared 3,932,160 BF16 elements across skewed, full, empty and capacity-boundary expert windows. The 140,262-byte official export applied and reversed against the published pin. V162 found little one-user headroom in changing full-attention addressing. Same-node V48b/V64 data also showed how a faster median stream could still produce lower end-to-end throughput. [Combined validation](serving_v0523/candidates/v164_tma_experts_sparse_prompt/validation.json) · [Window checks](serving_v0523/candidates/v164_tma_experts_sparse_prompt/independent_windows.json) · [V162 measurements](serving_v0523/candidates/v162_dense_verify_span_feasibility/README.md) · [One-user phase data](experiments/iterations/20261118-serving-one-user-phase.json)

V93/16 was queued at 19:21 UTC on November 18, with all three daily checks used and one scored run available. A separate November 20 worker can check V163/16 against the matched V64 private result of 17,349.9 tokens/s/GPU. It uses the second private slot only after V129/1 is accepted, both November 19 controls (V120b/32 and V139/4) have confirmed terminal failures, the other conditional workers have made no attempts and the judge reports two private checks left. The existing November 19 and 20 allocations keep priority. V164 has no judge job. [Conditional private runner](serving_v0523/runs/nov20_v163_runner.launch.json) · [Iteration and queue receipts](experiments/iterations/20261118-serving-v161-v164.json) · [Official four-load result](serving_v0523/runs/abd217baf0ed49fd.json)

**November 18, 19:01 UTC. V160 now combines prompt processing with speculative decode verification in the normal serving scheduler. V64 remains the measured four-load leader at 1.4119×.** In 96 end-to-end CPU rounds the mixed target ran once, committed 1,008 tail tokens, and published the correct lengths and bonus tokens through the following filtered decode. A veto reuses the same draft proposal. The source export and five launch flags passed the published patch checks. H200 speed, numerical agreement, memory and TTFT are open. [V160 candidate](serving_v0523/candidates/v160_mixed_dflash_serving/README.md) · [Independent replay](serving_v0523/candidates/v160_mixed_dflash_serving/independent_review.md)

At four, 16 and 32 users the archived V41 trace had 66, 99 and 55 prompt passes where every running request could fit within the two original graph budgets. One user had zero opportunities. V93/16 is still queued. All three Nov 18 private checks are used, and the scheduled Nov 19 and Nov 20 runners retain their existing allocations. [Iteration, queue and quota](experiments/iterations/20261118-serving-v155-v160.json) · [Measured V64 score](serving_v0523/runs/abd217baf0ed49fd.json)

**November 18, 18:24 UTC. V153 targets sparse decode at one and four users. V64 remains the best official four-load score at 1.4119×.** V153 distributes a 128-channel expert projection over eight warps. It compiles for Hopper, preserves each output’s accumulation order, and exposes 1,837 productive warps per rank in the uniform one-user routing model, up from 459. Hardware timing is open. V152 separately reproduced combined prompt and eight-row native verification in 64 CPU batches. Its production integration still needs graph replay and correct cache settlement. [Sparse expert kernel](serving_v0523/candidates/v153_cooperative_expert_channels/README.md) · [Mixed verification reference](serving_v0523/candidates/v152_mixed_prefill_verify_reference/README.md) · [Iteration evidence](experiments/iterations/20261118-serving-v152-v154.json)

V154 coalesces contiguous device-to-host KV backup copies on V151. V154c isolates the same change on scored V64. A fragmented allocator fixture cut 266 submissions to 12 with identical contents. V154c/16 is armed for a remaining November 20 private check only if one competing control has failed and two other checks were accepted. The matched V64 private comparator is 17,349.9 tokens/s/GPU. V93/16 was still queued at 18:21 UTC. The November 19 plan retains V147/16 while V93 is pending. Three November 20 runners now bind the load to its exact CLI receipt when the judge omits it from a pending response. All 155 runner regressions passed. [V154c control and check conditions](serving_v0523/candidates/v154c_v64_hicache_backup/README.md) · [Live schema verification](serving_v0523/runs/nov20_pending_schema.cpu.json) · [Runner activations](serving_v0523/runs/nov20_pending_schema.rollover.json)

**November 18, 17:55 UTC. V151 adds native FA3 schedule reuse to V147’s decode path. The best official four-load result is V64 at 1.4119×.** On a qualifying 62-layer verification, V151 prepares one full-attention and one sliding-window schedule. The independent page-one test compared all other native attention arguments across 1,488 calls. V151 uses V147’s four flags plus `--speculative-dflash-share-verify-schedule`. Its export is 123,180 bytes, SHA-256 `2923c97cf66ca37c2c27ded28b53658e1a6d6ea5da3e828907da46fb7290b543`. Its H200 speed, agreement, exact output lengths, TTFT and peak memory are open. [V151 evidence](serving_v0523/candidates/v151_integrated_verify_scheduler/independent_review.md) · [Official V64](serving_v0523/runs/abd217baf0ed49fd.json)

V150 sends long-prompt attention output directly to its row owners on V147. Four- and eight-rank Gloo fixtures checked 403,188 result elements, and the final reducer compiled for Hopper without spills. V149 overlaps the original grouped expert route sort with the large prompt-input gather on V64. V148’s best exact dequantization variant saved eight of 516 static loop instruction sites, below the gain needed to justify a check. V149, V150 and V151 are hardware reserves. The Nov 19 waiter still selects V147/16 if V93 is pending. [V150](serving_v0523/candidates/v150_owner_output_projection/README.md) · [V149](serving_v0523/candidates/v149_overlapped_expert_route_planning/README.md) · [V148](serving_v0523/candidates/v148_expert_collective_dequant_feasibility/README.md) · [Iteration](experiments/iterations/20261118-serving-v148-v151.json)

**November 18, 17:29 UTC. V147 is armed for a private 16-user check while V93 is pending. V64 leads the official four-load score at 1.4119×.** V147 combines V144’s pipelined sparse expert projections with V146’s zero-copy FA3 prefill on V64’s four flags. The export is 113,650 bytes, SHA-256 `7a3d4dc6c0e526b60ca50bc975a6036da7796d8eb05f9e8afb70423164425437`. Offline tests covered 538,624 newly written FP8 bytes and all 62 target verification layers. H200 throughput, output agreement and lengths, peak memory and first-token latency remain unmeasured. V145’s committed-row experiment retained the same single-user matrix tile and cold-weight traffic. [V147](serving_v0523/candidates/v147_integrated_experts_prefill/README.md) · [V145](serving_v0523/candidates/v145_committed_row_feasibility/README.md) · [Scored V64](serving_v0523/runs/abd217baf0ed49fd.json)

The November 19 private waiter runs from 00:03 to 22:00 UTC. Current evidence selects V120b/32, V147/16 and V139/4. November 19 also has one separate scored run available under the daily quota. No scored submission is armed. On November 20 the private order is V129/1, V132/32, V141/32 if V120b/32 passes in time. From 00:13 UTC V141 can take slot two with confirmed V129 acceptance, eligible V139 and exact quota. V132 can then use slot three if V141 is confirmed and V120b/32 passes. The official kernel phase is closed despite the daily counts shown by `okc problems`. Serving closes at 08:00 UTC on November 20. [November 19 waiter](serving_v0523/runs/nov19_integrated_verify_runner.launch.json) · [November 20 allocation](serving_v0523/runs/nov20_slot_arbitration.rollover.json) · [Experiment ledger](experiments/iterations/20261118-serving-v144-v147.json) · [Official phases](https://openkernelchallenge.dev/api/v1/serving)

**November 18, 17:05 UTC. V141 is armed for the third November 20 private check at 32 users.** V141 compacts actual expert routes during medium-density verification on V139. At 256 verification rows a uniform-routing model predicts 361 and 362 MMA row slots per rank for its two projections, versus about 768 aligned rows each for the parent. Its emulated projections matched 64,384 BF16 activations and 77,056 outputs. The runner requires confirmed exact V129/1 and V132/32 submissions, exact V139/4 queued, running or passed, and two private checks used with one remaining. **V64 still holds the measured four-load score of 1.4119×.** V93 remains queued; V88 crashed on a missing import repaired in V88a. [V141 experiment](serving_v0523/candidates/v141_adaptive_expert_projection/README.md) · [Nov 20 third-slot receipt](serving_v0523/runs/nov20_v141_runner.launch.json) · [V64 official result](serving_v0523/runs/abd217baf0ed49fd.json)

V142 separately shards both BF16 expert projections on V139. It saves 112 MiB of parameter storage per sparse layer at TP8; long-prompt communication and actual peak memory still need measurement. The 93,452-byte frozen patch is an offline candidate with no reservation. V140’s proposed draft projection overlap was declined after the source trace showed that ordinary V64 greedy decode already keeps committed lengths on the device. V143 is frozen on V64. It uses zero-copy FA3 cache views when the physical prefill KV layout permits them. Its 28 offline Hopper specializations, 243 interpreter cases and 63 paired FA3 wrapper cases still need H200 validation. V143 has no check reservation. [V142](serving_v0523/candidates/v142_sharded_expert_projections/README.md) · [V140 feasibility](serving_v0523/candidates/v140_decode_overlap_feasibility/README.md) · [V143 export](serving_v0523/candidates/v143_zero_copy_prefill_spans/validation.json) · [Iteration and evidence](experiments/iterations/20261118-serving-v140-v143.json)

**November 18, 16:38 UTC. V139 combines three serving optimizations on scored V64.** It uses V129’s low-density routed expert kernels, V135’s compact scaled expert prefill and V136’s fused query/key normalization, rotary embedding and cache store during target verification. V136 keeps the physical cache bound in stable device storage during replay. Its validation covers 24 offline Hopper specializations, 2,402,816 interpreter bytes and all 62 native attention-layer operand cases. The V139 export is 84,277 bytes, SHA-256 `c4428099805f991512ff14783368efdada1985b6a4b15c039a17aa54ac4782ed`. The November 19 runner is armed for V139/4 in its third private slot. With V93 queued, the other two slots select V120b/32 and V102/16. V126 is now an offline reserve. November 20 retains V129/1, conditional V132/32 and one spare private check. **V64’s 1.4119× remains the best measured four-load score.** [Nov 19 runner](serving_v0523/runs/nov19_integrated_verify_runner.launch.json) · [V139 export](serving_v0523/candidates/v139_integrated_experts_verify/validation.json) · [V136 evidence](serving_v0523/candidates/v136_fused_verify_qk_rope_cache/validation.json) · [V64 score](serving_v0523/runs/abd217baf0ed49fd.json)

V88’s four-user check crashed during draft graph warmup when `candidate_topk` called `is_cuda` without importing it. It produced no throughput result. V88a repairs that import. The separate V93 patch already contains the import and its 16-user job remains queued. V134 preserves a write-back HiCache policy experiment; V137 packs unequal full-attention prefixes within a bounded buffer; V138 overlaps BF16 QKV projection with a token-shard ring exchange during prefill. These alternatives await H200 measurement. [V88 judge log](serving_v0523/runs/30bb36feb528416f.log) · [V88a repair](serving_v0523/candidates/v88a_draft_head_capture_fix/README.md) · [Iteration and candidate evidence](experiments/iterations/20261118-serving-v134-v139.json)

**November 18, 16:00 UTC. V129 is armed for a one-user private check at 00:03 UTC on November 20.** Its low-density expert kernels on V64 compiled for Hopper without spills and matched the CPU reference across 35,968 BF16 activations and 33,792 BF16 outputs. V64 retains the official four-load score of **1.4119×**. V88 and V93 remain queued. Today’s results leave the November 19 runner in place, currently selecting V120b/32, V102/16 and V126/1. [V129 runner](serving_v0523/runs/nov20_v129_runner.launch.json) · [Nov 19 runner](serving_v0523/runs/nov19_decode_runner.launch.json) · [V64 score](serving_v0523/runs/abd217baf0ed49fd.json)

V132 shares full- and sliding-window FA3 scheduler metadata within one V120b target verification. Eligible long calls need two scheduler prepares across 62 layers, a projected saving of 60 launches. The frozen source starts `45c62d`. The conditional V132/32 private runner armed at 15:59 UTC. It requires confirmed submission of V129/1, a passing V120b/32 control, and a November 20 quota showing one check used and two left. That leaves the last private slot available. V133 separately writes prompt draft KV directly on V120b, preserving the ordinary writer’s slot-zero and bounds checks; it can remove five scatters and 79.69 MiB of logical traffic per full eligible chunk. Both await H200 verification. [V132 runner](serving_v0523/runs/nov20_v132_runner.launch.json) · [V132 export](serving_v0523/candidates/v132_verify_scheduler_metadata/validation.json) · [V133 direct writes](serving_v0523/candidates/v133_direct_prompt_draft_kv/README.md)

**V130 is withdrawn.** The executable CPU witness overwrote 64 bytes in reserved V slot zero and wrote into the guard at slot nine of a capacity-nine pool. V131’s exact two-pass verification proposal was declined because dense-weight rereads erased most of the modeled one-user saving and increased modeled traffic at higher loads. [V130 slot witness](serving_v0523/candidates/v130_streamed_prompt_draft_kv/reserved_slot_witness.json) · [V131 analysis](serving_v0523/candidates/v131_exact_verify_pruning/README.md) · [Iteration and hashes](experiments/iterations/20261118-serving-v129-v133.json)

**November 18, 15:03 UTC. V126 is armed for a one-user private check.** V64 remains the best four-load score at **1.4119×**. V126 splits active experts over two or four Hopper clusters and keeps all partial sums in FP32 until the existing BF16 output boundary. At one user the uniform-routing model predicts 230 blocks doing projection work versus 57 for an unsplit clustered implementation. The candidate compiled for SM90/SM90a. All 82,944 tested FP32 partials and 65,536 integrated BF16 outputs matched their CPU references. Independent tests confirmed that FP32 reassociation can change rounded BF16 results. Model agreement, peak memory, time to first token and throughput await H200 execution. [Candidate](serving_v0523/candidates/v126_split_cluster_expert_mlp/README.md) · [Independent numerical review](serving_v0523/candidates/v126_split_cluster_expert_mlp/independent_review.md) · [Iteration](experiments/iterations/20261118-serving-v126-v128.json)

The Nov 19 00:03 UTC runner currently selects V120b/32, V102/16, and V126/1. V126 replaces the V125/16 fallback. It uses V64’s measured one-user rung for the whole-build comparison and V68’s passing 16-user check to establish its software lineage. A confirmed V93 pass selects V101b/16 for the third slot if its dependencies qualify. V88/4 and V93/16 are queued. V127 and V128 produced feasibility results on shorter native verification and compact draft embeddings. Both remain outside the private allocation. [Runner](serving_v0523/runs/nov19_decode_runner.launch.json) · [V127](serving_v0523/candidates/v127_native_prefix_verify/README.md) · [V128](serving_v0523/candidates/v128_compact_noise_embedding_feasibility/README.md)

**November 18, 13:51 UTC. V125 moves the clustered expert kernel to measured V68.** V125 fuses gate/up, scaled SiLU and down projection for eligible expert tiles. Across 62 eligible layers it can remove up to 124 launches per target pass. Its integrated CPU test matched 524,288 BF16 outputs for remote-only, mixed, skewed and duplicate routes. Offline SM90a assembly reports 126 registers per thread and zero spills. Physical H200 cluster residency, shared-memory traffic, agreement, memory, TTFT and throughput remain open. V68 provides the matched 16-user baseline of 17,512.9 tokens/s/GPU and 48.0 median decode tokens/s/user. **V64 leads the full four-load score at 1.4119×.** [V125](serving_v0523/candidates/v125_v68_clustered_expert_mlp/README.md) · [CPU integration](serving_v0523/candidates/v125_v68_clustered_expert_mlp/integrated.json) · [V68 check](serving_v0523/runs/04b01f958aa5487c.json) · [V64 score](serving_v0523/runs/abd217baf0ed49fd.json)

V87’s four-user private check crashed before producing throughput. The judge reports `TypeError`; the captured server output contains no Python traceback, so the failing component remains unidentified. V88 and V93 remain queued. V120b corrects the native Muse processor dispatch for prompt projection; its 144 distributed CPU calls exercised TP2/4/8. V122 retains the clustered kernel on the V117 branch. V123b uses rank-major draft filter coefficients in eager execution and restores the original coefficient path under whole-module compilation. V121 found the greedy target head already sends row winners; V124 found full prompt chunks exceed the 16 MiB multicast pull workspace. Both remain feasibility studies. [V87 crash](serving_v0523/runs/00c134c978104d65.log) · [V120b native dispatch](serving_v0523/candidates/v120b_native_muse_prompt/README.md) · [Iteration and receipts](experiments/iterations/20261118-serving-v120b-v125.json)

The Nov 19 private runner opens at 00:03 UTC. With current receipts it selects V120b/32 first, V102/16 second and V125/16 third. The third slot selects V125 while V93 is queued, running or failed and exact V68 has passed. A passing V93 with eligible shared parents selects V101b/16 instead. An unknown V93 state defers the slot. All three are private checks; this plan schedules zero scored actions. [Live runner receipt](serving_v0523/runs/nov19_decode_runner.launch.json)

**November 18, 13:03 UTC. V120 is armed for the conditional 32-user check.** It splits the DFLASH context projection across the eight GPUs for target-confirmed decode and keeps the full weight for the existing sequence-parallel prompt path. At 256 verify rows the per-GPU projection falls from 171.8 to 21.5 GFLOP and uses 80 MiB of a resident 640 MiB weight. A 3.5 MiB peer exchange restores the full channels before normalization. Prefix-valid draft KV writes also save five launches per eligible append. Distributed CPU tests matched normalized BF16 inputs, lengths, positions and physical slots at TP2, TP4 and TP8, including the actual 40,960-wide reduction. V64 leads at **1.4119×**. V120’s end-to-end performance, peak memory and model outputs await the judge. [V120](serving_v0523/candidates/v120_phase_parallel_draft_context/README.md) · [Official V64 score](serving_v0523/runs/abd217baf0ed49fd.json)

The November 19 window opens at 00:03 UTC. Its first private check selects V120/32 while V87 is pending and the frozen V64 result is confirmed, V116b/4 after a V87 pass, or V80a/16 after a confirmed failure or skip. The latest official query showed V87/4, V88/4 and V93/16 queued. V117 is separately frozen for heavy decode, where it maps 2,048 expert IDs once per layer and preserves the stable order of every route. It matched 38,160,027 live BF16 values in the interpreter and compiled for Hopper without spills. V118’s tree budget showed a synthetic 5.1% average gain at 32 users, and V119’s normalization shortcut changed a BF16 output. Neither was scheduled. [V117](serving_v0523/candidates/v117_tiled_route_staging/README.md) · [Runner](serving_v0523/runs/nov19_decode_runner.launch.json) · [Iteration](experiments/iterations/20261118-serving-v117-v120.json)

**November 18, 12:36 UTC. V116b is armed for the four-user decode check if V87 passes.** V116b extends V89 by staging the IDs of all active local experts during normal routing. At native verification width eight, its W13 projection launches 640 thread blocks per layer for one request and 1,536 for four, down from 2,048 in V89. Every routed expert still runs. The final Hopper kernel uses 80 registers, has zero spills and emits 324 instructions in the inner loop. The stager matched 5,484,917 BF16 activations across 80 interpreter launches. H200 throughput, model agreement, output lengths, TTFT and peak memory await the judge. V64 remains the measured leader at **1.4119×** against the 4× objective. [V116b export](serving_v0523/candidates/v116b_staged_active_experts_regcap/patch.diff) · [Decode tests](serving_v0523/candidates/v116_staged_active_experts/README.md) · [V64 official score](serving_v0523/runs/abd217baf0ed49fd.json)

The November 19 window opens at 00:03 UTC. Its first private slot selects V116b/4 after an exact V87 pass, V111/32 while V87 is pending with the verified V64 receipt, and V80a/16 after a confirmed failure or skip. An earlier V89 or V116 attempt occupies that slot. The V87/4, V88/4 and V93/16 checks are queued. V113 separately reduces modeled QKV communication by 3.39 GiB per GPU over a long chunk. V115 replaces nine statically recorded expert waves with one device-controlled loop body and covers trailing tokens with zero local routes. Both prefill candidates await hardware. [Runner](serving_v0523/runs/nov19_decode_runner.launch.json) · [V113](serving_v0523/candidates/v113_paired_qkv_prefill/README.md) · [V115](serving_v0523/candidates/v115_conditional_expert_waves/README.md) · [Iteration](experiments/iterations/20261118-serving-v112-v116b.json)

**November 18, 11:56 UTC. V111 replaces V106 in the conditional 32-user check.** V111 combines V106’s paged prefill with a sequence-parallel projection for the draft and a fused query conversion and KV write during target verification. At TP8 the projection uses one eighth of the previous computation per GPU. The verify change can remove three launches in each of 62 target layers. The integrated export passed 69 distributed projection cases, 90 prefill dispatch cases and 96 verify dispatch cases. Its H200 performance, exact output lengths, agreement, TTFT and peak memory await the judge. V64 leads at 1.4119×. The 3× milestone and 4× objective remain open. At 00:03 UTC on November 19 the first private check selects V111/32 if V87 is pending and the V64 scored receipt passes, V89/4 if V87 passes, or V80a/16 upon a confirmed V87 failure or skip. [V111](serving_v0523/candidates/v111_paged_prefill_verify/README.md) · [V64 score](serving_v0523/runs/abd217baf0ed49fd.json) · [Private runner](serving_v0523/runs/nov19_decode_runner.launch.json) · [Iteration](experiments/iterations/20261118-serving-v107-v111.json)

**November 18, 11:28 UTC. V106 is armed for the 32-user comparison while V87 is pending.** V106 keeps V64’s decoder and combines 160-token pages on 16 full-attention layers with dense FP8 attention on 46 sliding-window layers during long prefills. The packer matched 89,743,360 bytes in the Triton interpreter. A separate oracle matched 77,312 attention values. The target Hopper binary compiled without spills. Packing adds about 413 MiB of traffic per GPU for a full chunk. V104 separately eliminates five draft KV launches per eligible decode verification. Both speedups need H200 measurements. At 00:03 UTC on November 19 the runner chooses V106/32 if V87 is pending and the V64 scored receipt passes, or V89/4 if V87 passes. V64 leads the four-load score at 1.4119× against the 4× target. [V104 decode](serving_v0523/candidates/v104_v64_direct_draft_kv/README.md) · [V106 prefill](serving_v0523/candidates/v106_paged_swa_suffix_prefill/README.md) · [Runner](serving_v0523/runs/nov19_decode_runner.launch.json) · [Iteration](experiments/iterations/20261118-serving-swa-prefill.json)

**November 18, 10:47 UTC. V68 passed at 16 users. V64 still leads at 1.4119× across four loads.** V68 measured 17,512.9 tokens/s/GPU against V64’s 17,349.9 in their respective one-repetition checks. Median decode fell from 48.8 to 48.0 tokens/s/user. Both used 1.7 GB/GPU above stock. The scored slot remains open. V101b tests whether intermediate branch depths can raise accepted tokens per target call at fixed width eight. V102 removes the large-prefill CPU routing synchronization on the measured V64 decoder. For November 19 the second private slot selects V102/16 while V93 is pending with a passing V64 receipt, or V98/16 after an eligible V93 pass. Such a pass also selects V101b/16 in the third slot. V79a remains the width-12 alternative. [Official V68 check](serving_v0523/runs/04b01f958aa5487c.json) · [V101b](serving_v0523/candidates/v101b_endpoint_guarded_spine/README.md) · [V102](serving_v0523/candidates/v102_v64_graph_grouped_prefill/README.md) · [Runner](serving_v0523/runs/nov19_decode_runner.launch.json) · [Iteration](experiments/iterations/20261118-serving-v101-v102.json)

**November 18, 09:56 UTC. V100 preserves branched decode on 160-token KV pages.** Verified target bytes move into sequential slots so eligible prompt chunks can use Hopper’s paged attention path. Local tests checked 64.9 million cache bytes and 552,960 attention output values. All 26 Hopper specializations compiled without spills. V100c isolates the page and chunk changes on V64. The first November 19 private slot selects V100c at 32 users while V87 is pending and V64’s scored receipt passes, or V89 at four users if V87 passes. The full V100 patch has no slot assigned. V64 remains the measured leader at **1.4119×**. On its stock curve, 4× requires a peak of at least 39,984 tokens/s/GPU, against V64’s 20,547. [V100 evidence](serving_v0523/candidates/v100_paged_tree_verify/README.md) · [V100c control](serving_v0523/candidates/v100c_v64_paged_prefill_control/README.md) · [Runner receipt](serving_v0523/runs/nov19_decode_runner.launch.json) · [Iteration record](experiments/iterations/20261118-serving-paged-tree.json)

**November 18, 09:16 UTC. V99 is frozen and eligible for a matched 16-user check if V93 passes.** It chooses the native eight-row chain or branching tree per request using one score from previously verified tokens. Its export passed 12,832 exhaustive comparisons, 1,008 production-kernel CPU cases, a two-rank worker broadcast test and 14 Hopper compiler checks. The third November 19 slot now selects V99 after an exact V93 pass, or retains V79a under its existing eligibility rules. The V89 and V98 comparisons remain armed. V64 leads completed four-load measurements at **1.4119×**. [V99 evidence](serving_v0523/candidates/v99_adaptive_draft_tree/README.md) · [Runner receipt](serving_v0523/runs/nov19_decode_runner.launch.json) · [Iteration record](experiments/iterations/20261118-serving-adaptive-draft.json)

**November 18, 08:10 UTC. V98 is armed for a matched 16-user check against V93.** Its grouped expert prefill plans complete tokens on the GPU and removes the large-prefill host synchronization and eager graph break. The 219,918-byte patch passed an exact V93 reconstruction, arbitrary-skew tests, two-rank readiness checks and offline Hopper compilation. The November 19 runner now allocates its second slot to V98, with V95 after a confirmed V93 or V68 failure. V98 replaces V84a in that window. V64 remains the measured leader at 1.4119× across all four loads. [V98 and test results](serving_v0523/candidates/v98_graph_grouped_prefill/README.md) · [Live runner receipt](serving_v0523/runs/nov19_decode_runner.launch.json) · [Iteration record](experiments/iterations/20261118-serving-graph-prefill.json)

**November 18, 07:23 UTC.** V93 combines the new high-load expert down projection, target greedy collective and dense FP8 prefill on V89. Its 16-user private check is queued as `75557c21595347de`. V87/4 and the V88/4 control took the other two November 18 checks. V64 remains the measured four-load leader at 1.4119×. The [current handoff](CODEX_HANDOFF.md) covers the 4× objective, evidence and remaining checks. [V93 evidence](serving_v0523/candidates/v93_expert_argmax_dense_prefill/README.md)

**Nov 17 23:21 UTC. V84 combines three decode changes on V80's native eight-node tree.** V81 normalizes each local participating token once and stages it for its routed experts. V82 uses the actual accepted tree rows for the draft context projection. V83 shards the target router's output columns and gathers its FP32 logits alongside the BF16 pre-expert projection. The V84 export is 168,614 bytes across 30 permitted files. All four patches passed CPU and source review. H200 compilation, replay, peak memory, first- and later-turn accuracy, TTFT and throughput still need measurement. [V84 implementation](serving_v0523/candidates/v84_routed_decode_committed_context/README.md) · [Iteration and frozen hashes](experiments/iterations/20261117-serving-v81-v84-decode.json)

V64 leads our completed four-load results at **1.4119×** with +1.9 GB/GPU. V68 is queued as of this snapshot. The Nov 18 UTC runner has V79 at width twelve and the V72/V76 width-eight comparison, with the latter pair gated on V68 passing. The [Nov 19 UTC runner](serving_v0523/runs/nov19_decode_runner.launch.json) is armed for **V80 versus V84 at 16 users and width eight**. It may submit while the earlier checks are pending, stops on a known shared failure, and leaves one private check available. V84 adds `--flag speculative-dflash-compact-context` to V80's seven flags. A useful gain must survive correctness, token-length, memory and TTFT checks. The 3× full-score objective remains open. [Scored result](serving_v0523/runs/abd217baf0ed49fd.json) · [Nov 18 runner](serving_v0523/runs/nov18_decode_runner.launch.json)

**Nov 17 19:39 UTC. V64 scored 1.4119× across the four official loads.** The stock-point ratios at 1, 4, 16 and 32 users were 1.170, 1.316, 1.592 and 1.621. The full run passed agreement and later-turn rescoring at +1.9 GB per GPU. V62 passed its 16-user private check at 1.164× and +1.8 GB. V68 remains queued, job `04b01f958aa5487c`. [V64 scored receipt](serving_v0523/runs/abd217baf0ed49fd.json) · [V62 check receipt](serving_v0523/runs/220305c15ded4f6b.json)

V77 fuses target expert route planning and input staging. V78 shards the draft convolution coefficient projections, saving 280 MiB per TP8 rank at the cost of ten gathers per draft pass. V80 combines both on V76's eight-row tree. V79 independently extends V76's verification attention and tree to wider blocks. At width 12 the target evaluates 50% more rows per step. Higher acceptance must repay that cost, and the released draft was trained at width eight. All four candidates are frozen and CPU/source checked. H200 correctness, memory, TTFT and throughput remain open. [Iteration and patch hashes](experiments/iterations/20261117-serving-v77-v80-decode.json)

The [Nov 18 private runner](serving_v0523/runs/nov18_decode_runner.launch.json) is armed. At 00:03 UTC it can use the spare daily check for V79 at 16 users with `--flag speculative-num-draft-tokens=12` while V68 or V76 is still pending. V79 requires three checks available and skips a known V68 failure. The matched V72 and V76 checks retain their V68-pass gate. The measured full frontier is 1.4119× toward the 3× goal.

**Latest continuation, Nov 17 18:57 UTC.** V64 passed its private
16-user check `ac89fddc4ee147ab` at an indicative stock-point ratio of
**1.412**, +1.7 GB peak GPU memory, first/later TTFT ratios .60/.72 and
later-turn agreement .97. The run returned zero short outputs. The V64
scored run `abd217baf0ed49fd`, submitted at 18:46 UTC, is **running** with
no result yet. Best completed full score remains **V44 1.1847×**. The
private-check ratio covers one load and one repetition.

V62 `220305c15ded4f6b` and V68 `04b01f958aa5487c` are still **queued**.
All three Nov17 checks were submitted by the existing runner, which has
finished. Collector PID54869 remains active. The Nov17 submission window
is exhausted, resetting at 00:00 UTC. The next private plan is gated on
V68 passing, with matched V72/V76 checks at 16 users and one check held
in reserve. V50, V53, V55 and V56 checks have now passed.

V73 adds output-channel sharding of the BF16 DFLASH context projection,
saving 560 MiB of weights per rank at TP8 before runtime cache resizing.
V74 selects two resident Hopper CTAs for compatible grouped decode
projections. V75 integrates the earlier V61 target projection sharding
with V72's grouped experts, tree and multicast workspace. V76 combines
these three deltas on exact V72. Its frozen patch is **143,800 bytes in
28 source files**, SHA-256
`199c3bd308a30addd45b798208a4f1573c28d0b5deb470df0de63b171b3a05ac`.
Component CPU checks, exact composition and independent source review
passed. V73–V76 still need H200 execution, numerical and later-turn
agreement, exact output lengths, peak memory, TTFT and throughput checks.

V76 uses the four V68 flags plus `--flag speculative-dflash-tree` and
`--flag speculative-dflash-tp-context-projection`. Humming's guarded
two-CTA policy defaults on in this candidate. The [private check runner](serving_v0523/runs/nov18_decode_runner.launch.json) is armed for Nov 18 at 00:03 UTC, subject to V68 passing. It reserves one check. The measured phase split and any 2× or 3× result remain
unestablished. Preserve the dirty V48p checkout and frozen candidates.
Read [V76's patch](serving_v0523/candidates/v76_sharded_draft_target_decode/patch.diff),
[its independent review](serving_v0523/candidates/v76_sharded_draft_target_decode/independent_review.json)
and [the V73–V76 iteration](experiments/iterations/20261117-serving-v73-v76-projection.json).
Earlier continuation entries retain the historical state at their timestamps.


**Latest continuation, Nov 16 16:33 UTC:** implemented and froze V72, the
V68 parent plus V69 activation multicast, V70 fixed-budget DFLASH selector
tree and V71 direct multicast workspace output. Exact V72 is 128,042 bytes,
SHA `e90c47ab73e66158b678893bb6fc7950c3b52f38cc3213ac6d7b7716d1b19705`.
The tree checks alternative branches within the existing eight target rows,
uses exact target-greedy verification, and commits the accepted KV/auxiliary
path with shared-cache stream fencing. A prefix fast path limits ancestry
mask work to proposal tiles. All combined CPU and independent source/rules
reviews pass; H200 JIT/replay, agreement, memory, TTFT and speed are untested.
Best full score remains **V44 1.1847x**. Its one-user check
`d40c45e5ca484591` now passed: indicative 1.084, acceptance 4, +1.9 GB peak,
first/later agreement .95/.97 and zero short outputs. This is not a score
or evidence of selector entropy. V50/V53/V55/V56 remain queued.

The Nov17 00:03 UTC (tonight Nov16 7:03pm Eastern) private-check schedule
is **unchanged: V64, V62, V68**, with runner PID84529/caffeinate84530 and
collector54869 active. Runner SHA remains
`10e91ed68e614d87bc5ba5913ca5a6b11883e00c44b6147ceb57f0fe0a55d42c`.
V72 uses the same four V68 flags plus `--flag speculative-dflash-tree` and
is an unscheduled reserve. No new GPU jobs, paid compute, scored promotion,
commit or push. Protected dirty V48p source was preserved; isolated scratch
was removed after exact-patch/Git/open-file checks.
Read [V72](serving_v0523/candidates/v72_tree_multicast_decode/README.md),
[its independent review](serving_v0523/candidates/v72_tree_multicast_decode/independent_review.md)
and [this iteration](experiments/iterations/20261116-serving-v69-v72-tree.json).
The older continuation entries below are historical and superseded.

**Latest continuation, Nov 16 14:30 UTC:** V65 combines V64+V62 exactly.
V68 adds reviewed GPU-only grouped decode (V66) and the DFLASH capacity repair
(V67). Exact V68 is 85,080 bytes, SHA `6248a58842e1a22c1cda5a957d5ab464a6ec824efcdf8308bdcc9c6a1046e505`.
All combined CPU and independent source reviews pass; no new H200 result.
Best scored result remains **V44 1.1847x**, not 2x. The Nov17 00:03 UTC
(tonight Nov16 7:03pm Eastern) private checks are now **V64, V62, V68**.
V68 replaces the unsent V60 experiment; existing queue/control jobs are intact.
Active runner `submit_nov17_integrated_checks.py`: PID84529/caffeinate84530,
SHA `10e91ed68e614d87bc5ba5913ca5a6b11883e00c44b6147ceb57f0fe0a55d42c`.
Old PID23738/caffeinate23739 are stopped. Collector54869 remains active.
V44 check1 is running; V50/V53/V55/V56 remain queued. No paid compute,
commit/push, automatic scored promotion, or tree-speculation patch.
Read [V68](serving_v0523/candidates/v68_grouped_decode_verify/README.md) and
[the iteration record](experiments/iterations/20261116-serving-v65-v68-decode.json).

**Latest continuation, Nov 16 13:40 UTC:** exact V44 scored **1.1847×**
(`c3b7bd142c264853`), correctness/later-turn pass, +1.8 GB GPU memory.
Exact V44 check32 also passed, indicative ratio1.274. Five other private
checks remain queued. New V61/V62/V63/V64 sources and independent reviews
are recorded at the end of `CODEX_HANDOFF.md`. The active Nov17 runner is
now `submit_nov17_architecture_checks.py`, PID23738/caffeinate23739,
for V64 grouped experts+SP, V62 verification attention and V60 paged TMA.
The old PID42924 runner was stopped; its historical plan below is superseded.
No paid compute, commit or push; 2× and3× remain unachieved targets.

**Latest continuation, Nov 16 04:51 UTC:** see the final section of
`CODEX_HANDOFF.md`. V44 scored and V53/V55/V56 checks submitted successfully
at the Nov16 reset and remain queued. V59 combined Humming+SP and V60
page160 FA3-TMA are frozen with independent source reviews. The Nov17
private-check runner is armed with exact source and result gates. Best
verified score remains1.1037×; the target remains2×, then3×. No paid GPU.

**Codex continuation, Nov 15 22:51 UTC.** The user resumed serving work and
set the performance goal to 2×, then 3×. The latest continuation is in
`CODEX_HANDOFF.md`; frozen candidate details are in their READMEs. Current
best score is still V48b 1.1037×. V53 indexed experts, V55 projection
sharding and the repaired V56 full prefill sequence parallelism have
private checks armed for Nov 16 00:03 UTC. V57 private startup timing is
frozen as an unscheduled reserve because dummy routing is unrepresentative.
No further paid compute is authorized. Older `serving_budget.py` phase
shares below were assumptions derived from rates, not measured profiles;
they cannot establish absolute speedup limits. The tool now labels this
and accepts explicit phase assumptions. Preserve the dirty V48p checkout.

**Update Nov 4, 23:56 UTC (Claude, after the user resumed with "get us to #1").** The public decoder is now
**2.2470×** (`c39d323f785c4e97`, private `97be48eb77234c77` 2.2398×), overall/decoder **#264**. Source is
keell0renz's Apache-2.0 decoder at `890809c`, executable AST unchanged, attribution added:
[record](experiments/iterations/20261104-decoder-v35-keell0renz-adoption.json). Overall rank tracks the decoder
board, so other boards do not move it. Top 50 needs 2.3777×, #1 needs >2.5762×. Tables below predate this.
V36 MoE A/B sweep (Nov 5): no gain worth shipping, nothing scored: [closeout](experiments/iterations/20261105-moe-ab-v36-closeout.json). `tools/moe_ab_v36.py` runs paired variants on the judge GPU inside `prepare`; its output files are for private checks only.

**Serving, Nov 13 06:55 UTC (Claude).** Working prompt: maximise the judge-reported score, budget first, one
change per scored run, predictions in `serving_v0523/runs/predictions.md` before each result.
Scored runs: V40 `c6bb173e484e4cb4` 1.051 (EP-aware Marlin routing); V41 `67e2364305c041ae` **1.093** (V40 +
free out-of-window SWA slots + `--swa-full-tokens-ratio 0.25`; ratios 1.043 / 1.065 / 1.105 / 1.165).
Budget (`tools/serving_budget.py` on V41): first-turn prefill is 27 / 44 / 56 / 63% of wall time at 1 / 4 / 16 /
32 users, so prefill throughput is the lever. Checks are starved behind the priority tiers (none has run in
18 h); the daily scored run is the only reliable full-system measurement.
V42 (`serving_v0523/candidates/v42_dense_prefill/patch.diff`, 20 KB, 4 files) adds a dense BF16 expert path
for large prefill batches: new kernel `kernels/jit/csrc/gemm/marlin_moe/nvfp4_marlin_dequant.cuh` (bit-exact
against checkpoint weights) and an eager break in `muse_spark_fp4_moe.py`. Modal evidence is under
`experiments/modal_serving/`; runner `tools/modal_serving_ab.py` (live-spend guard, `--gpu H100` when no H200),
worker `tools/serving_moe_ab_worker.py`. V42 is not yet run officially: checks `ec7cad544a0843ef` (16 users)
and `a0965d83c7b541ec` (32 users) are queued; next scored run opens 00:00 UTC Nov 14.
Ruled out with measurements: Marlin tile/occupancy variants (within 1%), smaller dequant pieces (fixed launch
cost dominates), 32K prefill chunks (activation memory scales with chunk size).
Update 07:10 UTC: on an H200 the dense path is only 1.13x at 16K tokens and 0.96x at 12K (per-expert fixed
costs dominate), so V42's gain is about 1.8%, not 3%. **V43**
(`serving_v0523/candidates/v43_dense_prefill_profile/patch.diff`, sha256 b2ecd244…, 31 KB, 6 files) is V42 with
the threshold at 448 routes per expert plus a startup profiler (`layers/muse_spark/prefill_profile.py`, laps in
`models/muse_spark.py`): the eager prefill warmups at 16K tokens are timed per stage, Marlin forced on the first,
and the lines `Prefill stage profile` / `Prefill expert check` land in the startup log. It is debug code: withdraw
that run (`okc withdraw`) or supersede it with a clean patch before the track ends.
`serving_v0523/runs/submit_v43_scored.sh` submits V43 after 00:00 UTC Nov 14 unless a V42 check has failed.

**Serving, Nov 15 05:10 UTC (Claude): V48b passed at 1.1037, the best judged build.** Job `8ebed2bced8942ff`
(V41 + all-reduce cap, sha256 e59d8efc...): 1.063 / 1.072 / 1.108 / 1.175 against V41's 1.043 / 1.065 / 1.105 / 1.165;
predicted 1.105. Queued 156 minutes, ran 50; GPU memory +0 GB; `symmetric_memory = 130.01 MB, multicast = True`.
First-turn TTFT p90 moved only -0.1% to -0.25%, so at 128 MiB the custom sum is about a tie with NCCL; the 1-user
gain coincides with acceptance 3.9 -> 4.1 (run-to-run variation). Real effect about +0.3% to +0.8%, near the noise
between runs. Dropped: moving the 256 MiB attention sum to the custom path (worth about 0.2% on the score). The
judge's `later_turns` block shows it starts a stock server (`startup_s` 161) and rescores our tokens there, and the
"outputs agree" numbers come from 176 reference turns. `driver_nov16.sh` is still running and will submit V48
(V44 + cap) at 00:00:30 UTC Nov 16; predicted about 1.14 if the dense path runs, about 40% chance of no score.
Read while looking for a cause of V44's timeout, none found: the breakable graph captures the transformer body
only and runs the logits tail eagerly, so requests with logprobs still replay the graph; Muse sets
`SGLANG_ENABLE_CUDA_GRAPH_DEDUP`, and the 209-second stall is probably the first unbroken graph being registered
after five broken ones (a fixed startup cost, not a hang). The node load generator (`tools/okc_node_loadgen.py`)
never sends `logprobs`, so that request shape is untested with the dense path. Untested idea if V48 also times out:
`--cuda-graph-max-bs-prefill 13824` keeps graphs for the Marlin sizes and runs the dense sizes eagerly, which takes
the graph break out of the picture; how `max_bs` trims the Muse shape list was not verified.

**Serving, Nov 15 01:50 UTC (Claude): V44 timed out at the judge; V48b is the Nov 15 run; a driver handles Nov 16.**
V44 (`8ff48fb6618b4188`) queued 13 h 20 min, then ran exactly 120.0 minutes and was stopped ("the serving run hit its
time limit"): no rungs, no server log, `refunded: true`. Cause unknown. It starts and serves 1 and 4 users on the
rented node with the same image, but had never run at 16 or 32 users or under the judge's correctness requests, and its
prefill graph capture has an unexplained 209-second stall on the node at the first shape under the dense threshold
(13,824 tokens; node log `experiments/modal_node/20261113T213641Z-bench/results__partial__v44__server.log`), which V41
does not have. The judge sandbox runs as uid 65534 with a read-only source tree and no network (`result.sandbox` in a
passed run's JSON); JIT compiles go to `~/.cache/sglang/jit` or `SGLANG_JIT_CACHE_DIR` (`kernels/jit/utils/compile/`),
and V44 is the first build of ours that needs a new JIT kernel compiled there. **A refunded slot was lost:** the
refund came at 18:27 UTC Nov 14 but the waiter's notification reached the session only at 01:41 UTC Nov 15, after the
reset. Do not rely on being re-invoked for anything time-critical. The gate script also did not know the judge's
status `failed` and waited 100 minutes, so **V48b** (V41 + cap, sha256 e59d8efc...) went in at 01:41:42 UTC Nov 15 as
job `8ebed2bced8942ff` (prediction 1.105). `serving_v0523/runs/driver_nov16.sh` is running in the background (log
`runs/driver_nov16.log`, `DRY=1` prints its decision): it resubmits V48b once if the judge refunds it the same UTC
day, submits **V48** (V44 + cap, sha256 5ef6d32f...) at 00:00:30 UTC Nov 16 as the second and last blind attempt at
the dense path (V44 alone if V48b shows the cap is bad), saves each finished run's JSON and log, and exits when both
are finished or at 23:30 UTC Nov 16. If the dense path gives no score again, drop it and spend the remaining runs
(Nov 17, 18, 19 UTC) on launch flags over the best passing build: `--chunked-prefill-size 8192` (the organisers' own
example in `serving_v0523/flags`; with the cap raised every prefill sum then fits the custom path, but the dense path
would no longer trigger) and `--enable-nccl-nvls`. Judge jobs record `duplicate_of`, so an identical resubmission may
be deduplicated. Checks have not run for two days (three are queued).

**Serving, Nov 14 03:10 UTC (Claude).** The gate was stopped and **V44 was submitted ungated** as the Nov 14 scored
run on the user's go-ahead: job `8ff48fb6618b4188`, flag `swa-full-tokens-ratio=0.25`; prediction in
`serving_v0523/runs/predictions.md` (point 1.13, range 1.118-1.166, about 10% chance of not passing). Facts found
while waiting: (1) the kernel track has closed, so queue priority (top 100 overall) can no longer be earned; we are
rank 281 on the decoder at 2.247x and checks will keep starving, so scored runs and the node are the only
measurements. (2) The judge also bounds later turns (`result.later_turns.bounds`: agreement >= 0.92, first-turn
agreement >= 0.85, NLL rise <= 0.07; V41 had 0.97 / 0.95), tighter than `result.correctness`. (3) Launch flags not
on the fixed list in `serving_v0523/flags` may be changed, including `--enable-symm-mem`, `--enable-layernorm-sp`,
`--chunked-prefill-size` and `--speculative-num-draft-tokens`; none of the first three has been measured. (4) A different DFlash block length is not worth a run. The GPU time per decode step left after prefill
(`tools/serving_budget.py` residual on V41) is 17.3 / 28.2 / 53.3 / 62.3 ms at 8 / 32 / 128 / 256 verify tokens, so
the marginal cost of a verify token falls from 0.45 ms to 0.07 ms as load rises. With acceptance p = 0.78 per
position, a shorter block loses at 1 user and gains under 1% at 16 and 32; a longer one loses everywhere. Block 8 is
near the optimum at every load. (5) `tools/okc_compare_runs.py base.json new.json --predict a,b,c,d` prints the
per-load comparison of two judged runs.

**Serving, Nov 12 (Claude).** The base moved to SGLang `297e54fd` (v0.5.23); okc is now 0.8.0. The new checkout is
`serving_v0523/` (its `sglang/` is git-ignored). The track opens 17:00 UTC Nov 12 and closes 08:00 UTC Nov 20.
Budget: 3 checks (one load, ~25 min) and 1 scored run per day, reset 00:00 UTC. We are outside the priority group
(#273), so our runs queue behind it. New rules: patches must be lossless; flags not marked fixed may change; at most
+2 GB GPU memory. The old V26/V27 work targets the a55ab7d layout, much of what it fixed is already in stock, and
it is not being rebased. First candidate, V40 (`serving_v0523/candidates/v40_ep_local_align/patch.diff`, 3.5 KB):
under EP, align Marlin routes over the 32 local experts (inactive -1 prefix skipped, sorted path, single-token
aligner) and size Marlin blocks from routes per expert using the global expert count. Not yet run on a GPU.
`serving_v0523/runs/run_v40_check.sh` submits it at opening. Leads not pursued: a cached Marlin workspace
(upstream keeps it per call to avoid capture deadlocks) and skipping the greedy spec TP broadcast (small gain,
desync risk). Open leads: `--swa-full-tokens-ratio` below 0.8 (read pool sizes from the check's startup log
first; Muse keeps out-of-window SWA slots), an SM90 finalize+all-reduce fusion, and graph-capturing DFlash
draft-KV materialization.

Prepared November 4, 2026, after official V33 completed. Repository:
<https://github.com/CristiSavca/kernel_challenge>. Original workspace:
`/Users/cristisavca/Documents/HASIDIC_PROJECTS/open-kernel-challenge`.

## Current state: paused, no GPU jobs running

The user explicitly paused the optimization goal, then asked for this handoff
and GitHub push. Both the goal and Codex automation
`okc-research-and-harsh-critic` are **PAUSED**. All optimization agents are stopped. No
official or Modal GPU job is in flight. Do not interpret the older automation
prompt, historical plans, or this handoff as permission to resume experiments.

The eventual objective is #1 and a competitive serving entry. It has **not**
been achieved. The requested 100% score gain was a stretch target, not a measured
result or defensible prediction. Recent private decoder experiments regressed;
none replaced the public incumbent.

The last saved board snapshot reports CristiSavca **#409 overall and decoder**,
with decoder 2.1812× against a 2.5762× leader. It is a dated snapshot, not a live
rank guarantee: [board evidence](experiments/leaderboards/20261104T2300-v32-closed.json).

## Read these first

1. [AGENTS.md](AGENTS.md): project constraints and existing authorization.
2. This handoff, then [tools/RESEARCH_LOOP.md](tools/RESEARCH_LOOP.md).
3. [tools/MODAL.md](tools/MODAL.md) before any resource use.
4. The relevant problem's `AGENTS.md`, `TASK.md`, and actual source.
5. The exact receipt/profile for the hypothesis being considered. Broad older
   research notes are not substitutes for current measurements.

## Passing public incumbents — preserve these

These source files match the recorded scored receipts. Full hashes are in
[incumbents.json](handoff/20261104/incumbents.json).

| Problem | Public speedup | Scored job | Source SHA prefix |
|---|---:|---|---|
| Dense projections | 1.4868267835× | `3caa02bd72da4c49` | `27a0afe97d0a9b8c` |
| Attention | 2.6119701145× | `8f9e5971f95841a0` | `87a9bddc1b013ff3` |
| Routed experts | 2.3736910444× | `d24dd50196cb4158` | `247888acca10f50d` |
| Whole decoder | 2.1811831350× | `7a8c4941741645ee` | `97af39947d4cab45` |

The decoder also has an exact-source private anchor `b2a773e2d7194de8`,
2.1766544507×. Evaluate a new proposal against both retained decoder anchors.
Canonical receipts are `experiments/runs/<job>.json`; frozen submitted sources
are under `experiments/candidates/<problem>/`; raw profile JSON/text/logs sit in
that directory's `profiles/`. A local compile or benchmark is not an official
pass. Do not promote a source with unverified flags or incomplete case coverage.

## Latest iteration and the approaches already rejected

V33 is **closed**, with no scored submission or public source change.

| Iteration | Official private job | Score | Decision / observed reason |
|---|---|---:|---|
| V28 decoder private partials | `2976c25cc3dc41b1` | 2.1146266798× | Correct, slower than incumbent; no retry/port |
| V29 exact BF16 repair | `185fee4d182f41dd` | 1.8985960692× | Correct, no compressed timed activation |
| V30 split MoE architecture | `d0510bc0412a4537` | 2.0335370900× | Standalone MoE; slower than its 2.3737× incumbent; no decoder port |
| V31 eligibility diagnostic | `5f48e0da4dcd4595` | 1.9115472025× | Selftests pass; tuner rejects compressed choices; array-sized local allocation |
| V32 scalar repair | `6a938fff9d5d417c` | 1.9080803869× | Local allocation removed, still no compressed timed activation |
| V33 missing-nibble sidecar | `60b61041d08a4880` | 1.9740118155× | Correct, partial activation, all affected performance caps fail |

V33 exact source is
`a8a619eaa0c531aa115d39fd2db8718d7d52828feac8e6fdea005c4e61a3c090`.
It retained a 768-byte primary per 512 BF16 weights and supplied missing high
nibbles from a 256-byte sidecar for escaped units. This removed per-word repairs
and the header dependency before primary loads. CPU packing tests passed; the
official check passed 13 tests and 12 timed cases with clean flags.

Nevertheless, V33's whole-score ratios were **0.905019 / 0.906902** against the
public/private incumbent; affected four-load speed ratios were
**0.864490 / 0.866609**. All eight affected cases exceeded the fixed latency cap.
Controls passed. Only some compressed kernels activated. All 22 observed
compiled pointers had zero local allocation, including 18 compressed pointers.
Those compiler observations do not establish a performance gain or sole cause.

Read [V33 closeout](experiments/iterations/20261104-decoder-nibble-sidecar-v33-closeout.json),
[raw analysis](experiments/iterations/20261104-decoder-nibble-sidecar-v33-root-result.json),
and [independent critique](tools/DECODER_NIBBLE_SIDECAR_V33_RESULT_CRITIC_20261104.md).
**Close the V29–V33 consumer repair/completion line.** No identical retry,
parameter rescue, forced selection, relaxed threshold, or post-result pooler
subset splice. A materially different hypothesis needs its own evidence.

Other closed work includes shared-A, SC6, R16, escape directories, prefetch,
wait-distance and register sweeps. Consult the current research loop and the
specific records before proposing something that sounds new. The unique-expert
reuse audit found only a conditional ~1.465% whole-decoder scenario, not a
demonstrated large lever.

## Proposed next public-kernel experiment: V34, not implemented

The distinct evidence gap is the **original incumbent's dense-kernel hardware
counters**. Existing retained NCU captures are standalone MoE; V31–V33 compiler
attributes and tuner times do not identify the original dense bottleneck.

Proposed scope, only after the user resumes and resource/worker review passes:

- Load exact decoder97af unchanged. Use `bfp_prepare` with all five dense weight
  matrices; no MoE/cache allocations are needed for this component diagnostic.
- Inspect exactly QKV M16 via `bfp__run_sk2` and fused pooler/router M16 via
  `bfp_fused_pooler_router`, at natural `bfp_SK2_PDL=0`.
- Pin source, image, generated CUDA/CPP/build flags, weights, templates and
  PK/FUSE identities. Record actual escape counts and patch-table sizes.
- Independently check outputs and collect ordinary cold graph observations
  separately from NCU replay. This is standalone dense evidence, **not** UV,
  PDL, full-forward timing, or an official speedup.
- Declare one metric set covering DRAM/L2, issue/eligible warps, shared traffic
  and conflicts, synchronization/scoreboards and pipeline pressure. Query actual
  availability; missing required metrics or denied counters means stop
  inconclusive. No metric fishing or allocation retries.
- Aggregate counters can distinguish broad pressure classes. They cannot
  identify escape instruction PCs or be summed into recoverable-time percentages.

**No V34 worker, test, manifest, or ready-to-run payload exists.** The controller
draft was started, then archived at the pause. Proposed worker contract:
`tools/decoder_dense_hardware_v34.py`, APIs `build_payload(root,image,tag)`,
`validate_payload`, `run`, `validate_result`; files `decoder.py`,
`hardware_metrics.py`, `dense_v34.py`; 900-second sandbox, 850-second worker
budget; atomic result <=1 MiB. The metric list, implementation, identity checks,
full-symbol parser, oracle and tests remain unfinished.

The archived controller patch references this missing worker. **Do not apply
the patch and assume it is executable.** The existing controller was restored
to its last committed version after the exact draft was preserved.

## Modal money: current guard still blocks allocation

Authorization is **free-credit experiments only**, not out-of-pocket GPU/API
charges. One H200 at a time, <=900 seconds, no automatic allocation retries,
guaranteed sandbox termination. Use the guarded runners. Adding a payment
method did not authorize paid usage. Official challenge checks remain a separate
resource, subject to current rules/quotas/deadlines.

The $30 grant, $5 headroom, historical provider spend floor **$4.83274092**,
$33.88 original reservations and already-reviewed $3.74/$9.51 adjustments leave
the old-policy conservative allowance **−$0.46274092**. This is ledger
accounting, not a statement that the provider's entire grant was consumed.
Never delete/reset `experiments/modal/.allocation.lock` or the reservation
journal, reduce the spend high-water, or count corrections twice.

A new read-only reconciliation found **$0.83** additional posted-cost overlap
and **$5.48** additional conservative estimate reductions. Conditional allowance
would be **$5.84725908** after reviewed activation and fresh preflight. These are
not settled refunds or currently spendable authorization. CPU runs and the V27
timeout retain full reservations. The lifetime policy has an explicit teardown
and delayed-billing uncertainty; maintain the existing margins and headroom.

The [proposal](tools/MODAL_ADDITIONAL_RECONCILIATION_PROPOSAL_20261104.md) and
[independent proposal review](tools/MODAL_ADDITIONAL_RECONCILIATION_CRITIC_20261104.md)
are complete. **Implementation was interrupted, untested and not reviewed.** Its
activation pin was `None`. The planned test file does not exist. The exact draft
guard and certificate are archived; the active guard is restored to
`9f70c5006d8afd6e4a077cec8f3fe3baada2620703479bebf87ea0a333f0c9b2`.

Before any future activation, finish the fixed additive policy, independently
review it, and test tampering, overlap, repeated corrections, missing/downward
billing rows, excess costs, resource changes and unrelated future reservations.
Preserve old certificates and the complete original journal. Requery billing,
all app activity and available grant through the guard immediately before use.

## Draft preservation and the nested serving checkout

[handoff/20261104/manifest.json](handoff/20261104/manifest.json) records exact
base/draft/patch hashes. It preserves:

- `modal-credit-additional.DRAFT.patch` and its draft certificate. These are
  incomplete, disabled accounting changes; no implementation approval.
- `decoder-dense-v34-controller.DRAFT.patch`. Only controller scaffolding;
  its worker is absent. Eleven existing V27 controller/worker tests passed
  during development, which does not validate V34.
- `serving-working-tree.PRESERVED.patch`, the pre-existing two-file nested
  checkout diff at SGLang `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`.

Both draft patches were checked as reversible and applicable to their recorded
base. Active tool files were restored only after exact patch preservation. No
user edits or unique nested commits were discarded. The nested serving working
tree itself was left untouched. Its full checkout is intentionally ignored by
the parent repo; on another machine, recreate the pinned SGLang checkout before
applying its preserved patch. That dirty patch is V22-related work, **not** the
isolated V27 serving entry and not a validated deployment.

## Serving work worth retaining

The prize path is the separate serving track. Prototype **V26** passed official
private job `274c69cbc0954d76`, five tests/six benchmarks, score **1.6851100701×**
its same-run stock component. It is slower than the public native MoE kernel;
its value is a serving integration candidate, not a public-kernel improvement.
No full-service throughput, rank, TTFT or prize claim is established.

**V27** is the isolated pre-Marlin installer/forward adapter at
`serving/candidates/native_serving_v27/`: native SHA `7310dd76…`, patch
`c0ca4e41…`, 213,160 bytes across four allowed files. CPU reviews passed. Its
bounded Modal run `20261104T213729Z-f6cf2030` installed the rank-3 native bank,
then timed out within a coarse stock-preparation/initial-forward interval.
It completed **0/38 numerical cases and no timings**. The sandbox terminated.
This is incomplete validation, not a numerical pass/fail diagnosis.

The distinct next serving probe would isolate the stock interval with stage
checkpoints, not rerun the same full suite. See
[diagnosis](tools/SERVING_NATIVE_MOE_V27_STOCK_PREP_DIAGNOSIS_20261104.md).
A separate Python3.13 `ast.dump` empty-field fingerprint mismatch was reproduced;
`show_empty=True` resolves it, but it did not cause the timeout. Preserve the
frozen failed worker and repair only a new diagnostic version.

## Rules and deadlines last verified November 4

Recheck [live rules](https://openkernelchallenge.dev/rules) before resuming:

- Kernel cutoff: **November 5, 2026, 08:00 UTC**. Stop kernel submissions then.
- Serving opens **November 10, 17:00 UTC**, closes **November 20, 08:00 UTC**.
- Prizes are for serving; kernel top50 earns queue priority. Serving board is
  private, so #1 may be unobservable before it ends.
- Live kernel quotas were40 private checks/20 scored runs per problem/day,
  resetting00:00UTC; `okc problems --json` is authoritative.
- Serving uses8H200, 3 checks/1 scored submission daily. Preserve serving-specific
  BF16, concurrency, output and TTFT requirements; kernel permissions differ.
- Serving patch: only source/JSON under `python/sglang`, no model_loader changes,
  binaries or renames, <=409600 bytes, no added line over400 characters.
- No timing/harness tricks, cached input-derived outputs, skipped experts/keys,
  lower-than-allowed precision or seed-specific policy. Weight-only lossless
  packing is allowed. Official evaluators stay unchanged. A human accepts terms.

## Environment, provenance and resumption

Original host: Apple Silicon Mac, no CUDA GPU. Installed tooling was `okc`0.7.1;
`.venv-modal` used Python3.13.4, Modal1.6.3, NumPy2.5.4. The venv, credentials,
keychain and nested SGLang checkout are not in Git. Authenticate on the target
machine through normal human-owned flows; never commit tokens. CUDA compilation
and numerical/performance validation require the actual H200 environment.

Some immutable receipts embed the original Mac's absolute artifact paths.
Strict readers such as `analyze_decoder_v28.py:read_file` reject paths outside
their current repository root. On a differently located clone, read the retained
JSON directly or implement an explicit, reviewed old-prefix-to-new-root mapping
in a new reader version. Do not rewrite frozen receipts, weaken path checks, or
silently invalidate their recorded hashes to make a historical analysis run.

The repository includes retained diagnostic binaries/SASS as evidence. They are
not permitted submission payloads: challenge GPU code must be built from readable
source during the run. Preserve third-party LICENSE/NOTICE files and attribution.

Research intake already reviewed EAT, ParallelKernelBench, Together Compiler,
X/arXiv/GitHub/Hugging Face/DeepSeek/CUTLASS leads. Start with their linked notes
in `tools/RESEARCH_LOOP.md`; do not repeat a broad scan without a specific new
lead. No verified stronger public whole-decoder drop-in was found in the scoped
scan. The Lean work under `formal/` proves limited source-bound integer/packing
properties, not whole CUDA numerical correctness, races or performance.

When the user explicitly resumes: first check the deadline, jobs/processes,
live quotas/boards and source hashes. Finish the disabled credit-policy review
only if needed for a concrete bounded experiment. Complete/review V34 before
allocating; or follow a newly justified distinct hypothesis. Keep one submitter
and separate implementer/critic roles, freeze sources/gates before measuring,
resolve material findings, and score only an exact officially passing useful
candidate. Record raw results, costs and keep/discard decision. Do not resume
the Codex recurring monitor unless the user asks to resume that work.
