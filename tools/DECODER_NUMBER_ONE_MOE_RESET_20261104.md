# V30: complete-K tiles with separate sparse-job geometry

One bounded architectural experiment is selected from public MoE
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
It replaces MAIN Stream-K coordination with complete-K output tiles. It does
not transplant the serving arithmetic or revisit V28 private partials.

The timed routing/gather kernel preserves expert/route order and FT64 jobs,
then partitions actual job counts into `c<=8` and `c>8`. Six ordered launches
perform routing/gather, sparse A, dense A, sparse B, dense B, and the unchanged
public final norm. Both bins execute for arbitrary routing; an empty bin does
no matrix work. No CPU reads input-derived counts. M<=16 and non-PRMT scales
retain their original implementations.

Sparse jobs use 128 output rows, K256, fixed n8, two consumer warpgroups and
two producer warps: 320 threads. Three joined rings require
`22272 + 3*(18432+4096) = 89856` dynamic shared bytes. Actual compiled occupancy
must admit two CTAs/SM. Dense jobs retain 256 output rows, adaptive n16..64,
576 threads and 231168 bytes, with at least one admitted CTA/SM. Every tile
reads all sixteen K stages, then writes its complete epilogue once. This
removes inter-CTA partial reductions, fixup buffers, readiness polling, owner
searches and the global spin barrier. Register allocation remains unmeasured.

The current standalone hardware captures contain 16/16 n8 jobs at M32 and
16/27 n8 jobs at M256. Therefore the first has 1024 sparse A tiles and 512
sparse B tiles; the second has those same sparse tile counts plus 352 dense A
and 176 dense B tiles. Against 132 SMs, sparse two-CTA capacity gives at least
four/two tile waves; dense one-CTA capacity gives three/two. These are counts,
not a timing model or decoder routing distribution. Coarser whole-K tails,
five additional launches, whole-phase synchronization, and twice as many
sparse TMA events/activation copies per 256 output rows are real adverse costs.
Packed weight bytes remain 27 MiB per job.

Current KPERM captures read about 1.007–1.009 times nominal weight traffic at
approximately 77% DRAM throughput. Thus this experiment targets coordination,
resource fit and useful work per CTA, not a fictional redundant weight bank.
The active decode stream already folds BF16 scale broadcasts. Its math-stall
counter does not identify a free algebraic replacement.

All code/scale packing, KPERM mapping, dequant expressions, GS2/wait1 groups,
NTT1 two-accumulator policy, FP32 epilogues, BF16 activation cast and final
norm/weighted sum remain from the public parent. Removing split-K changes
FP32 association; official numerical checks remain mandatory. Zero local jobs
must still write zero output via a newly produced complete route mask.

An illustrative profile-share scenario applies only to three MAIN loads:
10% less MAIN MoE time would reduce whole-decoder normalized time by roughly
2.4%; 20% gives roughly 5%. Profiles are not critical-path bounds. This alone
cannot plausibly deliver the user's requested 2x score; it is complementary
to the distinct dense V29 work, and no gain is promised.

Implementation stays under `muse_moe_w4a16/candidates/moe_arch_v30/` with
reproducible extraction from the pinned parent, unchanged fallback source,
source-bound address/coverage checks and actual resource markers. Root alone
owns one official private 5-test/6-benchmark/6-profile check after review.
No test-name, seed or routing-outcome special case is permitted. A failed
capacity/correctness gate closes this fixed implementation; measured timing
must preserve every affected case, especially M32, before any decoder port.

Root fixed the screen before measurement: compare against **both** exact
247888 public `d24dd50196cb4158` and private `bcb761dfca264db0`. Require clean
5 tests/6 benchmarks/6 profiles and empty flags; cold occupancy at least two,
hot at least one; all six new launch symbols on M32/128/256. The three affected
loads need speed GM at least 1.04, the whole score at least 1.02, and each
affected normalized latency ratio at most 1.03. The three unchanged controls
need normalized-latency GM in [0.97,1.03] and each ratio at most 1.03. These are
consistency screens, not statistical significance claims. No failed-case
shape rescue, threshold relaxation or identical retry is authorized.

Primary research context: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
uses grouped/masked GPU work scheduling, but its published MegaMoE FP8/FP4
path is not a BF16/NVFP4 Hopper replacement. No DeepGEMM code is imported.
[NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
and [PTX WGMMA semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)
support the resource/synchronization constraints, not this candidate's speed.
The [Nov4 Zimo public source report](https://github.com/zimo0110/muse-spark-open-kernel-challenge/blob/fe1c70bb3a1348096ebe6c9f6d93b960fccfad09/README.md)
reports similar parent-family performance and negative fixed-N/private-partial
experiments; those are not evidence for this new whole-K scheme. Parent
Apache-2.0/MIT notices remain in the isolated source.
