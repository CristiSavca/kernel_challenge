# Research reset, November 4, 2026

The verified 07:15 UTC decoder standing is CristiSavca #398, 2.1502x, with
gauravjain14 first at 2.5401x. Equaling that score requires about 15.35% less
normalized latency. The goal is still first place; passing checks, reading
papers, and small component gains do not establish success.

## What recent primary research changes

| Source | Actual result / mechanism | Application here |
| --- | --- | --- |
| [GC-MoE, submitted Nov 3](https://arxiv.org/abs/2611.04222) | Separates communication and compute resource budgets with CUDA Green Contexts, token-ring handshakes and concurrent graph roots; evaluated on eight H20/B300 GPUs. | The kernel track omits communication. Its 1.188x H20 mean is not a single-H200 gain forecast. Preserve this lead for the eight-GPU serving track. |
| [Understanding GPU Megakernels, Oct 31](https://arxiv.org/abs/2611.00495) | Reconstructs two megakernels on H100; properly ordered PDL split kernels improve TK by 5.7% and match VDCores. Resource admission and bandwidth saturation determine useful overlap. | Existing decoder already uses PDL. More fusion alone is unsupported; any rewrite must show ready work, capacity in the limiting resource, and correct release/acquire ordering. |
| [One Megakernel Is Not Enough, Oct 26](https://arxiv.org/abs/2610.35336) | Tile/event models select multiple megakernels or conventional execution across 13 tested model/batch combinations; none selects one monolith. | Use measured operator and dependency costs to select boundaries. The TTS latency gain is not a Muse decoder forecast. |
| [VC-Attention, Sep 14](https://arxiv.org/abs/2609.15810) | Approximate direct score-to-FP8 probability conversion plus video-specific value smoothing. | Fixed KV rules exclude the smoothing. An exact-rounding adaptation loses the cheap conversion advantage; CPU boundary analysis and rejection are recorded separately. No GPU variant is justified. |
| [ForgeMegakernel, Sep 11](https://arxiv.org/abs/2609.12379) | Per-SM instruction streams, dependency counters, reusable shared storage, and an independent intermediate-state oracle; H100 experiments report 1.21x geometric mean over SGLang. | Adopt intermediate checks and byte-accounted dependency analysis. Its conventional full-model graphs have hundreds of launches; our measured layer has 7–10, so the headline gain does not transfer directly. |
| [LiquidGEMM, Sep 1, 2025](https://arxiv.org/abs/2509.01229) | Mixed-precision load/dequant/compute scheduling, evaluated with a new quantization method. Separating dequantization into a producer introduces shared-memory transfers and synchronization overhead. | This older paper is a counterexample to a proposed separate-dequant rewrite. Our current register dequantization already follows the closer applicable pattern. Its new quantization is not a lossless NVFP4 substitution. |

Recent engineering sources also matter, but are not arXiv papers:
[Humming](https://vllm.ai/blog/2026-10-21-humming) and
[Chord's design](https://github.com/novitalabs/chord/blob/main/docs/design.md)
describe compact small-token schedules and multiple resident CTAs on H200.
Their formats and workload distributions differ. Simply selecting mma.sync is
not new: our small-token paths already use it. The concrete new direction is a
smaller WGS resource unit with enough shared-memory and register capacity for
two resident CTAs, while preserving exact scale values, BF16 activations and
FP32 reduction. This remains a hypothesis pending proofs, compilation and GPU
measurement; no advertised library gain is assumed.

## Portfolio and immediate experiments

The independent critic reproduced the complete decoder profile-share scenarios
in `experiments/iterations/20261104-decoder-research-roi-scenarios.json`.
Hypothetically reducing only the two large full-Q8 attention rows by 20% cuts
aggregate normalized latency by 0.74%. Reducing all MoE rows by 20% and all
dense rows by 25% gives a 17.84% scenario. These are neither bounds nor forecasts:
PDL rows include waits, can overlap, and differ from benchmark captures. They
explain why research now targets broad MoE and dense improvements together.

KPERM3's first official private result passed 5 tests and 6 timed checks with
no flags. The score is 2.389836x versus the parent's 2.356104x public score;
reported aggregate intervals overlap. All three changed sizes improve their
SGLang-relative ratios, with a ~3.3% geometric mean against the public parent,
but seeds, nodes and baseline times differ. The independent critic permits one
isolated decoder experiment, not an immediate public promotion.

Decoder candidate `97af3994` contains precisely that checked MoE mechanism.
Root and critic independently recovered 50,331,648 code nibbles and 3,145,728
scale bytes from actual decoder packing helpers. Other decoder AST nodes match
the incumbent except the new packing helpers, MoE source and namespace.
Private job `b2a773e2d7194de8` completed successfully: all 13 tests and 12
benchmarks passed without flags. Its score is 2.176654x, with a 95% interval of
2.144005–2.209304x, versus the public parent's 2.150186x. The changed loads
improve by a 1.71% geometric mean against the public parent; unchanged loads
also improve by 0.75%. This supports retaining a modest candidate, not claiming
a confirmed 1.23% whole-decoder gain. The critic recommends holding promotion
while the broader dense and compact-MoE experiments finish. Do not repeat the
same check merely to seek a better sample. Public submissions remain intact.

The compact-WGS rewrite is implemented separately from KPERM: 128-row/K128
tiles, two compute warpgroups and two producer warps per CTA, with a target of
two resident CTAs per SM. Static shared storage falls from 231168 to 98048
bytes per CTA. Actual compiled occupancy is a mandatory runtime gate. Root,
author and critic have passed actual-source CPU packing and schedule proofs;
an additional independent review checked synchronization and the changed FP32
normalization. No GPU performance claim follows from these checks. The design
also doubles activation requests and halves weight lookahead per SM, so its
resource change has a real cost. See `MOE_COMPACT_RESIDENCY_PROPOSAL.md`.

Dense candidate `b0387be3` preserves all correction warps and FP32 accumulation
order while allowing exact compressed-weight corrections to read their input
directly instead of waiting on the main TMA ring. Independent checks covered
19,673,088 addresses, 2,561,346 patch ownership/order decisions, and 448 ring
interleavings. Official private job `d4c26ebb65e94a84` is queued as of 07:33 UTC.
Resume that job, never duplicate it while pending. The new global reads can
also cost more than the removed waits; see `DENSE_ESCAPE_DECOUPLING_PLAN.md`.
