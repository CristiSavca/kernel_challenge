# V215 reduces padded expert work during decode

V215 combines V213’s grouped expert decode kernels with V214’s draft-cache projection on the real mixed prompt and verification scheduler. Each expert tile uses the smallest supported matrix width that covers its live rows. The cooperating warpgroups exchange partial sums once and retain the original FP32 addition order. A separate change retains one immutable 20 MiB peer weight bank per GPU and projects fresh target features directly to keys and values. Rejected speculative positions keep their original physical cache contents. [Official patch](patch.diff) · [Expert kernel](../v213_live_decode_expert_reduction/README.md) · [Draft projection](../v214_resident_peer_draft_projection/README.md)

## Estimated work

| Evidence | V211/V212 path | V215 path |
|---|---:|---:|
| Issued expert matrix rows on an archived routing histogram with 256 verification tokens | 792 | 456 |
| Intermediate expert reduction traffic per layer and rank on that histogram | 123.75 MiB | 57.19 MiB |
| Nominal packed expert weights and scales requested on that histogram | 891 MiB | 891 MiB |
| Eligible mixed projections among 99 possible batches in the archived 16-user schedule | 0 | 58 |

The first three rows apply the respective kernels to the same measured rank-local routing distribution. With 32 verification tokens the matrix row count stayed at 128, while partial-sum traffic fell from 20 to 7.5 MiB. The final row comes from a chronological replay that starts each repetition with a cold bank. The replay models traffic and eligibility. Neither trace measures V215 latency. [Routing histogram and arithmetic](../v213_live_decode_expert_reduction/physical.json) · [Chronological projection model](../v214_resident_peer_draft_projection/geometry.json)

## Verification and submission

All six expert variants compiled for Hopper SM90a with 96 registers and zero spills. The original reduction tree matched across 1,708,032 FP32 words in the host fixture. In the connected replay, 2,048 finalized BF16 expert words fed the actual mixed scheduler. Eighty-eight paired comparisons across real four- and eight-rank Gloo checked token IDs, acceptance lengths and the complete physical draft cache, including rejected slots. CPU operations modeled the expert GEMMs, transformer body and CUDA communication boundaries. [Compilation](../v213_live_decode_expert_reduction/sm90a.json) · [FP32 reduction](../v213_live_decode_expert_reduction/partials.json) · [Integrated replay](joined.json)

The 322,388-byte official export applies and reverses on the public base. Both donor orders reconstruct the same tree. The six launch flags parsed in all 720 orders and passed the published static restrictions. The runtime checks use general tensor shape, ownership and weight identity. The new peer bank holds immutable model weights across requests. H200 token agreement, per-GPU peak memory, time to first token at every load and end-to-end throughput still require a judge run. V64 holds the measured four-load score of **1.4119×**. [Validation](validation.json) · [Flag parser](flags.json) · [Serving rules](https://openkernelchallenge.dev/api/v1/serving) · [V64 result](../../runs/abd217baf0ed49fd.json)

The existing 16-user private standby can select V215 between 04:00 and 08:00 UTC on November 20, after the two preceding allocations and while V120b/32 remains pending. Selection requires a passing V204/16 result that beats V68/16 by more than 1.63% on both raw tokens per second per GPU and throughput at stock interactivity. The margin comes from the observed three-repetition spread of V64/16. Insufficient V204 evidence retains V198 in that slot. At activation all 31 existing receipt files retained their digests, and all eight other active November 20 workers retained their locks. At the latest observation V93/16, V204/16, V120b/32 and V139/4 were queued. [Private standby](../../runs/nov20_v215_runner.launch.json) · [Selector regression tests](../../runs/history/nov20_v215_staging/nov20_v215.cpu.json) · [Independent tests](independent_standby.md) · [Judge observation](judge_observation.json)
