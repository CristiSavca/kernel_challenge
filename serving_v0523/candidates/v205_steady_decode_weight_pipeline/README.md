# V205 steady decode expert down-projection

**Scheduler correction, November 19.** On the production mixed-batch entry, V202–V206 run the target outside the owner-projection context. V203–V206 also leave the sharded verification-head flag at its default. The earlier component tests enabled those features through a separate helper. Queued V204/16 measures mixed batching and expert changes with the full vocabulary gather and raw context features. V209 connects both optimizations to the production scheduler. [Scheduler replay and source](../v209_mixed_prompt_context_entry/README.md)

**2026-11-19 00:03 UTC.** The judge accepted V204 at 16 users as private job `7fcdf9abf39b486a`. Its observed state was queued. V205 and V206 remain available for a matched comparison after hardware results. [Judge observation](../../runs/nov19_initial_private_jobs.json)

V205 extends V204's prompt and mixed-verification changes to the target expert down-projection during ordinary grouped decode. The new schedule keeps four ordered Hopper matrix instructions in flight before waiting for their results. Operand preparation for the second half of each K tile can overlap with the earlier matrix instructions. The implementation preserves the prepared NVFP4 weight bank, native scale transfer, BF16 operands, FP32 accumulation order, routed experts and reduction across the four K-warps. [Two-file change](delta.diff) · [Full challenge patch](patch.diff)

| Native grouped row tile | Full waits per four-stage loop, V204 → V205 | Matrix instructions | Registers per thread | Spill loads and stores |
|---:|---:|---:|---:|---:|
| 8 | 12 → 8 | 32 | 96 | 0 |
| 16 | 12 → 8 | 32 | 96 | 0 |
| 24 | 12 → 8 | 32 | 96 | 0 |

Each warpgroup executes 32 waits across the complete K=4,096 reduction, down from 48. The native tuner chooses tiles using the real row count and kernel configuration. The small tile covers one- and four-user verification shapes in the archived workload, the middle tile covers 16 users, and the largest covers 32 users. Longer holding of a shared-memory tile can make the producer wait. Actual decode latency depends on how much conversion and synchronization lie on the critical path. [Native tuning](configs.json) · [Machine-code comparison](sm90a.json) · [Pipeline dependencies](kernel_evidence.md)

The portable arithmetic replay matched 13,824 FP32 outputs in 864 cases and checked 1,062,912 delayed register reads. Independent review followed all four math warpgroups in the compiled instruction order. A multijob replay covered 756 schedules and 348,516 interleaved operations through circular-buffer wraparound and expert-job transitions. All three previous down-projection fallbacks and all four gate/up binaries reproduced their archived digests. A startup replay exercised registration, fallback after a rejected binary, and incompatible tuner configurations. [Arithmetic results](cpu.json) · [Shared-memory release review](independent_release.md) · [Multijob replay](independent_multijob.json) · [Startup](startup.json) · [Gate/up binaries](gate_identity.json)

On the final source, 3,600 decode route selections and 288 joined calls exercised the normal and fallback paths. A separate connected fixture checked 1,048,576 finalized BF16 words and fed 10,240 of them through a synthetic vocabulary projection and real distributed verification. The official export is 293,076 bytes. It reconstructs from the pinned SGLang commit, reverses to the parent, and passes the published patch and flag restrictions. All six launch flags parsed in all 720 orders. [Decode routing](decode.json) · [Connected test](joined.json) · [Export validation](validation.json) · [CLI](flags.json) · [Challenge rules](https://openkernelchallenge.dev/rules)

V205 is frozen for comparison with V204. V204 took the November 19 private check at 16 users after its comparison and quota checks passed while V93 was queued. V64 holds the measured four-load score of 1.4119×. H200 token agreement, peak memory, time to first token and throughput for both candidates remain open. [Private reservation](../../runs/nov19_v204_runner.launch.json) · [Official score](../../runs/abd217baf0ed49fd.json) · [Four-load headroom](../v195_fused_expert_affine_verify/score_headroom.md)

V206 applies this scheduling change to the larger gate/up projection. The shared namespace regression also detects the missing import that failed V88 during GPU graph warmup. [V206](../v206_steady_decode_gate_up_pipeline/README.md) · [Regression fixture](check_global_names.py) · [V204 and V205 results](global_names.json)
