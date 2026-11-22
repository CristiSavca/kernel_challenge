# V74 two-CTA Hopper grouped decode

V74 admits two resident CTAs per SM for compatible Humming W4A16 decode
projections. V72 uses one CTA per SM with a 640-thread block. V74 retains
the same block and warp shapes, four pipeline stages and each output tile's
complete FP32 K accumulation. Humming's supported launch-bounds and warp
register allocation implement the change.

- Exact parent V72 SHA-256 `e90c47ab73e66158b678893bb6fc7950c3b52f38cc3213ac6d7b7716d1b19705`.
- Base SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`.
- V74 patch SHA-256 `d6b6ab0136ca6ed063ba8ae31d95f5422cf1fc24c57813127f23c968fcf2ce9b`.
- 130,193 bytes, 23 permitted source files, maximum added line 106 characters.
- State is CPU/source checked and H200 untested.

Use the five V72 launch flags, including `moe-runner-backend=humming` and
`speculative-dflash-tree`. The new switch
`SGLANG_ENABLE_MUSE_SPARK_HUMMING_DECODE_TWO_CTAS=0` restores the parent
residency choice. Its default enables the new candidate under the existing
explicit Humming backend selection.

## Resource selection

Preparation evaluates the exact pinned Humming 0.1.12 shared-memory
estimator against the actual layer metadata. Compatible SM90 decode
variants must have 512 math threads and 128 load threads, at least three
stages and original CTA count one. Two CTAs must fit the device's reported
shared-memory and resident-thread limits. The device must supply the 65,536
registers assumed by Humming's Hopper warp register allocator. Indexed and
large-prefill tuning stays at the inherited settings.
[Pinned Humming dependency](https://pypi.org/project/humming-kernels/0.1.12/)

The exact EP8 resource results are:

| Projection | M tile | Shared bytes per CTA | Chosen CTAs per SM |
| --- | ---: | ---: | ---: |
| W13 | 8 | 82,944 | 2 |
| W13 | 16 | 91,136 | 2 |
| W13 | 24 | 99,328 | 2 |
| W2 | 8 | 91,136 | 2 |
| W2 | 16 | 107,520 | 2 |
| W2 | 24 | 123,904 | 1 |

Hopper supplies 233,472 shared bytes and 2,048 threads per SM. For admitted
variants, 1,280 threads comprise 40 resident warps, compared with the
parent's 640 threads and 20 warps. The vendor allocator budgets 40 registers
for each load thread and 48 for each math thread. This is 29,696 registers
per CTA and 59,392 across the pair. The parent budgets 96 registers per
math thread. Actual compiler allocation and spills need GPU inspection.
[NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)

The A2 multicast pair retains identical expert, M, K and tail coordinates.
Each pair computes adjacent N tiles. The launch grid doubles and distributes
the same output tiles across twice as many CTA groups. Humming's existing
cluster barriers and per-CTA shared arrays retain their original scope.
Stream-K remains disabled, so there are no new partial output reductions.

## Opportunity and limits

The change can improve latency hiding while packed weights and dequantized
operands traverse the existing pipeline. EP8 target rows 8, 32 and 128 use
eligible tiles for both projections. At 256 rows, W13 remains eligible and
W2 retains the parent resource setting. The exact packed-weight volume,
activation staging, global-scale BF16 rounds, SiLU, per-route normalization
and final slot-order sum follow the parent computation.

Under uniform independent top-eight routing, the approximate packed reads
per rank and sparse layer at 8, 32, 128 and 256 target rows are 194, 551, 849
and 864 MiB. These are also the V72 volumes. This candidate supplies more
resident work to overlap memory and arithmetic. It introduces no additional
weight bank or activation tensors. The grouped TMA C descriptor reserve is
zero because the inherited `use_tma_c=False` remains in effect.

The lower math-register budget can cause spills. More resident CTAs can
also increase cache competition or cluster scheduling pressure. Occupancy
does not determine throughput. For any measured expert speedup `s` and its
decode time fraction `f`, the decode-only bound is
`1 / (1 - f + f / s)`. The full serving score additionally depends on
prefill, acceptance and the other decode operations. V74 has no measured
speedup or score, and it supplies no evidence of a 2x end-to-end result.

## Verification

The [CPU source harness](cpu_check_v74.py) executes the exact vendor
`TuningConfig`, SM90 tuner and shared-memory estimator through Python AST
extraction with CPU dtype/device shims. It checks the authored helper.

- 3,072 exact configuration comparisons across EP4/EP8 and both projections.
- 9,216 shared-memory, resident-thread and register-limit rejections.
- 768 device, deterministic, feature-switch and expert-count guard cases.
- 1,600 tile-ownership comparisons between the original and doubled grids.

The [complete results](cpu_result.json) retain per-geometry resource values.
The [export receipt](proof.json) records official local `okc serve diff`
byte equality, forward/reverse patch application, whitespace checks, all
Python source compilation and file-mode preservation. Every parent file
except the tuning helper and environment field has exact V72 parity.
An independent reviewer repeated the complete CPU suite, verified the
frozen patch digest and passed the source-level resource and scheduler
review. The exact applied tree is `be513987e27f828eff0b1f3207f41531f038823e`.

The candidate needs H200 compilation, cubin register/local-memory inspection,
static and changing-route graph replay, numerical and later-turn agreement,
output lengths, TTFT, throughput and measured peak memory. No remote GPU
job or submission was started, and scheduled checks remain unchanged.

The inspected scratch and isolated export checkout were retired after
Git/stash, open-file and durable hash checks. Allocated scratch was
49,596 KiB before the final alternate-index write. Available space rose
from 31,178,096 to 31,229,228 KiB at removal, an observed gain of 51,132 KiB.
