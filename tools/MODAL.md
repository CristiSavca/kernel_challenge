# Modal H200 experiments

The fixed `attention-v11-screen --timeout 600` mode tests the reviewed
compile-time-role revision after V10 failed its register/spill gate. It changes
only the pinned candidate in the existing eight-case attention protocol; the
parent, resource/activation gates, 128 checks, paired timing and allocation guard
are unchanged. It requires fresh credit preflight and independent source review.
See `tools/ATTENTION_Q8_V11_ALLOCATOR_HYPOTHESIS.md` for the falsifiable hypothesis.

Setup checked November 3, 2026. Modal 1.6.3 is installed in `.venv-modal` with
Python 3.13. The CLI is authenticated to workspace `cristisavca44`; credentials
remain in the user's normal Modal configuration, outside this repository.

## Access verified

The user completed payment verification on November 3 evening. A fresh billing
page showed the full $30 monthly Starter allowance. Smoke
`experiments/modal/20261104T035741Z-c81d970d` passed on one NVIDIA H200
(143771 MiB, driver 610.57.04), exited zero and terminated its sandbox.
The app is stopped with zero tasks. CLI billing showed about $0.00276863 of
ephemeral compute; aggregate billing can lag its resource breakdown.

The initial pre-verification state below is retained as history.

## Initial access blocker (resolved)

The first smoke request was rejected before GPU allocation: `Please add a
payment method to use H200 GPU sandboxes.` The account showed $1 in included
compute and $0 usage. Its Add payment method dialog explicitly says:

- Verification unlocks the full $30 monthly Starter compute allowance.
- Modal charges a one-time $0.50 verification fee, which it says is refunded.
- The card must not already be used on another Modal account.

No paid plan upgrade was needed. Inspect the actual balance and workspace
spending limit before launching work. Authorization covers free-credit
experiments, not charges beyond credits. The failed request's receipt is in
`experiments/modal/20261103T233733Z-177140a4/manifest.json`.

Sources: [GPU access](https://modal.com/docs/guide/gpu),
[Starter pricing](https://modal.com/pricing),
[workspace billing](https://modal.com/settings/cristisavca44/usage).

Public pricing rechecked November 3 evening: Starter includes $30 per month
for GPU/CPU/memory compute, including Sandboxes. The supplied `pip install`
and `modal setup` instructions install and authenticate the client; they do
not establish that a separate promotional grant was redeemed. A search of
the official Modal and challenge sites did not identify a challenge-specific
H200 credit offer. The observed account dialog remains the evidence for the
payment-verification requirement; an undisclosed private promo is unverified.

## Run one experiment

Every allocation now uses `tools/modal_credit_budget.py`. It checks the
reviewed monthly grant, active profile, fresh billing and running apps, holds
an exclusive local lock, and reserves conservative compute/build costs before
allocation. At least $5 is kept outside the working budget. Reservations never
refund automatically and are additional to observed provider usage, so the
budget intentionally errs on the conservative side. A separately reviewed,
immutable certificate now removes $3.74 of proved posted-cost double counting
from nine completed November 4 sandboxes. All original reservations and the
$3.90423640 provider high-water remain counted; future reservations retain their
full charge. Every allocation rechecks the certificate and fresh exact-interval
billing attribution. Missing or downward-revised evidence fails closed. See
`MODAL_OVERLAP_REVIEW_20261104.md` and `modal_credit_overlap_20261104.json`.
At current settings a 900-second benchmark reserves $2.42. This is an estimate
and local guard, not a provider-enforced spending cap; unrelated clients/hosts
are outside its lock.

Preserve `experiments/modal/.allocation.lock` (never unlink its inode), the
monthly credit reservation journal and referenced manifests. Unconfirmed
cleanup or missing budget evidence blocks future allocations until reviewed.
The observed grant is pinned in `tools/modal_credit_grant.json` and expires
at the month boundary; do not raise it or recycle reservations automatically.

From the project root, no activation or global Python changes are needed:

```bash
.venv-modal/bin/python tools/modal_h200.py smoke
.venv-modal/bin/python tools/modal_h200.py benchmark \
  --problem muse_moe_w4a16 \
  --candidate muse_moe_w4a16/candidates/public_local_align_v4.py \
  --timeout 300
```

Add `--dry-run` to validate the inputs and inspect the exact manifest without
creating any cloud resources. `--cases cases.json` accepts 1–64 explicit input
configurations; seeds are set with `--seed` and `--weight-seed`.

Use `--incumbent <exact-passing-source.py>` for a paired comparison against
our incumbent on the same GPU. The uploaded comparison slot named `baseline`
then contains that incumbent; it is explicitly identified and hashed in the
request, manifest and result. The official baseline file remains unchanged.
Those ratios compare two candidates, not a candidate against SGLang.
Add `--profile` for one extra graph replay per implementation after timing.
It records actual CUDA kernel names, counts and durations separately from
benchmark medians, or an explicit profiling error; no full trace is written.
`--resources` records `cuobjdump --dump-resource-usage` for each module-owned
`_EXTS` shared object, with the compile defines, bounded subprocess time and
bounded output. REG/STACK/LOCAL describe compiler resources, not dynamic spill
traffic; successful compilation cannot prove that WGMMA was not serialized.
The standalone MoE module exposes `_EXTS`; the decoder's `_MA_EXTS` is not
covered by this optional collector.

Every weighted run records each prepare's wall time including its weight clone.
Decoder runs additionally freeze and upload `decoder_policy_metadata.py`, which
records a bounded whitelist of scalar dispatch policies immediately after each
prepare. It never invokes builders or reads tensor values. This distinguishes
an accepted backend change from unrelated autotuner differences; unsupported
objects and truncated fields are explicit in the result.
Standalone MoE replays also record actual local route counts, U, J, and jobs
with at most eight rows before timing. Partial 64-row jobs count separately:
65 routes to one expert produce jobs of 64 and one row. These observations
describe the generated standalone inputs, not the decoder's routing.

The dedicated `attention-v10-screen` mode is a fixed experiment for the exact
decoder-attention parent and isolated V10 wrapper. It pins five uploaded files,
eight cases, seed121021 and six paired replays; accepts only the usual bounded
timeout/dry-run controls; and uses the same allocation guard and cleanup.
It captures complete actual ptxas diagnostics before any attention launch,
then requires zero local memory, two-CTA occupancy for the exact Q8 flat
specialization, and observed target dispatch. Compiler/resource checkpoints
survive ordinary worker failures, with best-effort recovery before termination
on timeout. See `tools/ATTENTION_Q8_V10_SCREEN_HANDOFF.md` for hashes, limits and
the distinction between original and host-instrumented sources. Use
`python tools/modal_h200.py attention-v10-screen --timeout 900 --dry-run`
to validate without allocation. This is not a standalone leaderboard candidate.

The smoke requests one H200 for at most 120 seconds and verifies `nvidia-smi`.
The benchmark requests one H200, four CPU cores and 32 GiB RAM, with CPU/memory
ceilings. Its default maximum lifetime is 300 seconds; the CLI accepts at most
900 seconds. Each invocation makes one allocation attempt, uses no deployment
or persistent volume, and terminates its sandbox in `finally`. Modal enforces
the remote timeout even if the local process disconnects. Network access inside
the sandbox is blocked. Image pulls happen on Modal, not this Mac.

Observed pricing: H200 $4.54/hour; Sandbox CPU $0.1419/core-hour and memory
$0.024/GiB-hour. At the requested resource ceilings a five-minute run is about
$0.49, excluding separate image-build charges. This estimate is not a billing
cap; check the current balance/rates and leave headroom. Do not start a long
batch from the initial $1 allowance.

## What the benchmark establishes

The worker uses the public SGLang image, pinned to linux/amd64 digest
`sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af`.
The image provenance is retained in the imported benchmark manifest. The private
judge has its own image and can change independently.

Only the selected source, the problem's official public baseline/reference/
checks/NVFP4 helper, the worker, and the decoder-only policy collector are uploaded. The whole project, account
files and credentials are not mounted. Every uploaded file is frozen and hashed
under `experiments/modal/<run>/`; results, logs and termination status stay there.
All weights/inputs are synthetic. No model download is required.

Correctness uses the public reference, including returned cache entries when
the problem supplies `collect`. It checks two fresh eager inputs, then every
fresh CUDA graph replay. Candidate and baseline use separate state and inputs;
timing alternates their order with a 128 MiB read sweep before each measurement.
The sweep uses Triton `.ca` loads and observable per-block sums; it compiles
and verifies those sums before capture. It runs on the same stream, outside
timing. This intends to evict L2 but does not measure candidate DRAM traffic.
Earlier frozen experiments used a write flush and retain that limitation.
The replay overhead is retained. Whole-decoder expert-bank scalar metadata is
fixed per case so it remains compatible with the captured graph.

The default cases are a small debugging subset, **not the complete official
suite or an official score**. There is no official router-load calibration,
load-weighted aggregate or confidence interval. A promising local result must
still pass `okc check` before the exact same source is scored.

Local lifecycle tests pass:

```bash
.venv-modal/bin/python -m unittest discover -s tools -p 'test_modal_h200.py' -v
```

These check resource bounds, cleanup after failure, explicit upload boundaries
and dry runs. They do not establish candidate CUDA compilation or numerical
correctness. The smoke now verifies H200 allocation and cleanup; each candidate
still needs its own benchmark and official check.

The separate reviewed `tools/moe_wgs_diagnostic.py` entry point uses the same
mandatory credit guard. See `tools/MOE_WGS_DIAGNOSTIC.md` for its descriptive
trace limits; it is never eligible for submission. Its first run passed all
72 checks and terminated. After that run, account billing showed $1.06 covered
by credits and $0 billed. These figures are a dated snapshot, not a fresh budget.
