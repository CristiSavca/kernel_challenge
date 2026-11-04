# Modal H200 experiments

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
budget intentionally errs on the conservative side. At current settings a
900-second benchmark reserves $2.42. This is an estimate and local guard, not
a provider-enforced spending cap; unrelated clients/hosts are outside its lock.

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
checks/NVFP4 helper, and the worker are uploaded. The whole project, account
files and credentials are not mounted. Every uploaded file is frozen and hashed
under `experiments/modal/<run>/`; results, logs and termination status stay there.
All weights/inputs are synthetic. No model download is required.

Correctness uses the public reference, including returned cache entries when
the problem supplies `collect`. It checks two fresh eager inputs, then every
fresh CUDA graph replay. Candidate and baseline use separate state and inputs;
timing alternates their order with a 128 MiB L2 flush before each measurement.
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
