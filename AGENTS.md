# Open Kernel Challenge working notes

Read the problem's own AGENTS.md and TASK.md before changing its submission.
Keep the official baseline.py, reference.py, checks.py and nvfp4.py unchanged.
The first Git commit preserves every official starter.

The user already installed okc and signed in as CristiSavca. Never print, save,
or commit an API token. A human must accept any updated challenge terms; agents
must not call the acceptance endpoint or answer an acceptance prompt.

This Mac is Apple Silicon, with no CUDA runtime. Compilation/correctness/timing
must be confirmed by the official remote H200 checks. A local syntax check alone
does not establish GPU correctness. Mark queued and untested changes honestly.

Use private `okc check` runs for experiments; inspect live `okc problems` limits.
Preserve the exact source and job result, preferably with tools/experiments.py.
Publish a scored candidate after a passing check with a useful measured gain.
Do not make duplicate retries while a job is queued. Avoid frequent status polls.

Only real computation speedups: preserve model math, required BF16/FP8/NVFP4
precision, every routed expert and every attended key. No output/input caching,
harness inspection, timing changes, seed-specific logic or workload shortcuts.
Only weights may be repacked in prepare; never move input computation there.

serving/sglang is a separate shallow Git checkout pinned by the official CLI.
Keep its Git history intact; the parent repository intentionally ignores it.
Export serving changes with `okc serve diff serving` and preserve the patch here.
Do not download model weights or create a paid GPU rental without authorization.

Keep generated scratch bounded. The setup had about 145 GiB free; the serving
source is about 192 MiB. Preserve useful results and exact tested candidates.
