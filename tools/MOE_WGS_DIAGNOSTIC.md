# MoE scheduling diagnostic

`moe_wgs_diagnostic.py` is a diagnostic entry point, not a submission. Its
instrumented source must never be uploaded for an official check or score.
Root owns GPU allocation. The author ran only CPU validation and dry-run mode.

V8's separate finalizer passed 84 checks in paired Modal experiment
`20261104T040903Z-f1613663`, but whole-layer M32/M128/M256 ratios were
0.9877/1.0003/0.9984. It is rejected for promotion. Do not start a larger
scheduler rewrite until actual routing and critical-path evidence justify it.

## What the driver measures

The control CUDA string is extracted exactly from frozen decoder
`80a4eaa3d863881e8159bb16469a9e51b96352c134b1ae43a232afe412a88e7c`.
Its SHA-256 is
`d0d950c33764d6b1193ea0caac9db9fb6cde4d986cd07fa862adab7f830fa39d`.
The frozen standalone MoE source
`ef7b210bdeab136d22cd7ca81cdcdad44bcd0d456d98a714e10a19f7093d596a`
supplies the original weight packing and extension builder. Their CUDA cores
differ only by one comment; the driver verifies that relationship and still
uses the exact decoder text. Only the main WGS control and diagnostic variants
are compiled. Tiny/MID kernels, the whole decoder and its autotuning are absent.

The probe adds eight per-CTA completion timestamps: A and B completion from
lane 0 of each of the four compute warpgroups. All timestamps use the existing
`globaltimer`, which is comparable across SMs; no cross-SM `clock64` subtraction
is used. Existing `OKC_DEBUG` gives the surrounding routing/gather/start/barrier/
finalizer timeline. `OKC_WGSFIT` stays off because it repurposes two timestamp
slots. `OKC_WGSPROF` is only declared in this imported code and records nothing.

The existing phase A/B timestamps are from CTA thread 0 only. Therefore the
old B-to-barrier interval includes other warpgroups' completion as well as the
global barrier. The eight added values identify each warpgroup's completion. Missing, zero,
partial and nonmonotonic traces fail closed. Control/probe scratch and the
trace have trailing guards. Fresh trace storage is cleared before each replay.

M32/M128/M256 each run three input seeds with one separately recorded weight
seed. Four cold-L2 replays per implementation and input are the default.
Each replay's output is checked against the unchanged public MoE reference;
graph replay overhead is retained. Actual GPU-produced routing IDs, output
hashes, all 32 local counts, U (distinct experts), J (64-row jobs), and padded
job rows are recorded outside timing. The actual input bank is checked against
the prepared bank; it is not inferred from nominal task statistics.

**The input distribution is the standalone MoE reference, not the whole
decoder's router.** PDL is set as in decoder M32, but there is no upstream
projection/route overlap in this isolated experiment. A negative diagnostic
can reject a broad hypothesis; a positive result needs confirmation on actual
decoder routes before claiming a decoder speedup or promoting source.

## Gate before a scheduler rewrite

Let A_i and B_i be each warpgroup's phase completion times and L=max(A_i).
The report includes a **non-certifying phase-window heuristic** from
`D_i=min(B_i-A_i, max(0, L-A_i))` and `max(B_i)-max(B_i-D_i)`, divided by the
full MoE span. Independent per-WG allowances omit delays propagated through
cross-CTA phase-B split-unit fixups. For example, a late-A CTA can wait for
another CTA's B partial after that other CTA stalled on A readiness. Removing
the original stall can accelerate both CTAs, which the independent calculation
misses. A low heuristic must therefore **not** reject a scheduling rewrite.

A substantive rewrite requires evidence of at least 15% exposed removable
expert wall time plus physical headroom. To establish that, correlate per-stage x-producer `doneAs` wait
intervals with consumer `BW_F` waits for the same CTA/sequence, focusing on
the latest-finishing warpgroups. Producer spins can overlap useful compute;
`BW_F` also includes weight-TMA waits. Summed spin time is not the gate.
Reject tight conclusions if instrumentation changes median time by over 5%
in either direction.

Also require bandwidth headroom. An expert's packed W13+W2 including scales
occupies exactly 27 MiB. The driver reports a cold-HBM lower bound from
`27 MiB * U`, divided by 4.8 TB/s, and the
989-TFLOP/s math bound. It separately reports `27 MiB * J` as nominal weight
requests: repeated jobs may reuse L2, so that request estimate is **not** an
HBM lower bound. The HBM headroom ratio uses graph timing with overhead retained
and is therefore optimistic. Minimum input/output bytes are reported separately,
using the actual number of tokens with a local route. A uniform 35–40% expert improvement must not be
assumed when the physical lower bound leaves less room.

If evidence supports a scheduler test, the simplest structural probe keeps
the current cost partitions and changes their phase-B ownership through a
bijection. Both producers/all consumer warpgroups must use the mapping, and
`wgs_cta_of` must apply its inverse for B split-unit fixup ownership. Choose
the mapping from measured dependency structure, not an arbitrary offset sweep.
The Lee pinned v56 alternative uses heavy-expert-first whole-output-unit
atomic claiming and next-unit prefetch; its apparent interleaving is dormant
in production (LAG=64 with at most 32 local experts). Do not describe it as
an active ready-first queue. Whole-unit dynamic scheduling is a later option
if measured load imbalance, rather than readiness, dominates.

## Commands and allocation boundary

```bash
python3 tools/moe_wgs_diagnostic.py self-test
python3 tools/moe_wgs_diagnostic.py dry-run --timeout 900
# Root only, after review and a successful current credit guard:
.venv-modal/bin/python tools/moe_wgs_diagnostic.py run --timeout 900
```

Run mode reuses `modal_credit_budget.allocation_guard`, its exclusive lock and
ledger. It requests one H200, four CPU cores, 32 GiB RAM, blocked network and
60–900 seconds, with termination in `finally`. Default timeout is 600 seconds;
900 leaves headroom for two cold native compiles. Every executed native source,
helper, worker and credit-guard source/hash is frozen under `experiments/modal/`.
No reference or submission file is edited. Dry-run mode makes no Modal call.
