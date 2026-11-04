# Compact WGS v10

Candidate source SHA256:
`cf09ba584cc1537802636bb905ac08cda2095f83a163358e1c9bb716490130b8`.
CUDA source SHA256:
`789c1a25dbf93483acc0715748110b2cb90560ed7156f4969d813a4b02e0bf82`.
Parent is scored `ef7b210b...`, without KPERM or other experimental changes.

This tests two compact resident CTAs per SM for the MAIN WGS path (M>16 with
supported PRMT scales). Each CTA uses two compute warpgroups and two producer
warps,128 output rows,K128 and three joined ring slots. FT_T64 and expert job
counts stay unchanged. Tiny/MID and non-PRMT fallbacks retain parent computation.
The host queries actual occupancy before any spin-barrier launch and rejects
the experiment if either compiled specialization cannot support two CTAs/SM.

Shared-memory accounting is98048B/CTA, versus231168B in the parent; the private
GPU check below confirms two-CTA resource capacity. Two compact CTAs retain four compute WGs
per SM, double producer warps, halve the aggregate weight-ring capacity, and
keep the aggregate activation-ring capacity. Independent progress may help;
more simultaneous CTAs does not establish more bandwidth. Weight payload stays
27MiB per expert job. Activation requests double because output tiles halve.
Streaming FP32 normalization reads ybuf twice to bound register pressure.

CPU proof passed for the executable code (d27b60fd source before a comment-only provenance correction); complete Python AST and CUDA bytes are unchanged. Command:

```sh
/tmp/okc-kperm-cpu-20261104/bin/python tools/test_compact_wgs_cpu.py
```

Saved evidence: `experiments/iterations/20261104-moe-compact-cpu.json`.
It checks100663296 actual packed FP4 codes and6291456 scale bytes for full W13
and W2 experts in both bit encodings;1179648 activation coordinates for all
NTT1..8;16 preprocessed fallback builds; and the candidate's extracted actual
C++ scheduling functions over146 histograms,209656 ranges and64296960 stage
assignments. Slot/reset arithmetic and native metadata layout are checked.
Independent critic additionally models408 routing histograms, including the
79-job realizable maximum-token stress. CPU proofs do not verify GPU ordering,
numerical tolerance, compiler resources or performance.

## Rejected after the official private check

Exact-source review passed before root submitted private job `94e63981c631474e`.
The completed check passed **5/5 tests and all 6 benchmark cases**, with no rule
flags. It returned **2.04721554× [2.01650730, 2.07792377]**, versus parent public
job `3c15b408fdb04c44` at **2.35610395× [2.32076239, 2.39144551]**. This is a
13.11% lower score, equivalent to 15.09% higher normalized aggregate latency.
The intervals do not overlap. Both receipts use evaluator hash
`c8a380059e3d1f0afaa9914d6d48b09cdc83efc7e110e7301bca3413720092c0`;
they use different input seeds and GPU nodes, so this is a cross-run comparison.

| M | Path | Parent paired speedup | Compact paired speedup | Relative normalized latency |
| --- | --- | --- | --- | --- |
| 1 | unchanged tiny | 3.0518× | 3.1403× | −2.82% |
| 16 | unchanged MID | 1.9962× | 2.0421× | −2.25% |
| 32 | compact MAIN | 1.6518× | 1.2341× | **+33.84%** |
| 8 | unchanged small | 2.2714× | 2.2649× | +0.29% |
| 128 | compact MAIN | 2.5952× | 1.8996× | **+36.62%** |
| 256 | compact MAIN | 2.8840× | 2.1621× | **+33.39%** |

Relative latency above is parent paired speedup / compact paired speedup −1,
not a ratio of raw profile durations. The second parent check
`143f8587286e4384` independently returned 2.36050170× and likewise higher MAIN
speedups. The unaffected paths are close, while all changed scored shapes lose
substantially. This is sufficient to reject promotion without a retry.

The actual host resource diagnostic was:

```text
OKC_COMPACT_WGS resources threads=320 dynamic=98048 occupancy=2/2 regs=96/96 local=64/64
```

Both compiled specializations therefore meet the runtime CUDA occupancy gate
of two CTAs/SM, at 96 registers/thread. `local=64/64` is CUDA's local-size
attribute in bytes/thread; the log does not identify its source, dynamic traffic
or spill cost. Occupancy capacity is verified, not the fraction of time both
CTAs were active. No full ptxas/SASS report or hardware counters were captured.

The source establishes costs that may explain the loss, but does not identify
which dominates: a fixed expert job changes from 48 output units ×16 K256
stages to 96 units ×32 K128 stages, **four times as many joined producer stage
events**. Each weight payload is one quarter as large, so total packed weights
stay 27 MiB/job. Activation payloads halve, so total activation requests double.
The final FP32 normalization adds a second ybuf read. Two CTAs still provide
four compute warpgroups/SM, with twice the producer warps and half the aggregate
weight-ring capacity. These are logical requests/events, not measured extra
DRAM traffic or an attributed latency breakdown. The tested hypothesis that
independent CTA progress would outweigh these costs is falsified here.

**Do not integrate, score, or create a derivative from this result alone.** Keep
the exact source and receipt as a rejected experiment. Full plan, primary
research, kill criteria and corrected PDL analysis remain in
`tools/MOE_COMPACT_RESIDENCY_PROPOSAL.md`; resource and numerical success do not
override the failed performance gate.
