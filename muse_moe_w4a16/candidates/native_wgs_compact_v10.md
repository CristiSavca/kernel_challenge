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

Static shared-memory accounting is98048B/CTA, versus231168B in the parent.
This is not measured CUDA occupancy. Two compact CTAs retain four compute WGs
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

The exact-source independent review and official private GPU result are pending.
Only root may submit; do not score or integrate this candidate yet. Full plan,
primary research, kill criteria and corrected PDL analysis are in
`tools/MOE_COMPACT_RESIDENCY_PROPOSAL.md`.
