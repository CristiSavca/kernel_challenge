# FT128 expert-job reuse: reject as the next broad decoder rewrite

2026-11-04. Read-only structural audit against compact candidate
`cf09ba584cc1537802636bb905ac08cda2095f83a163358e1c9bb716490130b8`.
No candidate changes, GPU allocations, submissions, or job polls. Reproduce with
`/tmp/okc-kperm-cpu-20261104/bin/python tools/analyze_ft128_reuse.py`.
Machine-readable evidence: `experiments/iterations/20261104-ft128-reuse-analysis.json`.

**Decision:** do not implement FT128 now. It has a real benefit for sufficiently
hot experts, but observed standalone routes and a separate distribution model
show too little normal-workload reuse to justify doubling stage bookkeeping and
the largest accumulator. Decoder histograms are still missing; standalone data
cannot exclude a decoder-specific opportunity. This is a priority decision, not
a proof that every FT128 implementation loses.

## Prior work and exact opportunity

`POST_NCU_TRAFFIC_REASSESSMENT.md` already rejected large weight-reuse claims:
measured M256 has U26/J64=27 and only one duplicate expert pass. No previous FT128
candidate or explicit FT128 rejection was found in tools, iteration evidence or
MoE candidates. The prior FT32/four-slot analysis concerns ring depth and is a
different mechanism.

For local expert count c, J64=ceil(c/64), J128=ceil(c/128). Each job requests
27 MiB of NVFP4 codes plus E4M3 scales over W13 and W2. Thus the removable logical
packed requests are **27 MiB × (J64−J128)**, not 27 MiB × (J64−U). These are
requests; repeated loads may hit cache, so this is not measured HBM savings.

| c per expert | J64 → J128 | That expert's request saving |
| --- | --- | --- |
| 1–64 | 1 → 1 | 0% |
| 65–128 | 2 → 1 | 50% |
| 129–192 | 3 → 2 | 33.3% |
| 193–256 | 4 → 2 | 50% |
| 257–320 | 5 → 3 | 40% |
| 321–384 | 6 → 3 | 50% |

The actual prior three standalone route draws give:

| M | U | J64 → J128 | Request savings |
| --- | --- | --- | --- |
| 128 | 25 / 26 / 23 | 26→25 / 26→26 / 24→23 | 3.85% / 0% / 4.17% |
| 256 | 26 / 27 / 23 | 29→27 / 27→27 / 26→24 | 6.90% / 0% / 7.69% |

The separate NCU M256 histogram has one count of 73, J64=27 and J128=26.
Its removable 27 MiB is 3.6783% of 769,691,136 measured DRAM read bytes. This is
traffic accounting, not a measured FT128 speedup. No saved M32 route can benefit.

## Distribution and general valid inputs

The audit models `muse_moe_w4a16/reference.py:53`: uniform scores plus the exact
per-expert heavy-tail formula for normal inputs, or uniform scores plus
4×Uniform^6 for skewed inputs; select eight distinct experts, then select the
busiest 32-expert shard. There are 512 independent NumPy trials per row. This
matches the routing distribution formulas, **not Torch CUDA RNG seeds**, and
does not model decoder routing. No sampled histogram is used by kernel code.

| M | Normal mean saving | Normal median / p90 | Zero-saving draws | Skewed mean saving |
| --- | --- | --- | --- | --- |
| 128 | 1.81% | 0% / 5.00% | 62.7% | 39.40% |
| 256 | 4.29% | 3.85% / 8.33% | 26.2% | 42.78% |
| 384 | 6.82% | 6.90% / 12.00% | 7.0% | 44.61% |

M384 and skewed routing matter for generality; they are not extra scored loads
in the current standalone task. All eight local experts receiving every token
is a valid histogram with 50% savings. Many cold experts dilute one hot expert's
gain. Equal route count and U do not determine reuse: at M256, U24/R1000,
`[42]*16+[41]*8` has zero saving, while `[128]*7+[7]*2+[6]*15` saves 7/31=22.58%.
Both fit the per-token eight-distinct-routes constraint after assigning rows.

The decoder reference constructs routes after attention from structured hidden
states and a learned router. Its `weights_fields` selects a router matching
mean route-count and distinct-expert targets; those targets do not determine
J64/J128. Consequently the table above is not a decoder prediction. Even under
the optimistic conditional assumption that the normal mean request savings
become equal MoE latency savings, the old 55%/54% verify-load MoE shares imply
only about 0.55% aggregate decoder latency reduction across six equal loads.

## Resource and schedule costs

Native clang extracted the actual compact `Smem` struct without importing or
executing the candidate. These are layout budgets, not CUDA compilation results:

| Design | JMAX | Metadata rounded | Three weight slots | Three x slots | Dynamic SMEM | Two CTA shared-only |
| --- | --- | --- | --- | --- | --- | --- |
| FT64 / K128 | 88 | 21,248 B | 27,648 B | 49,152 B | 98,048 B | yes |
| FT128 / K128 | 64 | 20,480 B | 27,648 B | 98,304 B | 146,432 B | **no** |
| FT128 / K64 | 64 | 20,480 B | 13,824 B | 49,152 B | 83,456 B | yes, unverified registers |

The two-CTA comparison includes 1,024 driver bytes per CTA and 228 KiB/SM.
Hopper provides 64K 32-bit registers per SM, so two 320-thread CTAs permit only
102.4 registers/thread before allocation granularity; the present uniform
register allocation is effectively constrained near 96. See the primary
[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/).

FT128 raises the worst NTT from 8 to 16: each consumer needs 64 accumulator
registers instead of 32, plus the existing 16 register-A operands and control,
scale, address and pending-work state. `d_odd` is only used at NTT1 and should not
be incorrectly counted as another 64 live registers at NTT16. Actual ptxas
allocation, spills and serialization remain unknown. A single compiled kernel
can inherit the worst specialization's resource allocation even for cold jobs.

K64 is therefore a substantial rewrite, not changing one macro:

- Full-K stages double 32→64. Across a histogram, joined producer events change
  by **2×J128/J64**, always at least 1 and near 1.9 for ordinary routes. Payloads
  shrink while barrier, descriptor and readiness work repeats more often.
- Total padded activation rows are exactly unchanged: splitting at multiples of
  64 or 128 preserves sum(ceil(c/8)). With 96 output units/job, activation read
  requests remain 96×4096×2×padded_rows bytes. Weight-request savings do not also
  halve activation traffic or padded tensor arithmetic.
- Add NTT9–16 WGMMA specializations/dispatch and replace eight-entry packed cost
  tables. Existing shifting by 8×(NTT−1) is invalid beyond their 64-bit storage.
- K64 has two GS2 commit groups (ci0–1), so the current deferred flush at FCI3
  would never run. Flush lifetime, ring order and split-unit progress need a new
  proof. Packed copies, descriptors and all 32-stage indexing need revision.
- One phase-A 128-row unit produces 64 intermediate columns, so K64 readiness
  becomes two WG arrivals per stage, with 64 stages/job and matching reset loops.
  At JMAX64 this is 16 KiB of counters; the existing 136×32 counter allocation
  happens to cover it, but existing indexing/reset logic does not.
- Stream-K fixup storage doubles 20→40 MiB; maximum padded activation capacity
  decreases only 3,776→3,584 rows. Existing oversized activation allocation can
  remain, but the current fixup allocation cannot. FP32 normalization, all routes
  and all scratch generations must remain intact.

FT128/K128 instead loses the proposed two-CTA residency. Keeping a 264-CTA grid
with the internal grid barrier would fail its forward-progress premise; that
path needs a different admitted grid/schedule. Shared-memory fit alone never
authorizes either launch.

**Reopen criterion:** fresh, untimed diagnostics outside the submitted kernel
show at least 15% removable request bytes in representative actual decoder
M128/256 histograms across multiple router/input draws, with a material weighted
share of decoder latency. Then propose one separately reviewed architecture,
prove all zero/skewed/ragged-route schedules and obtain actual two-CTA resources
before any performance claim. Current evidence does not meet that gate.
