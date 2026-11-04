# Preserve WGS stage size, seek two CTAs: no implementation recommended

November 4, 2026. Read-only feasibility audit of exact scored standalone
`247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`.
No candidate, GPU allocation, submission or polling. Evidence:
`experiments/iterations/20261104-main-wgs-two-cta-feasibility.json`.

**NO GO as specified.** Keeping 256-output-row/K256 stages avoids the rejected
compact rewrite's fourfold stage count and doubled activation requests. But
FT64 cannot fit two CTAs even without metadata. Shape-sized metadata requires
FT16 for the larger loads, where observed routes add substantial weight passes.
Moving the finalizer does not establish the required register allocation. FT32
at M32 is a smaller, unproved opportunity affecting one decoder load; it does
not justify this as the next broad improvement. No parameter sweep is proposed.

## Exact shared-memory budget

Parent geometry: 576 threads, four consumer warpgroups, 36 KiB weight stage,
FT×256×2-byte activation stage. Two slots require
`metadata + 73,728 + 1024*FT` dynamic bytes per CTA. Including the observed
1,024 driver bytes per CTA, two CTAs must each use at most 115,712 dynamic bytes
within Hopper's 228 KiB/SM. Hardware limits are documented in the
[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy).

The actual parent `Smem` was extracted and compiled with native clang for all
fifteen MMAX/FT combinations. Its size equals `4064 + 5*EMAX + 32*JMAX`, with
EMAX=8×MMAX and JMAX=40+EMAX/FT, rounded up to128 bytes for the ring base.
These are source-layout budgets, not CUDA compilation or occupancy results.

| MMAX / FT | Rounded metadata | Two-slot dynamic | Shared-only two-CTA fit |
| --- | ---: | ---: | --- |
| 32 /64 | 6,784 B | 146,048 B | no |
| 32 /32 | 6,912 B | 113,408 B | yes |
| 64 /32 | 8,448 B | 114,944 B | yes |
| 128 /32 | 11,520 B | 118,016 B | no |
| 128 /16 | 12,544 B | 102,656 B | yes |
| 256 /32 | 17,664 B | 124,160 B | no |
| 256 /16 | 19,712 B | 109,824 B | yes |
| 384 /16 | 26,880 B | 116,992 B | no |

FT64 rings alone are139,264 bytes, so no metadata cleanup can fit that case.
With an ordered standalone finalizer, MAIN could stop retaining `S.pos` after
writing a persistent mapping for the finalizer. Dropping that array alone gives
FT32/MMAX128 **115,968 B**, still256 B too much, and FT32/MMAX256 **120,064 B**,
still4,352 B too much. M128 could fit with additional field removal; this audit
does not claim arbitrary metadata/route rewrites are impossible. Root requested
this bounded clarification only, and no further rescue design is proposed.

## Saved routes: exactly what narrower jobs cost

For each local expert count c, JFT=ceil(c/FT). Every job requests27 MiB and has
48 output units×16 joined stages=768 stage events. The table uses saved real
standalone diagnostic histograms, not decoder routes or a timing prediction.

| M | J64→J32, three saved draws | Added FT32 passes | J64→J16 | Added FT16 passes |
| --- | --- | --- | --- | --- |
| 32 | 16→16;16→16;18→18 | 0% in all draws | 16→17;16→16;18→19 | 0–6.25% |
| 128 | 26→28;26→26;24→27 | 0–12.5% | 26→32;26→29;24→33 | 11.54–37.5% |
| 256 | 29→33;27→30;26→33 | 11.11–26.92% | 29→44;27→39;26→48 | 44.44–84.62% |

Each additional pass adds27 MiB of logical packed-weight requests and768 joined
stages; these are not measured DRAM bytes because repeated weights may hit L2.
The model applies generally: FT32 never adds a job when c≤32, but a valid expert
with64 routes doubles its passes; FT16 quadruples them. Eight always-local hot
experts are a valid worst example. No seed or histogram enters submitted code.

Padded activation rows sum(8×ceil(c/8)) are invariant when splitting at64/32/16;
this was checked for every c0–384. With output geometry unchanged, total logical
activation requests stay `48×4096×2×padded_rows`, and padded tensor arithmetic is
unchanged. Unlike compact WGS, there is no automatic twofold activation penalty
or fourfold stage factor: stage amplification is exactly Jnew/J64. However,
two resident CTAs double routing/prologue copies and change split/fixup work.

## Registers, scratch and ordering remain unresolved

Two576-thread CTAs impose a necessary aggregate register ceiling of
65536/(2×576)=56.89 words/thread before allocation granularity. This is not a
proof that56 registers suffice on the actual device; both compiled
specializations would need the real occupancy API. NVIDIA discusses allocation
granularity and authoritative occupancy queries in its
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy).

Parent MAIN measured96 registers/local0. Its finalizer caches64 FP32 values per
thread, but the prior V8 split-finalizer experiment saved no compiler resource
capture; it cannot prove that this tail caused96 registers or that removing it
reaches56. V8's whole-decoder M32/M128/M256 paired ratios were
0.987678/1.000261/0.998414, supplying no useful gain by itself.

The compute source must still retain two GS2 register-A buffers (16 words) and
up to16 accumulator words atFT32, or8 atFT16. Codes, four reused scale pairs,
descriptors, pending work and address state add pressure. GATHB8 has32 value
words plus eight64-bit destination pointers before control state. These are
source live-data inventories, **not compiler-independent lower bounds**: the
compiler may recompute, spill or change allocation. They show why subtracting
64 from96 is invalid; they do not prove any narrowed build must exceed56.

Further required changes would include:

- Increase fixed WGS_SLOTS160 to at least265 for grid264; slot ownership uses
  blockIdx or inverse owner. At265 slots the FP32 partial area is16.5625 MiB
  forFT32 or8.28125 MiB forFT16, versus parent20 MiB. Counter areas grow5120→8480B.
  Existing raw allocation might have enough capacity, but offsets, C++/Python
  constants and zeroing must agree. The fixed136×16 doneAs tail is insufficient
  when FT16 JMAX reaches168 atMMAX256.
- Decouple shape-sized metadata from global scratch layout, or update every
  dependent offset/guard consistently. MMAX currently controls p.ybuf, p.xA,
  p.xB and allocation bounds; changing it is not merely shrinking a shared struct.
- Correct FT16 specialization guards: current `FT_T>8` still instantiates
  NTT3/4 although WGS_NT becomes2. Generalize ring barrier indices/depth, handle
  zero jobs before stage lookup, and prove split counters/readiness and resets.
- Preserve BF16/FP32 math, all routes and ordered finalizer visibility/PDL. The
  existing finalizer consumes shared routing metadata, which must be persisted
  or recomputed in timed work once moved outside MAIN.

The independent critic confirmed the register evidence is inconclusive and the
scratch/FT16 hazards are real. Stronger concurrency might help, but the proposed
large-load version adds11–85% weight requests in saved histograms while relying
on an unproved register reduction. That is insufficient evidence for another
implementation or official check. This feasibility audit is closed.
