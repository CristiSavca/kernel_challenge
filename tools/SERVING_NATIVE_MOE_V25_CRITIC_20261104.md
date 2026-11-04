# V25 independent feasibility critique — November 4

**Final verdict: GO for one official private component check of frozen source `3a8c973b737dd074bcf5115f7418f5f0b34139fd4f9f637410ad1765ccbe050d`. No public promotion, Modal allocation or full-serving deployment approval.**
The concrete author proposal is one KPERM3/MULDEC WGS weight format for every
token count, a scale-LUT path when PRMT is inexact, explicit zero-job handling,
and serving-compatible arithmetic. Root narrowed the first candidate to emit
BF16 `[T,8,H]` expert outputs and use the **unchanged original stock finalizer**
(`9b816e97…`), not V22 or native normalization. This replaces alignment, two
GEMMs and activation while isolating norm/combination compatibility. This is materially different from the
closed compact/WAIT2/SS-A experiments and from V24's ID mapping. It attempts
to reuse the native expert computation, not add another micro-optimization.
The direct eight-copy transplant remains rejected.

The intended benefit is a faster complete expert stage than stock Marlin,
including preparation-independent routing, two GEMMs, SwiGLU, per-expert norm
and route summation. Native247888's prior score is useful motivation, not a
forecast: its arithmetic boundaries, small-token implementations and tested
route population differ. No defensible speed upper bound or whole-service
gain estimate follows from that score. No new implementation has been
measured. The public native/decoder incumbents must remain unchanged.

The initial review found four material blockers for a deployment claim (the isolated component addresses the source issues below; full deployment still needs separate evidence):

1. **Match Marlin's actual arithmetic domain, not only its output dtype.**
   Preparation collapses gate/up global scales to the gate column, casts the
   global scale to BF16, and applies a power-of-two conversion. Marlin's
   `write_result` casts each FP32 accumulator to BF16 **before** BF16 multiply
   by the processed global scale. Merely casting `acc * global` is different.
   Stock gate/up outputs then enter SwiGLU as BF16; its intermediate is BF16;
   down output is BF16 before the effective BF16 norm weight is applied.
   Do not replace that weight with native `1+w`, assume that a gain cancels
   through RMSNorm, or silently use a different epsilon/norm form. RMSNorm's
   epsilon and finite-precision boundaries invalidate exact gain cancellation.
2. **One additional full bank is still expensive.** With 32 experts and three
   4096² matrices, code plus FP8 block scales occupies 905,969,664 bytes,
   or 0.84375 GiB per layer. Retaining both Marlin and one native bank costs
   52.3125 GiB extra over the source-default 62 sparse layers, before every
   other model allocation. This is an illustrative configuration, not an
   inspected loaded checkpoint. A production plan needs one-copy replacement
   with complete runtime eligibility, or demonstrated whole-model headroom.
   Deleting Marlin weights while retaining runtime fallbacks that read them
   is not a valid design. Native canonical inputs are normally deleted by
   Marlin preparation, so the precise preparation hook and ownership must be
   specified before editing.
3. **Per-call allocation does not automatically bound captured memory.**
   Parent `_make_scratch` allocates 164,668,960 bytes per layout (157.04 MiB).
   Capturing separate allocations for 62 layers could retain 9.508 GiB even
   before multiple graphs. Fresh `zeros` calls also introduce timed GPU work.
   A geometry-based shared workspace has an existing SGLang precedent in
   `flashinfer_megamoe._ensure_shared_workspace`, but reuse requires an actual
   lifetime/stream/capture contract. Outputs must never alias that workspace.
4. **All-remote EP ranks are ordinary serving inputs.** The native benchmark
   uses the busiest rank; its evidence does not establish zero global jobs.
   `wgs_stage_at` reads `S.jC[0]` when `nj == 0` unless separately guarded.
   A zero-job path must set empty ranges before any such read or division,
   preserve required producer/consumer joins and counter cleanup, and produce
   zero output without a host read of route values. Actual expert maps and
   all ranks must be tested; subtracting a hardcoded local offset is not a
   general arbitrary-map adapter.

An exact-domain adapter is a plausible direction, **not a proof**. Parent
native decoding forms `q * 2^-126` times `sf * 2^64`; Marlin uses the scale
domain `sf * 2^7` for positive-normal FP8 scales. The precision author's actual
preprocessing model finds that bytes1..7 differ from that simple formula;
the exact Marlin-derived LUT is required. A serving build could reconstruct exactly the Marlin BF16
operands, then preserve its pre-global BF16 rounding and BF16 product. Scaling
the completed native accumulator by `2^-57` is not a universal substitute:
per-weight subnormal rounding can differ. Require actual-source finite-code /
scale equivalence, boundary counterexamples, and a GPU comparison that exposes
underflow/overflow and reduction-order effects. WGMMA and Marlin's MMA schedule
may still differ numerically; bit-identical full GEMMs are not claimed.
Marlin GEMM also lacks the parent's global `--use_fast_math`. Moving into the
tiny Marlin operand domain makes FP32 subnormal intermediates relevant: do not
inherit global FTZ unexamined. Preserve stock activation's explicit fast SiLU
intrinsics separately from the GEMM/fixup compiler policy.

Original stock small-T finalization zeros invalid route values with `where`
before arithmetic; its large-T kernel skips invalid-route loads. Thus remote
output slots may remain unwritten only if local IDs are complete and correct.
Every live output must land in the original route slot through `S.ent`, not
expert-sorted order. Test chunk offsets, duplicate routes and poisoned remote
slots. The route-output tensor itself is64KiB/token at H4096/K8. The proposed
core workspace trimming is not yet an audited candidate allocation.

WGS at M1/4/8/16 is a new dispatch, not the validated native skinny path. The
new LUT path also changes WGS decoding and resources. Both need actual
compiler/admission evidence and measured complete-chain behavior. Do not
inherit timing, registers or correctness from247888 just because much of the
source is retained. Keep the M384 chunk boundary, ragged tails, zero tokens,
highly concentrated routes and full prefill size in the supported-domain plan.

Separate scratch is insufficient to prove arbitrary concurrent-grid safety.
The native kernel uses a global spin barrier and launches one CTA per SM.
Independent streams can invalidate its full-grid progress assumption unless
the launch mechanism guarantees progress for the full grid. Root identified
a supported remedy: `cudaLaunchCooperativeKernel`. NVIDIA documents atomic
grid launch, stream-capture support, and concurrency only when cooperative
kernels fit together. Require the actual device's cooperative-launch support,
actual compiled-pointer occupancy at exact block/shared-memory dimensions,
grid size within that capacity, current-stream launch and checked return code.
No ordinary-launch fallback is acceptable for this path. Remove the inactive
PDL instructions/attribute path consistently. This addresses admission, not
shared-scratch races, numerical correctness or performance. Sources:
[NVIDIA cooperative launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#when-to-use-cudalaunchcooperativekernel),
[CUDA 11 capture support](https://developer.nvidia.com/blog/cuda-11-features-revealed/),
[CUDA 11.2 cooperative concurrency](https://developer.nvidia.com/blog/?p=22770).
The cooperative host is now source-reviewed below; actual CUDA admission remains part of the official check.

Before any GPU proposal, freeze a complete replacement/eligibility and memory
plan, the adapter's arithmetic contract, and one actual source candidate.
Use exact stock serving Marlin with the same prepared values and the same
inputs as the eventual deployment comparator. Native247888 is a separate
diagnostic reference, not interchangeable stock math. Root's fixed initial
official-component screen is **six-case geometric mean speedup >=1.15 against
the same-run supplied stock baseline, no individual candidate/stock latency
ratio >1.02, clean five tests/six benchmarks/six profiles, and observed WGS plus
original-finalizer activation on every benchmark shape**. All cases change;
there is no unchanged control family. Include every added mapping, copy and
initialization in the call. A result behind247888 does not justify public
kernel promotion. Numeric disagreement, missing activation,
invalid admission, unsupported reachable metadata, excess retained memory or
unsafe workspace lifetime kills the corresponding deployment claim. A failed
performance screen closes that fixed candidate; no automatic policy sweep.

The supplied official baseline is not identical serving scheduling. It divides
the block-size heuristic by local count32; serving divides by global256. At
M32/128/256 this selects official blocks16/48/64 versus serving8/8/16. Official
alignment uses256 buckets; the existing serving alignment patch uses32 local
buckets. Therefore even a useful official score qualifies only this isolated
component. It cannot establish matched current-serving latency, TTFT, memory
feasibility or eight-GPU throughput. The later true-serving comparison remains
mandatory.

The saved V24 accounting left $1.29 working allowance plus separate $5
headroom. The existing 600-second H200 reservation is $1.78 and does not fit
that dated allowance. A 300-second estimate fits narrowly but may not cover a
new extension build and packing. Do not weaken the guard or assume a larger
run. Root subsequently reported a higher4.25390221 observation and1.25609779
working allowance; no Modal allocation is planned. A proposed official private
component check does not use this local allowance. Source feasibility is useful before the later serving track opens; it
does not improve the imminent kernel leaderboard by itself. No provider,
GPU, official calls or candidate edits were made during this review.

The subsequent saved-floor edit is independently verified: reversing exactly
the three path/hash/amount literals reconstructs runner `d5bff016…` from current
`b781140b…`. The actual local helper reads receipt `590e57bb…` and returns
4.25390221. Guard `9f70c500…` and the complete17-row journal are byte-identical
to HEAD, with all $32.74 of reservations retained and unchanged A/E corrections.
This only increases the debit; no provider call or allocation occurred here.

## Actual isolated-component source review

Reviewed draft `3a8c973b737dd074bcf5115f7418f5f0b34139fd4f9f637410ad1765ccbe050d` now implements the cooperative host, one WGS bank and BF16 route-output boundary. The frozen actual-source CPU evidence has now passed independently; official GPU compilation/correctness remains untested. This is not a deployment approval.

The host checks both BIG kernel pointers at the exact576-thread/shared-memory geometry, requires cooperative support and at least one resident block, bounds the grid below160 fixup slots, and checks the selected pointer again before a current-stream cooperative launch. The launch has no ordinary fallback or PDL mode. Empty jobs set all four ranges to zero before cost lookup, retain the producer/consumer joins and grid barrier, and still write the complete local-ID mask. The unchanged route scatter preserves original token/slot positions through `S.ent`.

The newly identified global FP32 reduction mismatch is repaired at its actual active site: four integer-CAS loops preserve the original four scalar addresses, with explicit `add.rn.f32` and bitwise retry comparisons. This avoids the documented FP32 global-red subnormal flush without treating NaN as an equality-based spin condition. The parent WG barrier, release counter, acquire wait and final buffer-zeroing protocol remain unchanged. Scalar additions can still occur in a different order from Marlin, and no GPU memory-order or tensor-core precision proof is claimed.

The activation has a separate FTZ policy from the GEMM/fixup: explicit `mul.rn.ftz.f32` applies to the final SiLU/up product, while global fast-math stays absent. This fixes the concrete counterexample where the ordinary product preserved a subnormal that stock fast-math discarded. Division/exp intrinsic lowering and WGMMA accumulation still require GPU checks. The component imports exactly the same qualified finalizer as the official stock baseline; that establishes a shared same-run dependency, not a hash proof of the remotely installed source or validation of the dirty local serving checkout.

Current workspace allocation is108,033,568 bytes (103.028839MiB):4,128B counters,20,985,344B fixup storage,61,865,984B gathered activations,25,165,824B BF16 expert outputs and12,288B local IDs. Root explicitly chose reusable state storage for serial official calls/graph replays. Overlapping use of one workspace remains unsupported; full serving needs a shared resource with demonstrated lifetime and capture ownership. The code retains no source-weight reference in its packed bank. The bank alone is905,969,664 bytes; one-bank preparation is not a completed full-model weight-replacement plan.

For prefill beyond384 tokens, the final short chunk is padded to32 only if the original whole call selected the large stock finalizer. Padding IDs are-1 and scores0; added copies/fills remain timed. No real routes are omitted. Official cases do not exercise that chunk boundary, arbitrary EP maps, all-remote ranks or generic simultaneous streams, so a passing component check cannot certify those deployment cases.

The offline result analyzer now requires both the new core and stock-finalizer evidence on all six cases. Large shapes require the stock helper symbol; small shapes require the source-supported mask/mean/slot-sum families, using retained H200 profile evidence rather than a fixed kernel count. All six independent gate fixtures pass locally. The numerical/performance thresholds are unchanged: clean5/6/6 records, at least1.15 geometric-mean same-run stock speedup and no case above1.02 normalized latency. No automatic promotion or retry follows.


Final verifier `7711f3d446a17c4d9f473f48847f8b62a54bd5d69c3e300ddc6e7910fd875b5e` was independently executed through `run()` without writing the author's evidence. Its result exactly matches `cpu_result.json` SHA `14ace628a8a89c5be437f8877b74b9d86b058187baee098da13dd4c39bdb92a1`. It reconstructs the complete parent CUDA source, reproduces the builder, binds six unchanged packing helpers and checks both PRMT preprocessings for dead old-buffer accesses. Actual slab packing matches full packing for one gate/up expert and one down expert, with the existing independent inverse asserting50,331,648 code and3,145,728 scale coordinates. The32-expert retained-bank size is arithmetic, not32 separately executed expert tests.

Arithmetic checks cover127 nonnegative finite scale bytes,119 normal PRMT entries and16,129 scale pairs, plus explicit two-round, subnormal-add and final-product FTZ counterexamples. Output ownership checks enumerate8,519,680 formula coordinates over64 job sizes; rank mapping has2,056 formula cases. Only the actual zero-job range entry is compiled/executed natively across132 CTA indices; these are not complete GPU routing/barrier executions. Six host policy boundary cases are formula/source checks, not mocked CUDA API execution. CAS concurrency, WGMMA accumulation, exp/div lowering, capture and resource allocation remain official-GPU or later generic-validation questions. These limits are explicit, and no additional test framework is a prerequisite for the single bounded check.

The author froze the exact source after the repairs. The independent critic made no candidate, helper, runner, guard or journal edits and no provider/GPU/official calls. A failed numerical, activation or fixed-speed screen closes this attempt. Even a pass does not remove the separate single-bank loader replacement, full-model memory, generic-rank/empty-input and stream/graph ownership gates.


Final evidence-label refresh: verifier `cb9952efbc11a07e2dbb266a963731413795e2f98b855d0d541d4f712fabd697` and report `aa266717bfb13490f2e4db2bf3f717e8c9b752b23f7dd6763f8f9cc3c7ef94e3` replace the inverse routine's unused `None` return field with an explicit `inverse_coordinate_assertions: PASS` label. Reversing exactly those two verifier statements reconstructs independently executed verifier7711f3d4, and reversing those two report labels reconstructs14ace628 exactly. Actual inverse assertions and candidate bytes are unchanged. Final GO therefore applies to the current evidence pins; no second independent full rerun was needed for this label-only delta.
