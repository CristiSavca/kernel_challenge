# Native FP8 QK for full decode, V5

Status: isolated, CPU layout checks passed; CUDA compilation, numerical
correctness and timing are **unverified**. No GPU allocation, official check,
submission or active-source change was made for this experiment.

Candidate: `native_q1_fp8_qk_v5.py`, SHA-256
`2eb08ca476d2f86cb43d2b8b1d2d630e3aa7489359acaf41cdf646ec49cd00fd`.
Parent: scored attention `8f9e5971f95841a0`, SHA-256
`87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c`.
The parent remains active. Preserve the upstream MIT/Apache notices.

## Hypothesis and gate

Full-attention q_len=1 converts FP8 K fragments to FP16 in every tile and
performs 16 QK m16n8k16 instructions per consumer warp, using an extra
split-sum to form the eight real query rows. A native E4M3 QK path uses
eight m16n8k32 instructions and eliminates 32 FP8-to-FP16 conversion
invocations per warp per tile. It leaves the 16 FP16 PV instructions,
FP32 online softmax, K/V loads, cache stores and merge unchanged.

Falsifiable target: at least 15% less full-attention latency at B16 and B32
decode. This is a stretch, particularly at B32. More shared load instructions
are needed (16 scalar 32-bit loads versus four vector 128-bit loads per
warp; equal bytes), and the new path scales the QK results afterward.
Instruction counts alone do not establish exposed latency or throughput.

Kill immediately if CUDA fragment mapping, in-place staging, scale handling
or numerical tests fail. After one bounded paired comparison with the exact
incumbent, reject the 15% thesis if either B16 or B32 misses 15%. A smaller
result is only a promotion lead if it gives a repeatable >=3% improvement
on the affected attention cases, no >3% regression on another affected
case, and passes every official correctness case. Do not conduct a parameter
sweep in response to a null result. Do not promote a locally timed result
without an official private check of the exact source.

## Raw evidence and physical ceiling

Evidence is the `8f9e5971f95841a0` JSON/TXT in
`experiments/candidates/muse_attention_decode/profiles/`, and decoder
`4b216098993e4112` in the matching decoder profile directory. Values below
use raw replay median minus the recorded replay overhead, never the
paired-ratio-normalized `median_ns`.

| Full case | Raw attention us | Attention profiler us | Decoder attention profiler us |
| --- | ---: | ---: | ---: |
| Decode B16 | 86.384 | 85.6 | 90.0 |
| Decode B32 | 165.776 | 167.2 | 169.1 |
| Verify B16 | 100.304 | 100.3 | 105.2 |
| Verify B32 | 182.976 | 186.0 | 186.1 |

The active flat path already loads each K and V row once per KV pair,
sharing them across all eight query heads and all query tokens. With TP8,
each cached position requires 128 K bytes, 128 V bytes and a four-byte page
index. Scored lengths lie in [74K,86K]; the actual sums were not saved.
No duplicate cache stream was found to remove. Cache pages are randomly
permuted; assuming contiguous physical page order would be incorrect.

At 4.8 TB/s, the conditional floor for just these 260 bytes per key is:

| Batch | Minimum lengths us | Nominal 80K lengths us | Maximum lengths us |
| --- | ---: | ---: | ---: |
| 16 | 64.133 | 69.333 | 74.533 |
| 32 | 128.267 | 138.667 | 149.067 |

These exclude QKV, outputs, merge workspace, duplicate boundary reads and
overhead. They are optimistic conditional bounds, not measured traffic.
At nominal lengths, the maximum theoretical reduction is 19.74%/16.35%
for decode B16/B32 and 30.88%/24.22% for verify B16/B32. Using minimum
lengths makes the decode ceilings 25.76%/22.63%; actual lengths are needed
for a tighter claim. A 15% B32 decode reduction means 140.910 us and would
require 4.724 TB/s for these bytes alone, or 98.4% of the specified peak.

This candidate affects the two batched full-decode loads, not verification.
Even meeting 15% would contribute roughly one percent whole-decoder gain
under the existing weighted profile shares. It cannot close the decoder's
14.8% gap alone. Verification remains the more generous attention target;
its QK and PV paths already use FP8 WGMMA, so this change does not apply.

## Implementation and correctness boundaries

- The new path is compile-time `L == 1 && FLAT && OKC_SM90 && OKC_Q1FP8QK`.
  Default dispatch therefore changes full decode with multiple requests.
  The general fallback, MAXSEG=64 protection and all launch plans are unchanged.
- Queries remain unit RMS, rounded through BF16 and then permitted E4M3.
  `sm_scale * qk_scale_factor * log2(e) * k_scale` is applied after QK.
  This avoids scaling queries into E4M3 subnormals for the 1/64 cache scale.
  The original cache stores, runtime k/v scales and FP16 PV are unchanged.
- Each normalization thread writes eight packed FP8 bytes into only the
  first half of its own original 16-byte BF16 chunk. Compacting the whole
  row in-place would race other threads that still read raw queries.
  Existing consumer synchronization precedes the new fragment reads.
- The two FP8 A operands cover dimensions 0..31 and 32..63. Their unused
  query rows are zero. B uses the existing key swizzle. The resulting lower
  eight rows retain the original score layout, so all causal/window masks,
  online softmax, PV, per-split reduction and final output code stay intact.
- Every attended key remains traversed; no input-derived work is moved
  outside `kernel_fn`. No precision below the permitted E4M3 query is added.

The independent CPU verifier `native_q1_fp8_qk_v5_verify.py` reconstructs the
PTX A/B matrices from all 32 lanes and checks all 1,024 QK results against
direct dot products, all 8,192 swizzled key bytes, and disjoint ownership
of the 2,048 raw query staging bytes. It passed, as did Python AST parsing
and diff whitespace checks. This does not validate ptxas or GPU execution.

Fragment geometry follows NVIDIA's
[PTX m16n8k32 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832).

## Critic response and next experiment

The independent critic agreed that a small isolated candidate is worth
considering, while rejecting any instruction-count-to-speed inference.
Halving QK's exposed latency only yields 15% total improvement if that
portion previously occupied at least 30%; overlap and conversion removal
make a simple consumer-fraction gate inadequate. The candidate instead
needs a full-kernel paired result and numerical tests.

Before timing: compile and inspect register/spill changes; exercise full
B2/B4/B16/B32, page sizes 1/64, scale 1/1/64, soft/coherent distributions,
long contexts, and fresh repeated calls. Include untouched sliding/verify
as sentinels. Capture identical graphs with cold L2 and paired order;
record actual `seq_lens.sum()` outside timing for the physical bound.
Compare the parent loaded with its original module name against this
candidate's unique module name. A later decoder test must embed the
candidate's CUDA string in an isolated decoder, preserving decoder PDL
compile flags; active attention changes do not automatically change decoder.

If V5 fails, the next useful attention step is a separate diagnostic of
the q_len=8 flat path, not another guessed stage count. Existing
`OKC_PROF=1` must be combined with `OKC_PROFPV8=1`, otherwise it silently
disables FP8 PV. Per-warp `clock64` phase deltas are local elapsed-cycle
measurements; do not compare absolute clocks across SMs or sum all warp
waits into a removable wall-time claim. Use globaltimer completion marks
for the critical tail, and preserve the exact production path.

Storage: one ~250 KB candidate, one small verifier and this note; physical
free space was 39 GiB before work. No removal was necessary or performed.
