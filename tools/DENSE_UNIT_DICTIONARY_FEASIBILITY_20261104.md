# Exact unit dictionaries: retain as a broad lead, no GPU campaign yet

2026-11-04. Read-only strategy and bounded CPU feasibility on current decoder
`97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2`.
No candidate, provider call, official call, allocation, or change to the incumbent.
The direction is to remove mandatory correction-warp participation by decoding
original BF16 weights exactly from a local dictionary, with a complete fallback.
**The representation is feasible; a performance case is not yet established.**

The current MoE evidence makes another broad traffic-increasing expert rewrite
less attractive. Exact KPERM NCU transfers near nominal packed weights at about
77% reported DRAM throughput; current BIG1 SASS has the expected packed decode,
folded scale broadcasts, actual wait1 instructions and no C751x in the complete
build log. These are separate standalone diagnostics, not a decoder bandwidth
ceiling or proof of attained WGMMA overlap. They do not rule out the separately
owned instruction-footprint experiment.

The exact public decoder profile, SHA
`35ad7364409d49c2ded613234beea1572adc691c33f3c154a53b3dff313a2414`, gives:

| Weighted load | MoE share | Dense share | Attention share |
| --- | ---: | ---: | ---: |
| decode 1 | 34.35% | 52.78% | 8.10% |
| decode 16 | 43.99% | 33.50% | 17.94% |
| decode 32 | 39.71% | 29.10% | 25.22% |
| verify 1 | 46.97% | 41.38% | 8.23% |
| verify 16 | 53.26% | 28.29% | 13.24% |
| verify 32 | 54.41% | 24.09% | 15.77% |

Using normalized case times, 46/16 layer weights and six-load geometric
aggregation, a hypothetical 20% reduction of all dense time gives a 6.99%
decoder latency reduction; 20% of all MoE gives 9.10%; both gives 16.07%.
An 18.1% score gap corresponds to 15.33% less normalized latency. Dense alone
would require approximately 43.66% reduction, or MoE alone 33.65%, in this
model. These are conditional scope scenarios, not forecasts or physical bounds:
profiles contain PDL waits and overlap. This motivates broad work without
pretending a single micro-optimization can close the gap.

## Concrete representation and CPU evidence

For each original 8-row by 64-K unit, retain its 512 raw low bytes. Replace the
matrix-wide three-bit exponent offset with an index into up to eight distinct
`e71 = (BF16_bits >> 8) & 127` values. The sign remains the fourth code bit.
Use consecutive sign/index nibbles so four lookup selectors occupy 16 bits.
A 16-byte unit header contains eight dictionary bytes, a raw-sidecar unit index,
and a mode. If a unit needs more than eight entries, its consumer reads the exact
1024-byte BF16 unit from a sidecar. No expected distribution can waive fallback.

The proposed fixed primary remains 768 bytes even for a fallback unit. Append
the 16-byte headers as a contiguous stage tail, so the original aligned primary
unit locations can remain intact and one main bulk transfer can cover primary
plus headers. This requires a real stage repack and different transfer/stride
calculations; it is not a change to a typedef. The sidecar adds 1024 bytes per
overflow, **not** 256 bytes, because the primary is still transferred.

`audit_dense_unit_dictionary_feasibility.py` streams three fixed Python Gaussian
seeds, each 4096 units by 512 weights. It rounds the Gaussian to FP32, multiplies
by FP32 `1/sqrt(8192)`, rounds to FP32, then BF16 round-to-nearest-even. This is a
distribution proxy for QKV/pooler dimensions, **not Torch/CUDA RNG output, actual
judged weights, or proof of their eligibility**. No bulk weights are saved.

| Proxy seed | Units requiring raw fallback | Fraction | Requested bytes / original 768 |
| --- | ---: | ---: | ---: |
| 230401 | 28 / 4096 | 0.6836% | 1.029948 |
| 230402 | 24 / 4096 | 0.5859% | 1.028646 |
| 230403 | 30 / 4096 | 0.7324% | 1.030599 |

Combined: 82 of 12,288 units, or 0.6673%, overflow. Most have nine distinct
values; one has ten. The same CPU model round-trips all 65,536 BF16 bit patterns
through eligible eight-entry units, all 65,536 four-code sign/index selectors,
and adversarial nine-/128-entry raw fallbacks. Signed zeros, infinities and NaN
payloads are represented as bits. This does not establish CUDA conversion or
tensor-math behavior. The generic PRMT sign-replication model follows the
[NVIDIA PTX PRMT specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt).

For each four codes, lookup the dictionary using `selector & 0x7777`; a second
PRMT from `0x01010101` supplies the sign-mask complement. Dictionary bytes have
bit7 set, so XOR with that second result shifted by seven reconstructs the exact
signed high bytes. Four existing byte-interleave PRMTs finish eight BF16 values.
Source operation accounting is about 15 integer operations per eight values
versus the parent's 12, plus dictionary/header loads and the mode branch.
This is not an instruction count from compiled SASS. Two live dictionary words
replace the parent's single replicated base and can increase register pressure.

## Costs that prevent an implementation GO

Removing correction warps eliminates their stage waits/handbacks, correction
planes, table scans and final correction sums. It does not automatically add a
resident CTA. The independent source arithmetic for unchanged geometries finds
no new two-CTA shared-memory fit: for QKV TOK8/TOK16/TOK32, header/ring growth
minus correction planes is approximately -256/-1280/-8704 bytes; fused pooler
TOK8/TOK16 is -2304/-3328 bytes. Exact table/header alignment and actual compiler
attributes still require validation.

The proxy incurs approximately 3% more requested weight bytes and 25% more
source decode operations. Fallback is warp-uniform, but its consumer global
load extends the time before ring handback. With the observed unit fraction,
an independent-unit model puts a fallback in about 14.8% of 24-unit stages or
19.3% of 32-unit stages. Those are model probabilities, not measured cache
misses or observed stage delays. A whole-matrix eligibility/cost fallback must
avoid the worst case of 1808 requested bytes per unit versus 1024 raw bytes.

The original BF16 bits are reconstructed exactly, but replacing approximate
matrix-window values plus a separate FP32 correction sum changes accumulation
association. Official numerical checks remain necessary. Preserve UV, fused
FP32-router hi/lo, BND, PDL waits, and actual input computation in timed work.

This differs from rejected global-activation corrections and stage directories:
it has no correction path. It also differs from the prior mixed 12/16-bit tile
idea: a dictionary can represent noncontiguous exponents, and a fixed compressed
primary avoids worst-case full-BF16 shared slots. It pays sidecar global traffic
instead of variable-size full-unit TMA stages. These differences are material,
but are not evidence of a speedup.

## Next decision and minimum evidence

Keep this as the broad dense hypothesis pending the proposed exact-current
QKV16 and fused-pooler/router16 diagnostic. The question is whether mandatory
correction/ring participation is exposed enough to justify the added decode and
fallback work. DRAM, issue and shared/load-queue counters can reject an obviously
bandwidth- or decode-limited rationale. Aggregate barrier stalls alone cannot
attribute waiting to correction warps; that would require source/role attribution
or a separately reviewed targeted timing comparison. Do not reopen a GPU campaign
solely from the proxy histogram or fewer threads.

Before a candidate: obtain actual prepared-unit cardinalities and fallback stage
distribution, prove exact stream/header/sidecar bounds and state ownership, and
compile the fixed layout to check its real resources. A 20% reduction of only
the currently visible SK2 rows models 4.74% whole-decoder latency reduction;
30% models 7.16%. UB2 and large-T dense paths are outside that scope. This is
enough scope to matter if measured, but no such reduction is currently supported.

Evidence pins: script SHA256
`beb6d9845e912c194b14dda7ffd53366ce2daa1d99ad6241c77561a3081a0f55`;
`experiments/iterations/20261104-dense-unit-dictionary-feasibility.json` SHA256
`e4d44d3394ae1e2e9572bb7cb99ab73174103e4d96470a2f7aac43a72db09403`.
The script and report are feasibility artifacts, not a submission.
