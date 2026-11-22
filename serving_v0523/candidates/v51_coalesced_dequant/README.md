# V51 coalesced dense expert dequantization

V51 changes the NVFP4 dequantization kernel from V44. It uses eight lanes per
16-column K tile to issue a complete 32-byte BF16 sector in one store. Each
lane visits the two output rows encoded by one 32-bit Marlin word together.
The lookup table, scales, rounding, weights, GEMMs, routing and graph boundary
use V44's implementation.

- Base SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`
- Patch SHA-256 `cade9e982b30ecf6f0ac5d20bf850a696c526b17811a8437d849c5eea2cee4fc`
- Patch size 20,154 bytes, four allowed source files
- Serving flag `swa-full-tokens-ratio=0.25`

The H200 component run measured 42.2 µs for gate/up dequantization and 24.4 µs
for down dequantization per expert. At 32 active experts and 62 layers, that
is about 132 ms of kernel time per full chunk. A hypothetical 15–30% reduction
would save 20–40 ms per full chunk, with a serving score estimate of roughly
+0.8–1.8% over V44. The smaller K span doubles the CTA count and repeats table
setup, so performance can also regress. GPU compilation and performance
remain unmeasured.

Validated application with whitespace checking and reverse application to an
isolated index at the pinned base. An exhaustive CPU mapping check matched
the packed word, nibble shift and scale byte for all 1,024 entries in a
Marlin tile. A separate launch coverage check accounted for every output
word at nine K extents, including partial CTAs. Other patch hunks are byte
identical to V44.

The isolated 1,510,195-byte index was removed after verifying it had no open
handles. The nested working tree was not used for the candidate.
