# V89 paired expert weight dequantization

V89 removes 349 of 676 compiled instruction sites in the sparse W13 gate/up kernel’s inner reduction loop. The change arose from disassembling V87’s Hopper cubin, which repeatedly converted each FP4 magnitude to FP32, multiplied it by an FP32 scale, and converted the product to BF16. V89 constructs two BF16 tensor-core operands at a time in registers. The FP32 accumulators, iteration order, activation rounding and routed rows follow V87. The dispatch condition admits up to one route slot per global expert on average. With eight proposed nodes and eight routed experts per node, four concurrent requests can exercise this path. [Parent V87](../v87_fused_draft_head_skinny_experts/README.md)

## Exact conversion

The signed E2M1 magnitude codes 0 through 7 represent 0, ½, 1, 1½, 2, 3, 4 and 6. Placing the three magnitude bits at BF16 bit positions 6 through 8 and the original sign at position 15 represents each code times 2⁻¹²⁶. A packed BF16 multiply by 2¹²⁶ then gives the two exact signed values, including the two zero bit patterns. The smallest encoding before normalization is a nonzero BF16 subnormal. Explicit round-to-nearest BF16 PTX multiplication preserves it. [NVIDIA PTX half-precision multiplication](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-mul)

The second paired multiply applies the incumbent's zero-or-normal E4M3 block scale. E2M1 supplies at most two significand bits and E4M3 at most four, so their product fits exactly within BF16’s eight significand bits. The smallest nonzero scaled magnitude is 2⁻⁷, and the largest is 2,688. Positive prepared scales preserve the FP4 sign; multiplying a negative value by a zero scale gives negative zero. V89 uses the same packed checkpoint weights, scale bytes and workspace as V87.

## Compiled evidence

| Static SASS metric | V87 | V89 |
| --- | ---: | ---: |
| Instructions in inner K loop | 676 | 327 |
| Tensor-core MMA sites in loop | 8 | 8 |
| Global load sites in loop | 6 | 6 |
| Instructions in entire kernel, including padding | 960 | 608 |
| Scalar integer-to-FP32 conversions in kernel | 64 | 0 |
| Packed BF16 multiply sites in kernel | 0 | 64 |
| Registers per thread | 86 | 77 |
| Static shared memory | 16,384 B | 16,384 B |
| Spill loads / stores | 0 / 0 | 0 / 0 |

The 51.6% reduction counts static instructions between the innermost loop’s backward branch target and the branch itself. This kernel still streams every live expert’s W13 weights and issues the same MMA operations. Its latency depends on memory bandwidth, instruction throughput and occupancy. Whole-model speed, agreement, peak memory and TTFT require H200 execution. [SASS counts](sass.json), [compiled Hopper evidence](sm90_compile.log), [V87 disassembly](v87.sass.gz), [V89 disassembly](v89.sass.gz)

## Validation

The full [official export](patch.diff) changes 32 allowed files against the pinned SGLang baseline and spans 186,854 bytes. Its SHA-256 is `e20ea10ed16c9b4dec6e0d4c834ef44c086f5cbf468ee6574a5a258800da6e18`. The [one-file delta](delta.diff) applies to frozen V87. All 28 modified Python sources compile. Pinned-base forward and reverse application and official-export byte equality passed. [Export receipt](validation.json)

The CPU simulation checked 130,048 packed operands across every legal original scale byte, both signed nibbles, and four fragment selectors with unrelated bits varied. Sixteen complete simulated CTAs exercised 38,912 live BF16 output values and 30,080 inactive-prefix values. An independent comparison covered 16,516,096 packed outputs, including 2,974,848 negative-zero scalar results. [CPU harness](cpu_check.py), [CPU results](cpu_result.json), [independent review](independent_review.md)

Reproduce the instruction comparison with `python3 compare_sass.py v87.sass.gz v89.sass.gz` and the executable CPU check with `python3 cpu_check.py /path/to/patched/sglang` in a NumPy environment. The published cubin hash is `60df76855761fe75e34e146ceb698a45f5ba45047098a20579ef860a16642042`. V89 uses V87’s existing flags and check load.
