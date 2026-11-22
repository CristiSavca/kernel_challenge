# V148 expert reduction and NVFP4 arithmetic

V147 remains the candidate for comparison. Feature-sharded unpool raises the communication requirement and changes the GEMM reduction tree. Five exact alternatives to V147's NVFP4 operand reconstruction also failed to produce a material across-projection improvement in matched Hopper machine code. This investigation used an offline compiler and CPU arithmetic fixtures. It produced no official submission or hardware timing. [Compiler identities](validation.json) · [Ablation sources relative to the frozen V147 headers](ablations.tar.gz)

## Feature-sharded unpool

One native request has eight target verification rows. V147 reduces each rank's local 4,096-channel BF16 expert output. On eligible Hopper multicast communicators, the unpooler reads an 8 MiB output shard of its BF16 weight, performs full-K dot products and gathers the 8,192 output channels, possibly at the following residual/norm boundary. For more than eight rows the pinned gather selector admits SM103 only. The ordinary H200 path therefore reads the 64 MiB matrix in each rank and needs only the first all-reduce. V142 already implements output sharding for that geometry. [Pinned gather selector](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/kernels/ops/communication/muse_spark_all_gather.py) · [V142 geometry](../v142_sharded_expert_projections/geometry.json)

| Verify rows | Current V147 logical ring send, KiB/rank/layer | V142 output-sharded | Feature-sharded with BF16 partials | Feature-sharded with FP32 output reduce-scatter + BF16 gather |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 224 | 224 | 280 | 392 |
| 32 | 448 | 896 | 1,120 | 1,568 |
| 128 | 1,792 | 3,584 | 4,480 | 6,272 |
| 256 | 3,584 | 7,168 | 8,960 | 12,544 |

These are ideal balanced ring sends. Native NVLink multicast uses its own protocol. Feature reduce-scatter first rounds the reduced inputs to BF16 on their owners, preserving the pre-GEMM boundary. Each rank then computes only one eighth of K. A BF16 partial-output all-reduce introduces another rounding. Keeping FP32 partials and gathering only after the output reduce-scatter requires **three** dependent collectives. It still changes the summation order within each output dot product. Synthetic BF16 inputs demonstrate a full dot product of 1 becoming 0 under both forms of partial reduction. For rows above eight, the feature-sharded weight saving is already available with V142's full-K output-sharded GEMM. [Arithmetic and rounding examples](collective.py) · [Counts](collective.json)

## Same-bank dequantization

The trial kept all original expert weight and scale banks, MMA ownership and accumulation order. One design decodes eight FP4 magnitudes per word through register byte permutations. It moves a factor of 2¹¹⁹ from the FP4 operand into the E4M3 scale, leaving one exact BF16 pair multiply. A second design uses bitwise operations on all eight nibbles with a 2¹¹⁸ factor. A third stages an 8 KiB table of the eight possible E4M3 mantissas combined with all signed FP4 pairs, then applies each scale's exponent with one packed multiply. Two smaller trials rearrange the parent's original two-multiply bit extraction. The source labels below identify each compiler receipt. [Changes and constants](ablations.tar.gz) · [PTX byte-permute semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt)

| Variant | Low W13 loop sites | Low W2 loop sites | Registers/thread W13 / W2 | Packed multiplies W13 / W2 |
| --- | ---: | ---: | --- | --- |
| V147, `base` | 327 | 189 | 78 / 56 | 64 / 32 |
| Register lookup, `test` | 343 | 194 | 80 / 56 | 32 / 16 |
| Packed bit arithmetic, `swar` | 429 | 230 | 80 / 56 | 32 / 16 |
| Shared lookup, `table` | 382 | 208 | 80 / 56 | 32 / 16 |
| High-nibble extraction, `marlin` | 331 | 189 | 80 / 56 | 64 / 32 |
| Multiply and mask, `imad` | 326 | 182 | 78 / 56 | 64 / 32 |

All variants compiled spill-free for SM90a under NVCC 12.9.86 and kept the same tensor-core instructions. The shared-table version also issues 32 W13 or 16 W2 shared loads per loop and one CTA barrier per active block. The multiply-and-mask case reduces the combined low-density loop count from 516 to 508 sites, or 1.55%, and changes a 16-row medium W13 loop from 330 to 332. These counts are static instruction sites. They provide no measured serving latency or throughput. [All hot loops and compiler resources](sass.json) · [Disassembly parser](compare.py)

Each compiled CPU fixture executes the original C++ extraction expressions. It substitutes portable, bit-exact implementations for the PTX packed multiply and byte permute. For each of the six forms it checked 16,516,096 packed pairs against independently decoded FP4 and E4M3 products, including 2,974,848 negative-zero lanes. All 256 raw scale encodings are accounted for. V147's unchanged preparation flushes raw bytes 0–7 to zero, accepts 8–126 and rejects 127–255 before dispatch. Run `python3 check.py` with a C++20 compiler to repeat from the frozen official patch. [CPU comparison](check.py) · [Numerical results](cpu.json)
