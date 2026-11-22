# V208 residual tiles for long prompt experts

V208 reduces padded tensor-core arithmetic on the final tile of each local expert in both grouped prompt projections. The official patch is 300,360 bytes with SHA-256 `726ee2c1ab1e6e09a7cbe37a8c7aef14cafc33f979c50f7fddbb9766c213f82d`. It applies to the published pin, reconstructs in both directions against V206, and retains its launch flags. [Official patch](patch.diff) · [Four-file delta](delta.diff) · [Rules, flags and tree checks](validation.json)

The persistent Humming scheduler already knows the expert’s end row and current tile’s start row. Each consumer computes their difference before it enters the K loop. At up to 32 or 96 live rows it uses a narrower Hopper tensor instruction with the same register-source packed-weight fragment and the same K16 activation descriptor. Otherwise it executes the existing 176-row instruction. Every CTA keeps the incumbent 176-row storage pitch, its original output bands, weight tile requests and shared-stage barrier protocol. Humming’s own generator emits the BF16/FP32 instructions. V208 retains the full-width epilogue and its existing valid-row mask. The two constructors separately fall back to the V206 images, followed by the ordinary grouped path if those images also fail. [Source](delta.diff) · [Startup and fallback](startup.json) · [PTX accumulator layout, Figure 152](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n16)

## Work and machine code

| Input prompt tokens | V206 expected tensor rows per rank | V208 expected tensor rows per rank | Issued row reduction |
|---:|---:|---:|---:|
| 8,192 | 11,264 | 9,080 | 19.39% |
| 16,384 | 18,182 | 17,157 | 5.64% |

The model assumes each token selects eight distinct experts uniformly among 256, with 32 resident on each rank. At 16,384 tokens the arithmetic difference is 0.1032 TFLOP per local layer. Routing skew can change the result substantially. Experts with 97–176 rows in their final tile execute the original width. At this geometry the model still requests 2,789 MiB of packed weights and scales per local layer before caching. Native counts determine work during a run. [Model and intermediate cases](work.json) · [Reproduction](work.py)

| SM90a projection | V206 full-width loop sites | V208 loop sites at 32 / 96 / 176 | Drains per loop | Registers | Dynamic shared bytes |
|---|---:|---:|---:|---:|---:|
| W13 gate/up | 1,118 | 1,107 / 1,107 / 1,118 | 16 | 168 | 128,000 |
| W2 down | 1,003 | 989 / 993 / 1,003 | 8 | 168 | 136,192 |

Each loop contains 32 HGMMA sites and zero local loads or stores. The W13 cubin grew from 78,568 to 171,880 bytes, and W2 from 86,848 to 185,920 bytes. W2 now reports 68/196 spill store/load bytes across the function versus 8/8 previously, all outside its steady K loops. Instruction-cache pressure, simultaneous execution of different widths, padded epilogue work and memory-service time can offset the arithmetic reduction. A fourth 160-row branch added 16 sites to the common W13 full-width loop for another 1.44 percentage points of expected arithmetic reduction at 16K. The three-width candidate keeps the original full-width loop lengths. [NVCC 12.9.86 machine comparison and four-width alternative](sm90a.json) · [Discarded shared-epilogue compiler result](rejected.json)

The archived H200 16,384-token synthetic expert call with real 4,096-channel geometry took 7.188–7.394 ms for Marlin across four runs. Its compact dense alternative took 5.831 ms in the final run. No archived Humming W13/W2 phase timing or competition H200 V208 speedup is available. An eight-H200 16-user session recorded 75 long-prompt calls and 1,228,800 new tokens through that shape. The compiled instruction change alone cannot infer serving throughput or a 4× result. [H200 timing and scope](../v181_peer_inputs_paired_gate_up/physical_comparators.md) · [Archived H200 result](../../../experiments/modal_serving/20261113T065259Z/result.json) · [Prompt shapes](../v180_paged_verify_direct_prompt/trace_shapes.json)

## Correctness and reproduction

The published scheduler and generated width selector executed 8,161,632 mixed CTA jobs across 500 histograms, including 5,338 empty experts. A C++ harness executed the submitted residual helper with a host model of the PTX operand layout. It checked all live counts from 1 through 96, long K=4096 cases on both sides of the 32 boundary, all data descriptors, and 1,191,936 FP32 output values. Both operands follow the original K16 order. Four compiled epilogue cases covered both complete sets of 65,536 BF16 input patterns. A replay of the real grouped host route compared 50,350,080 final BF16 components. This establishes register mapping and same-arithmetic model agreement. PTX leaves physical accumulator rounding partly unspecified. H200 token agreement and elapsed time remain unmeasured. [Scheduler](scheduler.json) · [Operand results](operands.json) · [Fused epilogue](epilogue.json) · [Grouped host route](callback.json)

To reproduce, extract the pinned Humming 0.1.12 wheel's `humming/` into a scratch root next to the pinned, patched `sglang/` checkout. Use the project CPU PyTorch environment for the host renderer and NVCC 12.9.86 with Hopper support for machine checks.

```sh
python render.py "$scratch"
sh compile.sh "$scratch" "$cuda_bin" "$compat_include"
python isa.py "$scratch"
python scheduler.py "$scratch"
c++ -std=c++17 -O3 -I"$scratch/sglang/python/sglang/kernels/jit/csrc/muse_spark" operands.cpp -o "$scratch/operands"
"$scratch/operands"
python ../v197_grouped_prompt_gate_up/independent_epilogue.py "$scratch/sglang"
python ../v199_prompt_expert_overlap/independent_callback.py "$scratch/sglang"
python verify.py "$scratch/sglang"
```

The isolated stage and compiler scratch were retired after verifying the export, Git state and open handles. The local compiler VM is stopped. [Host cleanup](host_cleanup.json) · [Guest cleanup](guest_cleanup.json)
