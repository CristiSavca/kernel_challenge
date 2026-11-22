# V94 clustered expert down projection

V94 lets neighboring W2 output blocks share the scaled SiLU computation introduced in V90. On a compatible Hopper, a cluster of two, four or eight CTAs owns adjacent 512-column output bands for the same expert and row tile. Its threads divide the 128-wide K slab into four-element BF16 vectors, evaluate the incumbent gate/up arithmetic for each live vector once and write that vector into every member’s distributed shared memory. Each block then runs its W2 matrix loop using its own shared-memory copy. For H=4096 and a supported eight-block cluster this changes eight SiLU evaluations per live routed element to one. The FP4 decode, FP32 matrix accumulations, W2 output rounding and subsequent post-expert norm follow V90.

All blocks in a cluster share the same device route ordinal. They enter the same 8-, 16- or 32-row specialization or all exit with no work. An initial synchronization establishes remote shared-memory lifetime. Each slab has a publication barrier before W2 reads and a completion barrier before any producer overwrites shared data or exits. Only cluster peers participate.

At model preparation the extension probes the exact specialized kernel's occupancy for legal output-band divisors in order 8, 4 and 2. A zero-residency result or an unsupported cluster geometry tries the next size. An unsupported device, device mismatch or incompatible width uses one block. This selection happens before CUDA graph capture. The environment setting `SGLANG_ENABLE_MUSE_SPARK_HUMMING_DOWN_CLUSTER=0` selects the V90 block-local implementation. [CUDA thread block clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)

## First check at 16 users

**Choose V90 for the first high-load check.** It isolates the new fused W2 computation with block-local synchronization. For 128 target rows and uniform top-eight routing across 256 experts on eight ranks, V94 at cluster size eight saves 3,670,016 SiLU evaluations and 14 MiB of logical gate/up reads per rank per layer. Those input reads can hit L2. W2 still reads about 283 MiB of packed weights. Publishing the padded fragments now sends about 14.03 MiB through peer shared-memory stores. These are estimates for independent uniform token routes; the judge’s routing histogram and memory transactions have not been observed. [Work estimate](uniform_work_estimate.json)

| SM90 build | Registers / thread | Static shared bytes | Spill loads and stores | Dynamic synchronization per live tile at K=4096 |
| --- | ---: | ---: | ---: | --- |
| V90 or V94 one block | 96 | 8,704 | 0 | 64 CTA barriers |
| V94 two blocks | 112 | 8,704 | 0 | 65 cluster barriers |
| V94 four blocks | 112 | 8,704 | 0 | 65 cluster barriers |
| V94 eight blocks | 108 | 8,704 | 0 | 65 cluster barriers |

The compiled register budgets permit two 256-thread CTAs per H200 SM by register count. Cluster scheduling requires co-residency within a GPC, and the barriers must wait for all member CTAs. The compiler emits peer-address mapping, vector stores, and cluster arrive/wait instructions. At size eight, all 72 static mapped addresses in the three fragment branches feed exactly one four-BF16 store apiece. CUDA compilation and simulation establish the work and resource counts. End-to-end speed, graph replay, peak memory, TTFT, requested lengths and model agreement await H200 testing. [Instruction counts](sm90_cluster_isa.json), [PTX](down.ptx.gz), [disassembly](down.sass.gz), and [compiler output](sm90_compile.log)

## Reproduction

The official [patch](patch.diff) is 202,438 bytes across 33 permitted files and its SHA-256 is `242e038af77f04804623b4dc6fcbbed114a104dbc0554294c2750aed4ea558f2`. The [three-file delta](delta.diff) is against frozen V90. The release uses V89’s launch flags. The complete SM90 cubin hash is `90f86f2e13e9687e560eab839100dd826996e869f08b0d387a21e1f82d69cf0f`. [Patch validation](validation.json)

A CPU cluster simulator ran the actual vector publication and full W2 arithmetic for all four sizes, scheduling warps to run ahead until they reached a barrier. It checked 612,352 live BF16 results, 101,376 inactive-prefix elements, the exact live activation count, and every expected peer-vector store count across 17 scenarios, including K=N=4096. A separate executable tested 2,240 paths through the actual host occupancy selection, including recoverable unsupported sizes and propagated unrelated errors. The independent check covered 67,584 vector owners and 315,392 peer writes. [Numerical results](cpu_result.json), [host tests](host_result.json), [independent review](independent_review.md)
