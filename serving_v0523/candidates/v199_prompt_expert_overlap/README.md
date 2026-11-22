# V199 grouped prompt expert W2

V199 runs an additional Hopper kernel for the **down projection of long, grouped expert prompts**. It applies to V198's regular grouped expert entry, including the case where the paired W13 implementation declines. Decode and smaller prompts retain their existing kernels. The official cumulative patch is 203,767 bytes, SHA-256 `6a2c175fb20c4b6a0c8304467997f72f26a34e87f5f122434e4c975e274a5ec0`. The five inherited launch flags and three changed files are recorded in the [manifest](validation.json). [Full patch](patch.diff) · [Delta on V198](delta.diff)

## Matrix pipeline

The 176×256×64 grouped W2 uses eight math warps and four loader warps. For each K64 tile, the loaders fetch 1,024 E4M3 scale bytes and expand them into 2,048 BF16 bytes in shared memory while TMA transfers the activations and packed FP4 weights. The math warps read prepared scales, apply the original FP4 bit unpack, and keep two ordered K16 slices outstanding at a time. Both output bands finish before those register operands and the underlying A tile can be reused. The full K4096 reduction has the same operand order and 128 group drains per math warpgroup, down from 256. Preparation keeps the existing packed bank and requires no extra persistent GPU allocation. [Machine audit](sm90a.json) · [Barrier and scale-address replay](independent_ring.json)

Humming initially represents the FP4 value at its E2M1 exponent offset. V199 multiplies by 2^124, then by the same prepared BF16 scale, giving the native quarter-scaled MMA operand. The existing Humming epilogue recovers the remaining factor four and the original per-expert BF16 global scale follows. Checkpoint preparation already maps the seven positive E4M3 subnormals to zero and rejects negative or nonfinite values. All 33,292,288 tested operand bit patterns matched, including 2,998,272 negative zeros and every one of the 2,032 admitted checkpoint-scale/signed-nibble pairs. The host routing, global-scale and weighted-RMS fixture compared 50,350,080 BF16 elements across 36 finalizations. [Operand replay](operands.json) · [Call-path replay](independent_callback.md)

## Compiler and time evidence

| SM90a metric | V198 native W2 | V199 W2 |
|---|---:|---:|
| Consumer instruction sites in four K64 stages | 1,127 | 1,003 |
| Loader instruction sites in four K64 stages | 285 | 385 |
| HGMMA instructions / wait sites | 32 / 16 | 32 / 8 |
| PTXAS registers per thread | 168 | 168 |
| Function-level spill stores / loads, bytes | 56 / 80 | 8 / 8 |
| Local-memory instructions in the consumer backedge | 0 | 0 |
| Shared bytes per CTA | 128,000 | 136,192 |

At the model's geometry, both shared allocations limit an H200 SM to one resident CTA. Weighting instruction sites by eight consumer and four loader warps gives 10,156 versus 9,564, a 5.83% decrease. Neither figure measures elapsed time. The candidate trades a dedicated scale TMA for coalesced loads and arithmetic on the loader warps. Savings depend on whether those warps keep the math warps supplied. Actual V199 H200 latency, token agreement, score and 4× throughput remain unmeasured. [Compiler and register-lifetime evidence](sm90a.json)

An archived H200 trace at 16,224 local routes measured 7.188 ms for the older stock expert path, 5.831 ms for serially prepared dense experts and 2.698 ms with pre-decoded operands. It measured both projections and activation on a different software path. It does not establish a V199/V198 speedup. A full BF16 expert bank would occupy 3 GiB per layer and rank. Native Humming also rejects 16-bit weights. A direct dense GEMM can overflow an intermediate where Humming's internal quarter-scaled operand fits, as the executable witness shows. [Archived H200 measurement](../../../experiments/modal_serving/20261113T065259Z/result.json) · [Type restriction](configs.json) · [Precision witness](operands.json)

Two other SM90a builds expanded the complete FP4 tile on the loader warps. Their consumer loop fell to 159 sites over three K64 stages, while their producer grew to 2,013 or 2,325 sites. After normalizing the four-stage native loop to three stages, weighted issue work grew 22% and 39%, respectively. Both were rejected. A scale-only build with a separate ready barrier also cost 483 producer sites in four stages. The shipped version fuses scale expansion with the ordinary transfer stage. [Rejected and accepted builds](sm90a.json)

## Reproduction

Use the pinned SGLang base, the [official cumulative patch](patch.diff), and the published humming-kernels 0.1.12 package. The compiler fixture expects a scratch directory with the patched source in `sglang`, the extracted wheel's `humming` package and a writable `rendered` subdirectory. `render.py SCRATCH` constructs the incumbent and candidate with the vendor SM90 tuner, using PyTorch CPU to supply the offline context. Build the generated CUDA sources with NVCC 12.9.86 using [the compilation flags](compile.sh). Then run `startup.py SCRATCH`, `isa.py SCRATCH`, `operands.py SCRATCH` and `independent_ring.py SCRATCH`. The optimized native machine image is reproducible from the shipped sources without an NVIDIA device. [Startup and ELF symbol tests](startup.json) · [Host dispatch command and checks](independent_callback.md)

The public kernel loader must accept the native cubin and its checked instruction stream. An unsupported toolkit, missing disassembler, unexpected instruction schedule, incompatible tune or changed weight binding uses the existing Humming W2. No hardware job or runner change was made for this candidate. [Export and public-rule checks](validation.json)

The archived variants' loader implementations and source adapters are in the [compressed diagnostic source bundle](diagnostic_sources.json.gz). Their measured compiler counts and cubin hashes are retained in the machine audit.
