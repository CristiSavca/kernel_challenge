# V275 reuses one activation tile across four expert output bands

V275 doubles the number of output channels per W13 and W2 CTA for high-load grouped Humming decode. At 256 verification rows, the archived H200 routing distribution produces 1,056 tile jobs per projection with V68 and 528 with V275. The candidate saves 198 MiB of combined activation payload requested by TMA per layer and rank. Packed weights, group scales and the set of expert dot products remain identical. The official standalone export is 75,927 bytes with SHA-256 `ab90939602c3f275914cf76e4c232cabe93ec7b2c496c9e5355aadb910676284`. H200 token agreement, kernel latency and serving speed are unmeasured. [Patch](patch.diff) · [Official rules and reconstruction](validation.json) · [Quantities and assumptions](work.json)

## Geometry

The pinned V68 tuner selects 16-row tiles around 16 users and 24-row tiles around 32 users for width-eight verification. V275 selects a 64-output-channel warp in place of the 32-channel warp. The resulting four WGMMA output bands share one buffered BF16 activation tile inside the same CTA. The W13 block covers 512 output channels at K128, with two K-worker groups. W2 covers 256 channels at K256, with four K-worker groups. Each group still processes K64 fragments at the same offsets and the default Humming reducer combines their FP32 sums in order. A compile rejection restores that projection’s original configuration. Width-eight tiles keep the V68 choice, including the 1- and 4-user verification shapes. [Production change](delta.diff) · [24 startup and fallback cases](startup.json)

| 256-row stress input, each rank and layer | V68 | V275 |
|---|---:|---:|
| Nonempty local experts / total M tiles | 26 / 33 | 26 / 33 |
| Independent jobs per projection | 1,056 | 528 |
| Ideal complete waves on 132 SMs | 8 | 4 |
| A payload requested by W13 and W2 | 396 MiB | 198 MiB |
| Packed weight and scale payload requested by W13 and W2 | 891 MiB | 891 MiB |

Under independent uniform top-eight routing, the expected A request reduction over both projections is 125.80 MiB at 16 users and 191.94 MiB at 32. Widening does not remove dequantization of compulsory packed weights. Activation reuse between neighboring blocks can already hit L2. Fewer blocks can also leave an incomplete final wave, and each remaining CTA has more serial work. A separate standalone kernel on this stress input recorded 734.03 MiB of total DRAM reads against 702 MiB of unique weights. That comparison cannot measure V275’s HBM traffic or predict time saved. Earlier V69 used activation multicast across CTAs. V216 found limited additional reuse of the packed weights across row tiles. V275 needs no inter-CTA handoff. [Original H200 profile](../../../experiments/iterations/20261104-moe-hardware-profile-recovered.json) · [V69](../v69_humming_decode_a_multicast/README.md) · [V216](../v216_decode_weight_multicast_feasibility/README.md)

## Offline evidence

| Projection and native M | Chosen stages | Dynamic shared bytes per CTA | Fewer K-loop instruction sites per equal output/K unit |
|---|---:|---:|---:|
| W13, M16 | 4 | 164,864 | 5.80% |
| W13, M24 | 4 | 173,056 | 6.74% |
| W2, M16 | 3 | 136,192 | 6.48% |
| W2, M24 | 3 | 148,480 | 6.66% |

NVCC 12.9.86 compiled 18 SM90a modules from pinned Humming 0.1.12. The four selected images use 96 registers with zero stack and zero spills. Each K16 warp step executes four HGMMA sites before draining. Across the same output channels this halves warpgroup drains compared with two independent V68 CTAs. Four stages on wide W2 produced seven local-memory access sites in the M16 steady loop. Three removed them and reduced its resident shared storage. The differing stage depth can change memory latency and producer stalls. The hardware launch still requests one resident CTA per SM. [Machine code and source digests](assembly.json) · [Compiler analysis](assembly.py)

The operand fixture compares the packed-weight and scale addresses selected by the 32- and 64-channel lane layouts, and the output addresses after Humming’s XOR transpose. It checks 131,072 FP4 coordinates, 32,768 scale-byte coordinates and 147,456 packed BF16 output addresses. Across 192 full-K=4,096 synthetic dots the reference and widened K-worker sequences and FP32 host outputs matched. A linear K-order control changed 188 outputs. These are offline index and arithmetic checks. [Numerical fixture](operands.py) · [Results](operands.json)

Folding the per-expert global gain into packed weights would change the BF16 projection boundary. With activations [1, 1/256], two FP4 weights of 1, a unit block scale and a global scale of 1.5, the incumbent produces 1.5. Folding the gain produces 1.5078125. The constructor retains the original post-projection scale. The selected-source replay regenerated all six native or widened images from the actual production setup and matched each archived compiler input. [Scale counterexample](operands.json) · [Source closure](selected_codegen.json)

To reproduce, apply the official export in a pinned checkout and unpack Humming 0.1.12 into a sibling directory. Run `render.py humming sglang images` from a CPU PyTorch/Jinja environment and direct its JSON to `configs.json` in their common scratch root. Run the existing V213 compiler wrapper over the generated CUDA sources, followed by `assembly.py scratch`, `operands.py humming`, `startup.py humming sglang images` and `verify.py sglang`. The native serving compiler, device, routing and cached graph remain the required speed test. [Compiler wrapper](../v213_live_decode_expert_reduction/compile.sh)
