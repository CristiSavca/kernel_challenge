# V25 bounded-memory native MoE serving design

Source base: standalone `muse_moe_w4a16/submission.py` SHA256 `247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0`. Serving base: SGLang `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`. This is an isolated component prototype design, not a checkout patch or validated serving replacement. No GPU/provider call is authorized by this document.

## Selected format and why the original dispatch cannot be retained cheaply

The original `_prepare_layout` retains four complete code+scale copies, twice: separated packed codes/scales (entries0/1/3/4), fat (8/9), unit (10/11), WGS (12/13). For32 local experts, gate/up8192×4096 and down4096×4096, one full NVFP4+E4M3 copy is905,969,664 bytes =864MiB =0.84375GiB. Eight copies are6.75GiB/layer, before scratch or the original weights.

Actual enabled device readers permit fewer copies, but not one while preserving dispatch:

| Selected parent build | Reachable packed reader | Bit positions | Required full copies |
|---|---|---|---|
| M<=4, TINY/WS | `ws_helper`/`skinny_ring_ws` via `ring_issue`, plus FAST1 preissue, use only `p.wAu/p.wBu` | `_POS` / MULDEC0 | unit |
| M5..8 TINY2, M9..16 MID | same unit readers; MID splits jobs into <=8 routes | `_POS_MUL` / MULDEC1 | unit |
| M>16, PRMT-qualified MAIN | `wgs_wproducer` reads only `p.wAs/p.wBs` | `_POS_MUL`, KPERM3 | WGS |
| M>16, non-PRMT MAIN fallback | skinny ring uses unit; fat ring uses `p.wAf/p.wBf`; default WGMIN9 excludes natural WGMMA | `_POS_MUL` | unit+fat |

The host binds every legacy pointer, but host pointer construction is not a device read. Dead ordinary/natural/fat banks cannot justify persistent allocation. Nonetheless both qualified and fallback parent dispatch need **three** full copies across token counts. Making tiny use MULDEC1 would reduce this to two, still1.6875GiB/layer and a changed tiny build. It is not the selected design.

**Selected candidate: WGS-only for all token counts, one `_POS_MUL` KPERM3 layout.** Preserve256-row/256-K units, FT64 jobs, the three joined weight/X ring slots, four consumer warpgroups, two producer warps, and all existing scheduling/accumulation order. WGS already supports NTT1..8; the M1/4/8/16 dispatch is new and has no inherited skinny performance evidence. Chunk larger token counts into <=384, with no skipping of routes. No natural/unit/fat tensor remains in the weight state. The existing source functions may remain for attribution and review, but the dedicated host interface binds only the active WGS banks/scalars, BF16 route outputs and scratch.

The implemented scale decoder follows the reviewed Marlin operand contract: positive normal bytes0x08..0x7e use the exact PRMT bias0x3f803f80; zero and subnormal bytes use a LUT derived from the actual stock FP8→BF16→FP16→half-times128→shifted-byte transformation. Direct `sf*128` is wrong for bytes1..7. The initial supported domain is nonnegative finite E4M3 bytes0..126 and finite prepared BF16 globals. Unsupported layers must remain stock before ownership transfer. No secondary weight format is retained.

## Exact bounded memory

One retained weight copy consists of603,979,776 B for gate/up and301,989,888 B for down. Global scales and effective BF16 post-norm weight add only bounded small arrays; their final dtype/storage follows the precision contract. A preallocated final pair is filled in expert order, dropping temporary repack tensors after each bounded block/expert. The parent lists plus `torch.cat` are not used because they retain an entire second bank during concatenation. Original code nibbles and E4M3 bytes are repacked losslessly; no dequantized weight tensor is retained.

Parent scratch is164,668,960 B per layout. The component stops at BF16 expert outputs and removes the native norm sums, FP32 y buffer and row-major `inter`. Its fixed384-token **Workspace is108,033,568 B =103.028839MiB**:

| Allocation | Bytes |
|---|---:|
| Counters |4,128|
| Split fixup storage/counters |20,985,344|
| BF16 xA/xB,3776 rows each |61,865,984|
| BF16 route outputs[384,8,4096] |25,165,824|
| Local IDs[384,8] |12,288|

The initial official component state owns one explicit Workspace and reuses it serially. Counters/fixups are zeroed once at construction; inherited per-call reset/release remains in the kernel. No fresh full-workspace allocation/zeroing is hidden or added per invocation. Independent callers need distinct workspaces; simultaneous reuse of one object or its captured graphs is unsupported. Cooperative launch fixes grid admission, not workspace races. This component ownership is not a production model resource: one workspace per layer/graph would multiply storage, and a safely serialized model-resource workspace/capture policy remains an integration gate.

Packing preallocates only the final two tensors and writes one expert in2048-row slabs using the unchanged KPERM helpers. This bounds temporary repacking, but original checkpoint/Marlin tensors still belong to the caller until a future validated ownership transfer. No production parameter is freed here.

One additional layout beside Marlin still costs50.625GiB for60 NVFP4 layers (52.3125GiB for62 illustrative layers); it is not a completed memory solution. Full deployment requires a pre-capture ownership transition that replaces selected Marlin weight buffers and drops raw/repacked temporaries. The current weight loader converts raw tensors to Marlin layout and deletes/replaces them; an integration cannot presume original raw tensors remain. An isolated component can accept raw checkpoint tensors during prepare without retaining them, but that alone does not implement this full-model transition. Model-loading rules, exact conversion timing, state-dict references, fallback policy and capture pointer lifetime require a separately reviewed integration boundary. No existing parameter is freed by this work.

## Runtime and precision contracts

Every EP rank is supported in the source contract, including no local routes. The new `wgs_ranges` nj0 guard initializes both phase ranges to zero before stage lookup/division. Producers/consumers continue through inherited joins, and CTA0 publishes a complete local-ID mask. Remote route output bytes remain undefined; the original stock finalizer masks them before reading. CPU tests compile the actual zero-job range entry, but do not establish GPU global-barrier/reset behavior or all-remote correctness.

The implemented host uses **only `cudaLaunchCooperativeKernel`** on the current stream. It checks cooperative support, both BIG pointers' dynamic-shared configuration and actual occupancy, selected-pointer occupancy at launch, grid<160 fixed fixup slots, and grid<=SMcount*resident capacity. Every CUDA status and launch return is checked; there is no ordinary-launch fallback. Inactive PDL asm is removed. Generic capture/multi-GPU and independent-workspace concurrency still require actual runtime validation.

The reviewed arithmetic boundary is documented in `tools/SERVING_NATIVE_MOE_V25_PRECISION_20261104.md`. Gate/up collapse uses the stock gate scalar, rounded to BF16 before multiplying2^119. Both GEMM epilogues perform BF16(accumulator) followed by BF16(global product). PhaseA then uses explicit fast exp/div and **only its final SiLU×up product uses `mul.rn.ftz.f32`**, matching the stock activation's fast-math policy. The module otherwise has no global fast-math/FTZ. The active split FP32 `red.global.add.v4` was replaced with integer-CAS retry loops and explicit `add.rn.f32`, since the original reduction instruction independently flushes subnormals whereas stock Marlin's ordered FP32 reducer does not. CAS retry/address/release source is independently reviewed; CPU scalar tests are not a GPU atomic-interleaving proof.

PhaseB writes BF16 expert output at the original route-slot index. The adapter imports the original stock finalizer (required source SHA256 `9b816e971817f96a22d2d09424dc8dfa6f6020e18de73fb895674c76494fe591`); the modified local checkout is not evidence of evaluator binding. Norm/score ordering remains stock. For totalM>384 the adapter chunks the core; a final chunk<32 is padded/masked to32 before finalization so an original large-T call retains the large-T finalizer branch. This extra timed work and larger-prefill behavior are unmeasured. No universal numerical identity is claimed across different tensor-core accumulation orders.

Initial component domain is explicit: BF16 activations, H=I4096, topK8, contiguous ownership of up to32 local experts within256 global IDs, effective BF16 weight-first post-norm and the actual epsilon. No claim covers arbitrary expert mapping, alternative norm semantics, other GPUs or graph ownership. Unsupported metadata must choose stock **before** freeing its buffers; it must never run a partially supported native path after replacement.

## Frozen component and next gates

The isolated source is `serving/candidates/native_serving_v25/component.py`, generated by `build.py` with exact edit reversal to parent247888, six byte-identical packing helpers and two PRMT preprocessing checks. Its distinct device symbol is `native_serving_v25<false/true>`. The focused CPU report is `serving/candidates/native_serving_v25/cpu_result.json`: actual slab packing and inverse coordinates,127 valid scale bytes/16,129 scale pairs, BF16/subnormal boundary counterexamples,64 job sizes/8,519,680 output coordinates, compiled132-CTA nj0 range entry, rank mapping and exact scratch bounds. Host checks are source/policy tests, not CUDA runtime mocks. There is no V25 Lean certificate: the existing strict binder does not cover this composition.

The fixed first experiment is one root-owned official MoE component check, all5 correctness tests/6 benchmarks/6 profiles clean, intended cooperative core and original stock finalizer active. Require geometric-mean speed>=1.15 against the same-run official stock baseline and no individual candidate/stock latency ratio>1.02. No threshold retune or unconditional public promotion follows. Current native247888 remains the public incumbent.

The official stock baseline uses a different block-size heuristic from the actual serving caller (local32 versus global256 experts), so passing the component screen does not establish actual serving speedup. Deployment additionally requires matched serving comparison, unsupported-scale fallback before weight replacement, loader/buffer ownership, full-model memory/capture fit, all-remote/other ranks, scratch reuse and independent-stream tests, larger prefill/TTFT, and full8-GPU correctness/performance. Preserve the active submissions and dirty checkout throughout.
