# V210 paired query-tile multicast for cold prefill

**Retain native FA3.** Two adjacent full-causal query tiles often read identical K/V tiles, including at one user. The pinned Hopper binary exposes no FP8 D64 causal cluster specialization. Its distributed wheel omits the native Hopper mainloop headers. Exploiting this opportunity requires a new scheduler and attention specialization. This study has no serving patch, official submission or H200 timing. [Reproduction](check.py) · [Results and source hashes](check.json)

## Eligible work

V178b provides a dense FP8 K/V operand during eligible long prompt appends. It writes all current keys and values into the ordinary cache at the original physical positions. V60 enables TMA for a compatible coarse page layout, V105 gathers the sliding-window suffix, and V62 splits target verification. V206 changes expert projection scheduling. Each retains the native full causal prefill launch. V178b is an alternative input producer on V64. A composition with V206 would need separate validation. [V178b input contract](../v178b_live_direct_prompt/README.md) · [V60](../v60_fa3_paged_tma/README.md) · [V105](../v105_swa_suffix_prefill/README.md) · [V62](../v62_paged_verify_attention/README.md) · [V206](../v206_steady_decode_gate_up_pipeline/README.md)

The executable calls the pinned C++ BlockMN on all five chunks of the 74,160-token first-prompt geometry from V64. Packed GQA groups eight heads, so each 192-row M tile covers 24 logical queries. Each N tile carries 160 rows of K and V for one 64-channel FP8 KV head. A pair qualifies when both CTAs have identical lower and upper K-tile bounds and offsets. The model retains the two local KV heads and each CTA's original row-specific causal mask. [Actual per-chunk counts](check.json) · [Native bounds](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/block.h) · [Parent geometry](../v170_long_prefill_attention_feasibility/check.json)

| Quantity across the five prompt chunks | Result |
| --- | ---: |
| Adjacent pairs per local KV head | 1,544 |
| Pairs with identical K-tile sequences per head | 1,389 |
| Unpaired query tiles per head | 4 |
| K/V tile reads before pairing, per layer and rank | 1,436,918 |
| K/V tile reads saved, per layer and rank | 645,326 |
| Logical input bytes before pairing, all 16 full layers per rank | 438.51 GiB |
| Maximum logical input bytes saved by this pairing | 196.94 GiB |

This pairing reduces modeled operand loads by 44.91% in these layers. Perfect scaling of that operation alone would be 1.815×. Both CTAs still perform their own QK and PV operations and maintain independent FP32 softmax states. The 46 sliding-window layers are excluded. At 74,160 live rows the layer's two local FP8 K/V heads occupy about 18.11 MiB, so cache hits can absorb repeated loads. V64 measured 5.419 s first-turn p90 TTFT at one user with no in-request counters attributing time or HBM traffic to these loads. An end-to-end speedup remains unknown. [V64 official run](../../runs/abd217baf0ed49fd.json) · [V206 phase sensitivity](../v206_steady_decode_gate_up_pipeline/phase_sensitivity.json)

## Deployment constraints

1. The pinned launcher's compile-time cluster gate requires head dimension 192 for FP8 and excludes causal, local, variable-length, split and non-TMA paged attention. The served D64 causal variable-length call selects one CTA. Its persistent scheduler assigns individual work tiles. A multicast pipeline needs both CTAs to agree on request, KV head, ordinal K tile and barrier epoch before every transfer. [Native launch selection](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_fwd_launch_template.h#L226)
2. In the first 16,384-row chunk, M tiles 12 and 13 start with K tiles 1 and 2 respectively. The native cluster partitions the transfer by CTA rank. A shared mask would combine parts of different tiles. Changing only the mask also leaves each half-transfer incomplete. A new implementation must explicitly handle unmatched and unpaired tiles, including padded query rows. It must preserve descending K traversal, FP8 probability conversion, register permutation, input descales and final value descale. [Native load and MMA mainloop](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp)
3. Pinned SGLang’s FA3 backend calls sgl_kernel.flash_attn. Its AOT CMake fetches the vendor repo at build time and installs flash_ops as a shared library. The checker enumerated all 91 paths from the central directory of the x86-64 0.4.9 wheel specified by SGLang's Dockerfile. It found sgl_kernel/flash_ops.abi3.so and no native Hopper mainloop header. The in-tree CuTe SM90 frontend fixes its cluster shape to 1×1 and its FP8 interface requires SM100. Depending on an undeclared compiler source checkout would introduce a runtime dependency. [Pinned CMake](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/kernels/aot/CMakeLists.txt#L84) · [Pinned package version](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/docker/Dockerfile#L12) · [Package inventory](check.json)
4. The original launcher's direct and transitive quoted same-directory headers total 475,369 bytes across 22 files, including SM80 code. Copying them in full exceeds the 400 KB submission limit before wrappers or CUTLASS. A hand-specialized rewrite might fit. The V206 base alone uses 293,751 bytes, and the original Hopper mainloop alone uses 122,885. This leaves no direct drop-in source path on that base. [Header inventory](check.json) · [Serving restrictions](../../RULES.md)

Run `python3 check.py` to repeat the immutable-source inspection, wheel-index check and host C++ replay. The checker downloads small source files and archive-index ranges and releases the temporary build directory on exit.
