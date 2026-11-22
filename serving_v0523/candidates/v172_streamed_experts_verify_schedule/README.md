# V172 shares FA3 verification schedules on V168

V172 removes 60 repeated FA3 scheduler preparations from an eligible 62-layer target verification in the observed Muse configuration. The 16 full-attention layers share one native scheduler tensor and the 46 sliding layers share a second. Both use the original query, physical KV, lengths, output and native attention kernel. V172 keeps V168’s streamed decode experts and sparse prompt exchange. The candidate adds V151’s opt-in `--speculative-dflash-share-verify-schedule` to V168’s five flags. The public parser accepts the switch and defaults it off. The frozen patch is 160,935 bytes across 24 paths, SHA-256 `ba8bff16c3e03e8f39160812d839564d2da4e6b477a3e3579f23aaef50f7ae26`. [Patch](patch.diff) · [Full flag map and reconstruction](validation.json)

| Simultaneous users | Native full / window split count | Distinct scheduler tensors |
| ---: | ---: | ---: |
| 1 | 57 / 15 | 2 |
| 4 | 15 / 14 | 2 |
| 16 | 4 / 4 | 2 |
| 32 | 2 / 2 | 2 |

The cache is scoped to one model call and one stream. Its key includes the live length and query-offset tensor identities and window geometry. Graph capture records the two native preparations and replay refreshes their contents. Equal split counts at the larger loads still require two tensors. The pinned FA3 allocation code produced 65/49 int32 words at 16 users and 129/97 at 32. Full attention sorts the batch and the sliding kernel reads its swizzle vector and semaphore at different offsets. [Native FA3 allocation replay](native_layout.json) · [Pinned builder](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_api.cpp#L616-L637)

The integrated host fixture ran the Muse attention writer and FA3 consumer 1,736 times, checking 47,108,096 newly written FP8 bytes across 28 verifications. A direct V168 comparison checked 1,488 native call records across four loads, including changed lengths, native fallbacks and stale prefill certificates. V168’s prompt suite also passed 1,344 model and replay cases at TP2/4/8, with 2,943,360 BF16 owner outputs matching the reference. The native BF16/FP8 CUDA kernels and FA3 timing were simulated in the host fixtures. [Writer and attention evidence](combined.json) · [Direct V168 comparison](independent_modes.json) · [Prompt Gloo replay](prompt.json)

The inherited decode expert replay also matched V168 exactly across 54 setup cases, 72 dispatch cases and 18 warp-level numerical cases. An independent review reconstructed the four donor changes, verified the sixth flag and checked the native full/window launch geometry and semaphore reset. [Decode dispatch](decode_dispatch.json) · [Warp results](decode_warp.json) · [Independent source review](independent_source_review.md)

The launch count is a per-pass source result. The native fused path can overlap some preparation, and its standalone alternative uses a different launch policy. Neither end-to-end improvement nor 2× or 3× follows from the count. A private H200 check against V168 at the same load will be needed to isolate this component. The submission window and allocations were unchanged. [V151 scheduler provenance](../v151_integrated_verify_scheduler/README.md) · [V168](../v168_streamed_experts_sparse_return/README.md)
