# V293 writes RMS-normalized prompt rows directly into peer memory

On an eligible long-prompt Humming expert call, V291 first normalizes the local token shard to a BF16 temporary, then copies that temporary into a peer-visible buffer. V293 writes the normalization output directly into the selected buffer. Each rank publishes the same values at the same synchronization point. The existing Hopper RMS kernel accepts a single full-width input partition and an identity map with one destination per token, so its CUDA source and compiler flags carry over unchanged from V284. At the 16,640-row TP8 ceiling, the identity map takes 8,320 bytes per GPU across all layers. [Production delta](delta.diff) · [Compiled Hopper source and comparison to PyTorch](../v284_rank_major_routed_pool/independent_native.md)

Eight production-constructor cases also exercised native-width buffers and JIT, build and allocation fallbacks. [Constructor tests](native_setup.json)

The direct path requires the pinned PyTorch build, native unhooked RMSNorm, aligned contiguous BF16 input and effective gain, and an available compiled operator. An uncached zero-centered weight, hooks or dispatch overrides retain the original normalization. A rejected rank vote gathers normalized rows before running ordinary experts. A privately replaced grouped helper receives the original five-member route plan and normalized input even if it changed after capture. The two-slab and single-slab publication protocols keep their existing barrier counts. [Runtime guard tests](admission.json) · [Distributed replay](transport_norm.json)

## What the change can recover

At TP8, a 16,384-token prompt has 2,048 local rows of 4,096 BF16 channels per rank. Across 62 layers, direct output removes 62 copy launches and 1.9375 GiB of logical reads and writes per rank. At 4.8 TB/s, that entire byte count corresponds to 0.433 ms. Cache hits, launch overhead and the later scheduling of normalization relative to the routing vote affect real time. The historical V41/V44 four-user comparison moved the mean 16K chunk by 177.50 ms. Getting that historical interval to one quarter would require a further 874.27 ms. V293 has no measured H200 speedup or four-load score. [Work model](work.json) · [Historical intervals](../v288_paired_cold_and_verify_intervals/check.json)

The target model’s NVFP4 constructor requires expert TP1. Under TP8/EP8 each of the 256 routed experts has one owner. Each rank holds 32 distinct experts, and moving one whole expert would require its 27 MiB of packed weights and group scales per layer. In the archived 16K synthetic replay, the busiest rank owned 16,540 live routes against the mean of 16,384. Even if every rank already held every weight and each route cost the same, perfect balance of those counts would give 1.0095×. [Source-backed ownership and histogram](work.json)

## Verification

The unchanged CUDA body ran under a host warp interpreter at nine row/width combinations, including native width 4,096 with 2,048 and 2,080 local tokens. All 21,738,788 BF16 output words matched an independent replay of ATen’s vector and warp reduction. All 21,878,684 words checked in the neighboring bank and unused suffix retained their sentinel. CPU rsqrt was common to both numerical models. [Arithmetic and layout](numeric.json)

The TP4/TP8 Gloo replay ran actual Python peer publication and Humming grouped staging with mapped host slabs and the interpreted CUDA body. Grouped staging and gathered-fallback inputs matched across 6,307,840 BF16 words, including asymmetric norm eligibility, empty expert owners, a remote veto and large-allocation fallback. The normal module and compiled output are compared on host. H200 token agreement, latency, TTFT and combined peak memory require hardware measurement. [Distributed results](transport_norm.json)

The official export is 401,544 bytes across 61 permitted paths, with a longest added line of 147 characters. Three files differ from V291. Strict reconstruction, reverse application, flag checks and CUDA source identity passed. [Official export](patch.diff) · [Validation](validation.json)
