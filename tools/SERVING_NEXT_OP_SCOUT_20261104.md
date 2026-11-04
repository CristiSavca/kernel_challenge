# Next serving operation scout — November 4, 2026

**Highest-confidence next hypothesis: have the existing fused router also emit Marlin's local expert IDs, eliminating the later Torch indexing launch.** Preserve the global IDs and FP32 scores. On the selected Marlin path, stop generating unused TRT-LLM packed IDs. This is a serving-only fusion opportunity with a modest measured component scope; there is no supported large end-to-end speed claim. No implementation, provider call, GPU run, official check, checkout change or commit was performed.

Pinned SGLang HEAD is `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`. The working tree contains the separately owned finalizer and older alignment changes. The routing/model files inspected here are unchanged. The new idea does not modify alignment, Marlin tile selection, expert arithmetic, finalization or collectives.

## Actual call chain and duplicate work

1. `python/sglang/srt/models/muse_spark.py:806–907`, `_forward_experts`, computes the pooler and then invokes `FusedMuseSparkRouting` on its existing side stream. At871–872 it always requests `want_packed=True`.
2. `kernels/ops/muse_spark_v12/fused_muse_spark_routing.py:899–938`, `forward_jit`, allocates the packed int32 `[T,K]` output and calls the CUDA JIT wrapper. JIT is the first backend priority and is eligible for the actual CUDA/256-expert/top8/sigmoid configuration. A full serving runtime trace is still needed to establish its actual selected backend; a user override must retain the old fallback.
3. `kernels/jit/csrc/muse_spark_v12/routing.cuh:253–272` already holds each selected global expert index and its FP32 normalized weight in the owning lane. It writes IDs/scores and calls `pack_id_weight`, adding a BF16 weight conversion and bit packing for the optional third output.
4. `srt/layers/muse_spark_fp4_moe.py:683–704`, `MuseSpark1xFp4Experts.forward`, selects Marlin immediately when `self.use_marlin`. That call does **not** pass `packed_topk_ids` to `_forward_marlin`; the buffer is unused by this backend.
5. `_marlin_expert_outputs:469–475` converts IDs to int32 and separately computes `expert_map[local_topk_ids]`. On the production path the router has already produced contiguous int32 global IDs, so the conversion/contiguity calls are normally no-ops; the advanced indexing is a real kernel and creates the local-ID tensor. `_expert_map` at174–181 is int32 length256, mapping this rank's32 experts to0..31 and every remote expert to−1.
6. These local IDs feed the unchanged aligner, SwiGLU masking and finalizer. Marlin GEMM arguments, output BF16 casts, FP32 expert norm/weighting, all eight routes and rank reduction remain unchanged.

The existing router's `local_expert_mapping` is **not sufficient by itself**: it changes the ordinary emitted `topk_ids` to local IDs. Setting that field globally would change consumers that require global identities. Nor are TRT packed bits a valid local-ID array: their low half contains rounded BF16 score bits. Use a separately named optional output with exact int32 values.

This opportunity is already absent from our current native decoder: `muse_decode_layer/submission.py:14103–14107` `_route_split` writes both global `IDS` and mapped `LIDS` in the same Triton launch; `_route` does likewise. Standalone247888 consumes supplied routes and performs its local filtering inside its fused kernel, not a separate Torch indexing launch. Consequently this is **not** a new native-kernel leaderboard candidate and does not reopen failed V20/V21/V23.

## Concrete source boundary for a future implementation

Add optional per-invocation `marlin_local_ids_out: int32[T,K]` and `marlin_expert_map: int32[E]` arguments at the selected JIT routing boundary. Defaults remainNone. Retain the current return tuple contract and all old `want_packed`/`local_expert_mapping` semantics; the caller owns the optional output buffer rather than overloading the packed return value. Extend only the required JIT wrapper/CUDA parameter validation, with an unchanged old path when the actual router backend cannot serve the new output.

In the selected non-SP NVFP4 Marlin caller, allocate the local output instead of the unused packed output, request `want_packed=False`, and pass the actual current expert-map tensor. The final selected-ID lane writes the original global ID and FP32 score exactly as before, plus `map[global_id]` to the new output. No top-k comparator, score accumulation, sigmoid/division, normalization or ordering changes. Identity mapping must work when the expert map is absent; a conservative first implementation may keep that non-EP case on the existing path.

Thread a new optional `local_topk_ids` argument through `MuseSpark1xFp4Experts.forward` → `_forward_marlin` → `_marlin_expert_outputs`; validate device/dtype/shape/contiguity and use it instead of the indexing expression when present. Missing output must use the exact original mapping. Do not infer that an arbitrary int32 input is already local from its value range. The caller/parameter explicitly owns that contract. This does not require a pre-expert-norm or finalizer rewrite.

Do not use mutable module-global output buffers or permanently cache the mapping pointer in a way that survives `.to()`/buffer replacement. Read the actual `_expert_map` each call. Add the optional output to the existing side-stream→main-stream `record_stream` ownership list at874–884, retaining the existing wait. The new output is published by the same router launch and consumed after the same join. No extra cross-stream event or collective is required.

## IDs, padding and graph hazards

- The explicit global consumer is `get_global_experts_capturer()` at886–893: it receives every selected global ID in original token/slot order. Preserve those values, including remote routes. The SP branch at895–897 and `_forward_sp_experts:923+` dispatch by global IDs; keep it on the old path initially. First/last BF16 expert layers and non-Marlin backends retain their original routing/packed-output contracts.
- No direct DFlash acceptance decision consumes these expert IDs in the inspected call chain; DFlash changes physical token counts and graph padding. Normal verify graphs use8 rows per captured request, so test actual T8/16/32/128/256, not request concurrency as T. Do not specialize on scored seeds, selected expert sets or conversation content.
- The reusable router supports `num_valid_tokens`: its early padded-row branch writes ID−1 and returns before score arithmetic. A new local output must also write−1 there and must never index the map with−1. The pinned Muse caller does not set this field in the inspected functions. Do not silently expand its padded-score semantics: either handle the existing sentinel contract exactly or leave non-None valid-token mode on the unchanged fallback for the first experiment.
- Global capture must still work with its optional preallocated `sol_topk_out`. Output strides, pointer aliases, empty T, maps for all eight ranks, all-remote rows and concurrent graph instances need explicit validation. The global/local buffers must not alias each other or the input map. Existing PDL trigger/wait placement is unchanged.

## Measured scope and logical byte accounting

Official v22 component job`cffa322caeff4aa2` contains one `index_elementwise_kernel` row per shape. Its source chain matches the separate expert-map lookup. The trace does not include the router, pooler, attention, rank collective or full serving model, and does not isolate critical-path duration. Its other generic `unrolled_elementwise_kernel` row is **not** attributed to mapping or claimed removable here.

| Physical T | Index row, µs | Share of component profile | Saved logical bytes if packed output is replaced |
|---:|---:|---:|---:|
|1|1.8|3.2%|64|
|8|1.9|1.7%|512|
|16|1.9|1.0%|1,024|
|32|2.4|1.3%|2,048|
|128|3.2|0.7%|8,192|
|256|3.3|0.5%|16,384|

For K8, ignore the common global-ID/FP32-score outputs. Old work writes an unused packed array (`4TK` bytes), then the mapping kernel reads global IDs (`4TK`), reads map entries (`4TK`, often cached), and writes local IDs (`4TK`). New work reads the same map entries and writes the local output in the router (`8TK`). The difference is **8TK=64T logical bytes** and one `4TK=32T`-byte allocation. No lower-bound HBM savings follow from these cached requests. The valuable change is one launch/dependency and unnecessary packed-score conversion per layer, not bandwidth compression. The new map loads/pointer checks may lengthen the router and consume part of that saving.

Even granting zero extra router cost, removing the profile row deletes only0.5–3.2% of this component capture. The two Marlin GEMMs still account for83.2–96.8%. Official component load weight60 and the source comment about two BF16 endpoint layers motivate a conditional sixty-block repetition:1.8–3.3us per layer would sum to108–198us **if** every row were serial and equivalent. That is not measured target-step latency, not multiplied by eight ranks, and not an end-to-end speed prediction. Actual checkpoint layer eligibility, overlaps, slowest-rank behavior and DFlash acceptance must be measured. There is no evidence for a large model gain from this operation alone.

## One falsifiable component experiment, if root authorizes it

A single-H200 graph test can validate this exact boundary without weights or eight GPUs: run the actual pinned JIT router plus original Torch mapping against the dual-output router, with fresh FP32 logits/bias and all eight EP maps in sequence. Include T0/1/8/16/31/32/128/256 and a nearby non-power-of-two batch, ties/extremes, all-local/all-remote/mixed selections, unchanged capture buffer, and generic fallback/valid-token cases. Require **bit-identical global IDs, FP32 scores and local-ID outputs** to the original two-operation chain for the supported fast domain. No arithmetic change justifies a looser score tolerance. Source finite map checks can cover all256 IDs×8 maps cheaply, but do not replace GPU stream/graph tests.

Time the complete router→map chain with CUDA graph replays and alternating parent/candidate order on identical changing inputs, not isolated router timing. Verify exactly one indexing launch disappears, the router backend and stream dependency remain selected, and only the unused packed production output is disabled. Use the existing bounded runner infrastructure under a separate reviewed plan/fresh guard; this scout authorizes no allocation or new generic platform. An official MoE-only check begins **after** routing and cannot certify this fusion or measure its net gain. A later full caller/serving check is required.

Suggested pre-result screen, subject to root approval: at least5% geometric chain latency reduction across production verify T8/16/32/128/256, no tested production case over2% slower, exact outputs and no capture/concurrency failure. Kill the attempt if the real serving backend does not select the new path, if dual outputs add another launch/copy, if score/global-ID semantics change, or if the net chain gain fails. Do not retune warps/top-k sorting or switch routing arithmetic to rescue it.

Only after a useful component result should a separately attributed patch be tried on the exact eight-H200 server: global capture, rank mapping, BF16 collective/unpool boundary, repeated DFlash graphs, ordinary prefill and required concurrency1/4/16/32. Model-output/TTFT/throughput criteria remain mandatory. Do not combine this with the existing alignment/tile patch or unresolved finalizer changes merely because they touch the same files.

## Deployment and evidence pins

All contemplated files are under the allowed `python/sglang/` source tree, including the existing JIT `.cuh`; no dependency, model loader, precision, KV format, expert count or collective change is needed. The pinned checkout is Apache-2.0 and its notices must remain. This is a feasible small source patch under the retained serving diff limits, not an official eligibility decision. The current staged finalizer remains independently owned and unmodified here.

Inspected hashes:

- model`muse_spark.py`: `7e5018b3cb2ef41ee6cf09407e91e28dde069e5a33719ea58beb0d7137087355`;
- current Marlin caller (contains the older separate alignment patch): `bf166896bf1b2184abd27cf49f222a72ee118ffe116049f050a1cf0f1c0e960b`;
- fused routing Python: `1ab541227d1f768379424e8ba644afc7488cde1ca691cdc5680d53809fc96952`;
- JIT Python wrapper: `9d88c80d5351c0aba45fa671f49d83026b0891ef3ffa9d4271d062f0992d0e99`;
- CUDA routing header: `0a965acc078e4d8fd97349f98a1f386be75baed7745a49d031cd5c1111e04ef0`;
- v22 raw receipt: `9ea87f08e6b475c490db0540b2f5ec8ff95b7c2ca6e3c298ac9f20cc26c1844e`;
- v22 profile JSON: `999a1d94ddc9cbb8496868a8816ab468f74fe6e673f333cff9d35d327d8138c6`.

Other inspected boundaries are already fused: per-head norm/gate, QK norm/RoPE where supported, residual/post/pre-norm, and eligible unpool gather/boundary. Moving Marlin pre-expert norm before the router join is another scheduling possibility, but has no isolated timing evidence here and removes no work; it is not bundled into this proposal. A broad new serving speedup beyond the measured component scope remains unsubstantiated.
