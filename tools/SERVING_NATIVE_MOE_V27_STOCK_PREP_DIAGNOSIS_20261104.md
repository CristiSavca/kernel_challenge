# V27 timeout: unresolved stock interval, not a demonstrated repacker failure

Run `20261104T213729Z-f6cf2030` reached the `stock_prepare` checkpoint at **88.045482834 seconds**, then ended with remote exit `-1` under the 280-second command deadline. The controller recovered the checkpoint and terminated the sandbox. Stdout and stderr are empty. No correctness case or timing completed; V27 device validation remains unresolved.

The checkpoint establishes that actual rank-3 native preparation, CUDA extension initialization and installation **returned**: one PRMT bank of 905,970,816 bytes was registered, candidate raw parameter slots were removed, and one workspace existed. It does not establish native forward correctness. It also rules out describing the observed terminal interval as an unfinished native compile.

Evidence: [checkpoint](../experiments/modal/20261104T213729Z-f6cf2030/result-checkpoint.json), [manifest](../experiments/modal/20261104T213729Z-f6cf2030/manifest.json), and [compact source/hash record](../experiments/iterations/20261104-native-serving-v27-stock-prep-diagnosis.json). Checkpoint SHA-256 `0735f37d069b6c10eb5ff5242f10c40e84d91400e5e2da21caf7ae0cb5e8e50b`; frozen executed worker `d5e789f64c4fa0f74daa38117fddbb1d5a70e49b103b69a43522bac067edf3a8`.

## What the coarse marker actually covers

The frozen worker writes `stock_prepare` before creating the stock layer. Its next save is **after two initial stock forwards and their exact comparison**, not immediately after preparing weights. The unresolved interval therefore contains, in order:

1. `Layer(3)` construction, canonical expert map allocation, six independent raw-weight clones and parameter registrations.
2. The unchanged `prepare_moe_nvfp4_layer_for_marlin` call: two 32-expert code repack loops, two 32-expert scale conversion loops and two global-scale conversions.
3. An explicit `torch.cuda.synchronize()` and the cooperative budget check.
4. Generated T=32 mixed-route input creation.
5. Two unchanged `_forward_marlin` calls, including pre-norm, local mapping/alignment, two GEMMs each, activation, original post-expert finalizer, and the first output clone.
6. The exact stock-repeat comparison and `baseline_stability_checked` checkpoint, which was not recovered.

Thus neither “stock repacking hung” nor “first forward compiled too slowly” is established. If the explicit synchronization returned after the 240-second cooperative deadline, the following budget check should have produced an incomplete checkpoint. The saved running state is consistent with an unfinished earlier call, or with reaching a later blocking stock-forward call before that budget check had expired. It does not distinguish those cases.

## Source-backed possibilities

The [original preparation functions](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/quantization/marlin_utils_fp4.py#L300) make **64 repacker calls** and **64 scale-processing calls** at this exact 32-expert geometry. The first [GPTQ repacker call](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/quantization/gptq_marlin_repack.py#L18) can invoke `load_jit`; its zero-argument factory is cached, so the source does not imply 64 independent builds. Transposes, stacks and scale conversions still execute per expert. The nonnegative-scale `.all()` check also forces host observation; host return timing cannot be treated as pure GPU execution time.

Even if preparation completed, the first stock forward can independently build the [Marlin MoE GEMM module](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/moe/moe_wna16_marlin.py#L21). Its factory key here is `(torch.bfloat16, is_ep=True, has_bias=False)`; both matrix dimensions are runtime parameters to that module. The [activation wrapper](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/activation/activation.py#L22), normalization, alignment and original Triton finalizer can also have first-use work. None has a retained completion marker in this run.

The pinned [JIT loader](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/utils/compile/loader.py#L121) checks cache entries, acquires an interprocess build lock, builds, loads the library and publishes it. Its [Ninja runner](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/utils/compile/ninja.py#L138) captures compiler output and has an internal 1,800-second timeout, longer than this sandbox. Empty outer logs therefore do **not** prove there was no compilation. No retained build manifest, process stack or compiler transcript identifies which loader stage was active. GPU synchronization, resource contention and cache/lock behavior also remain hypotheses rather than diagnosed causes.

A separate prospective reader defect is confirmed: default `ast.dump` on local Python 3.13.4 omits empty fields, changing the three extraction fingerprints despite identical source bytes. I independently recomputed all three with `show_empty=True` and reproduced the retained remote fingerprints exactly. The partial-result reader returns false before this completed-result comparison, so this defect did **not** cause the timeout. A complete result would nevertheless have been rejected by the current host default serialization. The future probe must record runtime Python version and use explicit-empty serialization (`show_empty=True` when supported, ordinary 3.12 fallback) or decorator-inclusive verbatim selected-source fingerprints. This is a protocol repair, not a numerical or kernel change; the failed worker remains frozen.

## One proposed next probe — plan only

Run **only the unresolved stock interval once**, on the same pinned image and original source, using the same generated 32-expert rank-3 weight fixture and T=32 norm/input configuration. Do not compile, prepare or call the native candidate. Do not replay the 38-case suite, create graphs, change dependencies/build flags, force caches, or increase the existing 300-second sandbox / 280-second command / 240-second cooperative bounds. A fresh guard decision is required before any future allocation.

Add durable before/after checkpoints around layer creation/cloning, original stock preparation, the existing synchronization, input creation, each of the two stock forwards, and their exact comparison. Within these calls, a small read-only `sys.setprofile` observer may record call/return events for the exact original preparation functions and existing JIT factory/load/build code objects. It must not replace installed functions or alter arguments, results, cache state, flags or arithmetic. Record at most 512 compact events in a 256-KiB checkpoint, retaining current nested stage and counts even if event capacity is reached. One bounded main-thread stack snapshot during a long call can distinguish a Ninja subprocess wait, build lock, library load or Python-level synchronization boundary; it must perform no CUDA operation. Do not add per-expert device synchronization: the existing final synchronization remains the completion boundary.

The probe is falsifiable:

- Preparation-end **and** synchronization-end markers falsify “preparation has not returned” for that fresh run. A later first-forward/JIT entry places its delay after preparation.
- A repacker/JIT factory that returns promptly while repeated repack or scale calls remain active weakens the cold-build explanation.
- An entered Ninja build with a retained subprocess-wait stack supports host-build attribution. No compiler return means no claim of compiler success or failure.
- Both stock forwards returning with an exact repeat match completes this diagnostic. If the whole interval finishes promptly, the original failure was not reproduced; that is not authorization to rerun V27 automatically.
- At any budget limit, preserve the last entered/returned stages and stop. A still-ambiguous checkpoint remains inconclusive; do not compensate with a blind timeout increase.

Fresh cache state and provider scheduling may differ, and Python instrumentation adds overhead. These observations can localize a future stall but cannot retroactively prove the original cause or measure a serving speedup. This task changed only this note and its compact evidence JSON; the frozen worker, candidate, runner, budgets and receipts remain unchanged.
