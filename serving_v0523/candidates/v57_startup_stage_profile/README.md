# V57 CUDA graph warmup stage profile

V57 adds a private startup diagnostic to V50 compact expert outputs. It passes the original CUDA graph warmup callback through once per invocation. Only the second existing eager warmup at the largest prefill and target verify buckets is timed, on TP rank 0. The process records each phase at most once, including after graph recapture. It creates no extra forwards or tensors that survive that warmup. It changes no model inputs, dispatch, collectives, capture graph nodes, serving flags, or request behavior.

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- V50 parent SHA256 `7ba1708e049dbfae89eac613efe0bdeb9b7c8ccbcf0c11859ccad49f7f4441ae`
- Patch SHA256 `b6cb1eec3ef76f6d6f771f65088f0b8d8afb82512588205bdaa70706e11e2d75`
- Patch size 41,877 bytes across nine allowed files
- Use `swa-full-tokens-ratio=0.25` with standard prefill and verify graph backends. This is a private-check candidate.

The startup log should contain up to two `Muse startup stage profile {…}` lines. Extract them with `jq -r '.result.startup_log.text | split("\n")[] | select(contains("Muse startup stage profile ")) | split("Muse startup stage profile ")[1] | fromjson' RUN.json`. `gpu_ms` is the CUDA event span on the forward stream. It includes GPU waiting, collective completion and CPU-induced launch gaps. `enqueue_ms` and `wall_ms` show host time up to the last event record and its completion, respectively. Stage sums cover the same event span, aside from independent decimal rounding. Each profile uses a single terminal event synchronization. The router measure includes its existing side-stream join. The prefill runner warms the transformer body and the verify warmup invokes its normal target forward. Counts expose the observed number of each stage. The prefill log also reports rank 0's dense-layer and local-route counts using the V50 host counts already produced by the forward.

## Scope of the observations

The existing capture tensors are synthetic. Prefill starts from a zero-filled token-ID buffer with zero cached prefix, and verify uses the runner's short dummy sequence lengths. Each log reports its actual maximum sequence and prefix. If all 16,384 prefill rows select the same top eight experts on a layer, one rank can see 0 to 8 active experts and 0 to 131,072 local routes. The nominal EP8 average is 16,384 local routes, with the actual request distribution spread across more experts. Rank 0 may have no local computation or may wait for slower other ranks. These timing samples cannot establish real-request expert throughput, long-context attention cost, or a serving bottleneck. They can isolate startup stage structure, projection and communication spans, and residual warmup overhead.

Each selected shape has run once before profiling. The timing may still include one-time setup and event instrumentation overhead. The diagnostic never profiles steady-state requests or compares two model paths. A clean V56 sequence-parallel check has greater potential to improve the serving score and should take the scarce score-development check ahead of this startup diagnostic. Keep V57 as a reserve for localizing startup cost when a suitable private slot becomes available.

## Local validation

The exact patch applies and reverses on the pinned source, preserves file modes, and passes patch path and line limits and all Python compilation checks. Each changed V50 source has the original executable AST after removing diagnostic calls, wrappers and imports and normalizing two return assignments for stage labels. A CPU mock exercised the actual event wrapper, including second invocation only, one terminal wait, recapture suppression, guards, shape and route statistics, stage aggregation, result identity and cleanup on a forward exception. CUDA event behavior, log retention and startup timing remain unverified on the judge.
