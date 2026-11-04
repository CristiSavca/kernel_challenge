# V26: immutable split partials and ordered FP32 owners

This isolated implementation starts from V25 source SHA256 `3a8c973b737dd074bcf5115f7418f5f0b34139fd4f9f637410ad1765ccbe050d`. V25 job033a2d2eca374d82 passed correctness but failed its fixed component screen: M1 normalized latency was1.1363×stock. V25 remains closed. V26 changes the reduction algorithm for every token count; it adds no shape fallback, dispatch threshold or precision exception.

Files are confined to `serving/candidates/native_serving_v26/`. `parent.py` is the immutable V25 source; `build.py` records exact reversible replacements in `edits.json`; `manifest.json` binds the generated `component.py` and builder. Active submissions and the serving checkout are untouched. The distinct CUDA symbol is `native_serving_v26<false/true>`; resource rows use `V26R`.

## Work changed

V25 accumulates each split-unit scalar using an integer atomic read plus a compare-and-swap loop, because ordinary global FP32 reduction flushes subnormals while stock Marlin's FP32 reducer does not. V26 writes each partial once into a private slot, then one owner loads all contributors and combines them using explicit `add.rn.f32` without FTZ.

Unchanged: the single905,969,664-byte WGS weight bank, lossless packing, KPERM, FT64 jobs,256-row/K256 work units, three joined weight/X slots, cost tables and stage ranges, per-partial MMA order, weight and activation requests, producer scheduling, cooperative launch, and serial Workspace API. Both BF16 epilogue roundings, exact Marlin scale LUT/normal PRMT path, narrow SiLU product FTZ, complete local-ID mask and original stock finalizer remain unchanged. No input computation moves into prepare.

This is not whole-K tiling: it retains load balancing across132 CTAs. Whole-K256-row units would provide only32A/16B units per job and could underfill the device for the smallest jobs.

## Payload ownership

Each physical CTA owns a contiguous cost-derived stage range `[g0,g1)` per phase. Only its first and last units can be partial. A short range entirely within one unit has one payload. Whole units write their final BF16 result directly.

The slot edge is `unit == (g0 >> 4) ? 0 : 1`. Thus a distinct trailing partial uses edge1 even when phaseA's inherited rotation executes it first. A same-unit short range uses edge0 once. Empty ranges write nothing. The full payload index is `(phase, CTA, edge, WG, n-tile, lane, component)`; every private payload remains immutable until the complete grid finishes. No payload is cleared during reduction or replay. The entire scratch allocation is still zeroed once at Workspace construction, as in V25.

The owner retains `pd.slot`, the CTA containing the unit's first stage, and computes the last CTA from the existing exact inverse `wgs_cta_of` on the last stage. It traverses only that inclusive span. Starting from the exact cost-derived `g0`, it computes each `g1` with the inherited helper and carries `g1` forward. Empty and nonintersecting ranges are skipped before any payload address is loaded. The same edge rule selects each contributor exactly once.

The reader processes two n-tiles at a time, starting eight FP32 accumulators at+0 and summing contributors in increasing CTA/original K-stage order. Reconstructing ranges repeats for each two-tile chunk and is a real added cost. The compiler's register/local-memory behavior is unmeasured. No unsupported claim says removing atomics must improve latency.

## Publication, progress and reset

V25's complete publication block is retained byte-for-byte. Each contributing warpgroup reaches the existing128-thread barrier after its stores. PhaseA retains the last-arrival owner elected by the acq_rel stage-count atomic; phaseB retains the designated first-stage owner, which acquires completion of all other stage counts. The target remains16 stages. The owner consumes all payloads before the inherited counter clear and phaseA readiness release.

This preserves the existing phaseA rotation and phaseB progress dependency. It does not introduce a fixed phaseA owner or a new grid barrier. The inherited cooperative-only admission remains required. Per-call grid completion and the explicit serial Workspace contract prevent payload reuse while an older call is reading. Concurrent use of the same workspace remains unsupported; separate workspace/capture/multi-rank deployment still needs runtime validation.

There is no global-nj0 early kernel return: the inherited V25 zero ranges, full mask publication and all participants' joins remain. No source path reads private partials for an empty global job list.

## Precision and costs

Every partial MMA result has the same precision as V25. Explicit non-FTZ FP32 additions preserve the subnormal domain that required V25's CAS path. Addition association changes from CAS arrival order to deterministic CTA/stage order, so bit identity is not claimed. Actual GPU/reference correctness remains mandatory, including the unchanged BF16/activation boundaries and tolerances.

Fixed private payload storage is **41,943,040B (40MiB)**. The original count/doneAs tail remains13,824B. All device accessors, host offsets and Python allocation lengths move that tail consistently. Total Workspace is **129,005,088B (123.028839MiB)**,20MiB more than V25. The weight bank remains864MiB. Production model-resource ownership and full-model/capture memory remain unresolved; this is one serial component-state workspace.

For a conditional uniform one-job/NTT1/132-CTA scenario, exact inherited ranges produce152A+136B partial contributions,2,048 scalars each. V25 therefore executes at least1,179,648 scalar integer atomic operations before CAS retries. V26 substitutes4.5MiB of logical payload writes+reads and589,824 ordinary FP32 additions. These are source-operation counts, not the actual saved run's route histogram, DRAM transactions or a latency forecast. Private-store traffic, range queries and owner work may erase the benefit.

## Validation and fixed decision

The focused CPU verifier reverses the entire candidate into immutable V25, checks exact preservation of the packing/precision/host contract, and compiles the actual range helpers, rotated phase traversal, private stores and owner loop into a CPU harness. It tests job counts0/1/2/3/8/31/32/33/64/65/88, n-tiles1..8, multiple legal grid sizes, and separate oversized-grid helper-only cases that create empty intermediate ranges. Those cases do not expand the actual G<160 host contract. Poisoned/read-authorized payloads reject wrong-edge and unused-tile reads. Native negative controls corrupt edge selection, range carry and empty-range filtering. Publication checks model stage counts and source identity; they do not emulate GPU memory ordering or full async progress.

The fixed official screen remains: all5 correctness tests,6 benchmarks and6 profiles clean; all-shape geometric-mean speed>=1.15 against the same-run stock baseline; every candidate/stock normalized latency<=1.02; intended V26 core and original finalizer path. M1 needs at least10.24% less normalized component latency than V25 just to meet stock+2%. This is a hurdle, not a prediction. No retuned threshold, automatic retry, public promotion or full-serving gain claim follows. Root alone owns the official check.
