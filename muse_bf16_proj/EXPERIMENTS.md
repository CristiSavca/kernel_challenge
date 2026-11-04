# Dense projection experiments — updated 2026-11-03

## Current candidate: v003 profile hybrid

Active source: `submission.py`, identical to `candidates/v003_profile_hybrid.py`.
Private check: **ce713efc5e9b44d8**.
Raw source SHA-256: **b2e4fd2caaad6f68fd0022af15c95beb60935b180cad41632c824b53bcabe53b**.
The helper froze these bytes before upload in `experiments/candidates/muse_bf16_proj/` and registered the receipt in `experiments/runs/ce713efc5e9b44d8.json`.
Official H200 result: **passed**, 9/9 numerical cases and all static checks.
Measured speedup **1.132050×**, 95% interval **[1.115069, 1.149031]**,
score **4,267.30 µs** versus the normalized SGLang baseline **4,830.80 µs**.
No flags or rejected checks. This exact snapshot is eligible for scoring.

The prior profile had clear winners and regressions. v003 keeps one-token GEMVs for qkv, pooler, sliced unpooler, and router; retains the split-K pooler for 2–16 tokens; and uses SGLang's own `_bf16_gemm_dispatch_impl` for the other BF16 paths. The FP32 router keeps its permitted BF16 high/low split, with eight K partitions at every batch size instead of a single partition above 32 tokens. That specifically addresses poor occupancy in the 128/256-token router measurements.

Selecting only prior measured wins and stock fallbacks models approximately **1.108x** against SGLang using the old run's times; this is an estimate, not v003's measured score. v003 additionally experiments with the larger router's partition count.

Only the pooler transpose and router high/low transposes are now packed. Other BF16 transposes from v001 were unused after dispatch changes and have been removed from preparation.

## Historical check: 43cb543b57c64b29

Saved profile: `profiles/submission.43cb543b57c64b29.txt` (also JSON/log).
Result: **9/9 tests passed, 0.950503x**, 95% interval **[0.936246, 0.964761]**, score **5,082.36 µs**.

Useful per-shape findings:

| Path | SGLang | Candidate | Ratio |
|---|---:|---:|---:|
| qkv, 1 token | 13.95 µs | 12.18 µs | 1.146x |
| pooler, 1 token | 20.91 µs | 18.99 µs | 1.101x |
| sliced unpooler, 1 token | 6.72 µs | 4.80 µs | 1.401x |
| router, 1 token | 10.40 µs | 5.62 µs | 1.851x |
| pooler, 8 tokens | 21.22 µs | 19.49 µs | 1.089x |
| pooler, 16 tokens | 20.91 µs | 19.31 µs | 1.083x |
| router, 8 tokens | 26.64 µs | 10.45 µs | 2.549x |
| router, 16 tokens | 14.69 µs | 10.10 µs | 1.454x |
| router, 32 tokens | 15.07 µs | 10.41 µs | 1.448x |
| router, 128 tokens | 16.22 µs | 57.55 µs | 0.282x |
| router, 256 tokens | 17.47 µs | 57.21 µs | 0.305x |

The o-projection at 1–32 tokens, unpooler at 16/32 tokens, and qkv at 32 tokens also regressed. The pooler at 32 tokens and qkv at 16 tokens were too close to justify keeping custom dispatch.

## Precision and graph audit

- Dispatch uses only fixed dimensions and the layer name. Every call reads fresh activations; nothing derived from inputs is cached.
- `prepare` retains/repackages only weights. BF16 weights retain every original BF16 value. Router uses original FP32 for one token and the explicitly permitted BF16 high/low residual split otherwise.
- GEMV, both router products, and partial reduction accumulate in FP32; BF16 projections round at the final output store. Router output remains FP32.
- Eight split-K partitions cover disjoint 64-channel tiles with stride 512, and the final reduction reads all eight exactly once. Fixed K=8192 gives 16 iterations per partition.
- Nonmultiple token counts are masked in Triton and handled by stock dispatch where selected. The single-token unpooler uses the original weight's correct K stride while restricting its N bound to the slice.
- Allocations have fixed shapes. No host synchronization, input-dependent allocation/control flow, harness access, timing changes, module mutation, network, or process calls appear in the submission.
- The SGLang fallback import is identical to the official baseline's import. The Triton kernels are unchanged from the earlier passing implementation; the larger router's grid changes its reduction grouping and still requires fresh validation.

Local AST parsing and the official H200 compilation/numerical checks passed.
The larger-router split-K change passed strict router tolerance and improved
both larger shapes. Timing variance remains relevant when comparing later runs.

## Source reproducibility history

`candidates/v001_splitk.py` preserves the local v001 source, SHA-256 `eed1a2f6b8122617194d0f7163accd8a1e1fabc09791651804dbd824c3e4e4d8`. It was copied 19 seconds after the original check, with no intentional source edit in between. The server reported a different `content_hash`, `f1667fba62e225b06f1fc75e9b0c304a2c34fafb00b935167055fd61465bb841`, so queued-payload identity was not cryptographically verified. A loopback-only mock confirmed the CLI sent source unchanged; common newline/header/comment/JSON variants did not explain the discrepancy. Keep that caveat with the historical run.

`candidates/v002_splitk.py` was proposed in October and never submitted. v003 supersedes its tuning plan using the actual profile.

## Resume

The November 3 result validates the hybrid. Router 128 tokens measured
**11.00 µs versus 16.29 µs** (1.481×), and router 256 tokens measured
**13.33 µs versus 17.50 µs** (1.313×), resolving the earlier 57 µs regression.
Other retained wins: one-token QKV 1.187×, pooler 1.124×, unpooler 1.318×,
and router 1.824×; eight-token router 2.549×. Stock-dispatched shapes mostly
remain close to parity (128-token pooler was 0.974× in this run).
The normalized aggregate time is 16.0% lower than the earlier private V1 run;
these are separate runs, not a paired V1/V3 measurement.

Score the frozen `b2e4fd2c...` source associated with this passing check, then
consider integrating the router and projection wins in the decoder. No scored
submission has been made by this result-analysis subtask.
# November 3 scored hybrid

The exact passing V3 snapshot was scored as `b5538a0dd25d46be` and passed at
**1.1303169997×**, CI [1.1133622447, 1.1472717547], score 4273.8453 µs.
The corresponding private check was `ce713efc5e9b44d8` (1.1320×).

## Goal continuation: readable lossless-compression candidate

`candidates/public_compressed_v4.py` is based on vjeux/okc-2026 commit
`f35a41ee1c13e316dabbf772830d47b23b453393`, with full Apache/CUTLASS notices.
All 14 native source strings match upstream; only an unused environment flag
hook was removed. Private check `8c992affff0e45b8` is pending.

CPU review covered all codec base/code/low-byte combinations, finite escape
reconstruction, fragment/split ownership and router hi/lo representation. No
active packing defect was found. Keep compressed SK2 routes at CL=1: an unused
CL>1 compressed epilogue drops peer escape buffers. Native barriers, numerical
accuracy, performance and the 300-second preparation limit need the official
GPU check. The existing scored hybrid remains active.

## Native V4 passed and submitted

Private check `8c992affff0e45b8` passed all nine numerical tests and 30 timed
shapes at **1.4774342558×**, CI [1.4552727419, 1.4995957696], with no flags,
rejections or runtime errors. This validates compilation, numerical accuracy,
and preparation within the official limit for this exact candidate.

The native router's paired speedups range from 2.399× to 5.239×. Small-batch
BF16 projections also improve; most large-batch non-router shapes are close
to baseline. Do not claim every individual shape improved: three ratios are
slightly below 1.0 (0.998×, 0.984× and 0.982×).

Exact frozen source
`27a0afe97d0a9b8cd11a58f243b44f2c375d84cec5cb7b742f451da45a239e30`
is now active and was submitted for scoring as `3caa02bd72da4c49` (pending).
Keep the existing compressed-CL=1 constraint; the unused compressed cluster
branch remains outside supported configurations.
