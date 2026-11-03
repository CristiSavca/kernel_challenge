# Dense projection experiments — updated 2026-11-03

## Current candidate: v003 profile hybrid

Active source: `submission.py`, identical to `candidates/v003_profile_hybrid.py`.
Private check: **ce713efc5e9b44d8**.
Raw source SHA-256: **b2e4fd2caaad6f68fd0022af15c95beb60935b180cad41632c824b53bcabe53b**.
The helper froze these bytes before upload in `experiments/candidates/muse_bf16_proj/` and registered the receipt in `experiments/runs/ce713efc5e9b44d8.json`.
Latest observed status: received. No new correctness or speed claim is established yet.

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

Local AST parsing passed. This Mac has no CUDA runtime. Remaining risks are the real GPU result, larger-router occupancy and memory traffic, strict router error tolerance after changed grouping, and variance under graph replay/cold-L2 timing.

## Source reproducibility history

`candidates/v001_splitk.py` preserves the local v001 source, SHA-256 `eed1a2f6b8122617194d0f7163accd8a1e1fabc09791651804dbd824c3e4e4d8`. It was copied 19 seconds after the original check, with no intentional source edit in between. The server reported a different `content_hash`, `f1667fba62e225b06f1fc75e9b0c304a2c34fafb00b935167055fd61465bb841`, so queued-payload identity was not cryptographically verified. A loopback-only mock confirmed the CLI sent source unchanged; common newline/header/comment/JSON variants did not explain the discrepancy. Keep that caveat with the historical run.

`candidates/v002_splitk.py` was proposed in October and never submitted. v003 supersedes its tuning plan using the actual profile.

## Resume

Use `python3 tools/experiments.py status ce713efc5e9b44d8` from the project root. Inspect correctness, all load scores, and particularly router 128/256. Keep checks private until a passing candidate has a useful measured gain. Compare changes against the reported uncertainty interval. No scored submission has been made by this projection task.
