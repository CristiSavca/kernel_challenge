# Open Kernel Challenge notes

Verified against the official site and API on 2026-10-24. Re-read the live rules and limits before using a day's runs.

## What wins the prize

The $100,000 is the total serving-track cash pool: first $50,000 plus an H200, second $30,000, third $10,000, fourth $3,000, fifth $2,000, and sixth through tenth $1,000 each. Kernel boards have no cash prizes. The overall kernel board's top 50 get serving-queue priority when the kernel phase closes.

The entrant must be one individual, at least 18, with one account. The terms exclude sponsor personnel/family and jurisdictions or people prohibited by applicable law. Agent assistance is contemplated by the technical rules. The person must accept the rules through `okc login`; an agent must not accept them on their behalf. Submissions are licensed under Apache 2.0.

## Dates and a discrepancy to resolve

The live API reports:

- Kernel submissions close 2026-11-04 at 11:59 pm Pacific, or 2026-11-05 at 2:59 am Eastern.
- Serving opens 2026-11-05 at noon Pacific, or 3 pm Eastern.
- Serving submissions close 2026-11-19 at 11:59 pm Pacific, or 2026-11-20 at 2:59 am Eastern.

However, `terms.pdf` section 3 says the contest ends **2026-11-18 at 11:59 pm Pacific**, and section 5 says those official terms control discrepancies. The sponsor can modify dates. Until the organizer clarifies this mismatch, target a finished serving entry by **November 18**, not the live API's last day.

## Workloads and scoring

The kernel GPU is one H200 SXM. Serving uses a whole 8×H200 node. The image is pinned in `serving/RULES.md`; the source base is SGLang `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`.

The model is Muse Spark Open, with 256 routed experts, top 8, NVFP4 expert weights and BF16 activations. Under TP8/EP8, a GPU has 32 experts, 16 query heads and 2 KV heads of dimension 64. Scored decode loads use 1, 16 and 32 requests and 8-token verify blocks; contexts are around 80K. Attention alternates sliding windows of 2,048 tokens and full-context layers. The request cap is 32.

Each kernel problem's score is a geometric mean over load speedups. Within a load, shape times are weighted by how often they occur. Measurements alternate the submission and SGLang on the same GPU, capture CUDA graphs, flush L2, and report an interval of at least ±1.5%. Eager timing and tiny gains are unreliable. `muse_prefill_layer` has a separate board and does not count toward the overall kernel board.

Baseline geometric-mean times shown by the API: projections 4,830.8 µs/step, attention 3,766.3 µs/step, experts 13,448.5 µs/step, whole decoder layer 22,620.3 µs/step, prefill 387,403.9 µs/load. These component and whole-layer figures overlap; do not add the whole layer to its components to estimate production time. Expert work is the largest share at speculative decode loads of 16 or more requests.

Serving evaluates hidden multi-turn coding conversations at 1, 4, 16 and 32 concurrent users. Score compares throughput at stock SGLang's interactivity across the load curve. Every request must return 220 tokens, output agreement must remain near stock's agreement with itself, and time to first token must be at most 2× stock at each load for that load to improve the score. The top 25 serving entries are rerun at least three times on held-out data; median rerun scores decide final placements.

## Iteration budget

Live unauthenticated limits on 2026-10-24 were 10 scored submissions per kernel per UTC day; checks were 20/day for projections and experts, 10/day for attention, decoder and prefill. The prose docs still say 20 checks generally, so use `okc problems --json` for the current account's real allowance. Caps reset at 00:00 UTC, or 8 pm Eastern before the November clock change and 7 pm after it.

Use `okc check <file> --json` for experiments, retain its profile and job ID, then submit a passing candidate to the board. Repeating identical code returns its previous result. A kernel run has 10 minutes for projections or 15 minutes for the other inspected tasks, including compilation. `prepare(weights)` is untimed, capped at 300 seconds. Serving allows 1 scored and 3 short check runs a day; the scored run can take up to 90 minutes.

## Implementation guardrails

- Preserve original BF16/NVFP4 weights or use a lossless representation; do not pre-dequantize experts into persistent higher-precision weights. The router alone permits FP32 weights represented as BF16 hi + lo with FP32 accumulation.
- Keep the computation CUDA-graph compatible. Branch on dimensions/configuration, never on host-read input values. Compute each call from its fresh inputs.
- Keep BF16 activations and FP8 E4M3 KV cache. Attention queries may use E4M3 as stock FA3 does. Attend to every required key and value.
- Serving diffs may touch only source files or JSON under `python/sglang/`, excluding `python/sglang/srt/model_loader/`. No renames, binary or mode changes; patch ≤737,280 bytes and added lines ≤400 characters. Prebuilt `sgl-kernel` binaries are not rebuilt.
- Preserve all model parameters, top-8 routing, every norm, the 2,048-token window, and DFlash's released drafter. Request-derived prompt lookup/n-grams are allowed.

## First MoE opportunity

The baseline maps global expert IDs to local IDs (`-1` or `0..31`), then asks `moe_align_block_size` to process 256 experts. Passing the local expert count 32 after that mapping is a small first optimization to validate; it avoids work and padding for 224 impossible local IDs. Keep `-1` as the non-local sentinel and retain correct zero output for tokens with no local routes.

For the single-token path, direct packed-NVFP4 GEMV can avoid the general grouped-GEMM alignment and padding overhead. Preserve the exact scale order in `nvfp4.dequantize`, BF16 GEMM boundaries, per-expert norm before weighted sum, and graph safety. Start only at `bs == 1`; keep stock Marlin for larger batches until profiles justify extending it.

A second shape-only adjustment is to estimate routes per expert using the global expert count: `M * top_k / 256`, not `M * top_k / 32`. Non-local routes do not execute GEMMs on this GPU. The existing threshold then chooses block M of 8 through 128 tokens and 16 at 256/257 tokens, instead of 48 at 128 and 64 at 256. This preserves the arithmetic and can reduce padding, but skewed expert popularity makes its speed a measurement question. Validate the count-only change separately if the combined candidate regresses.

## Research scratch cleanup

Removed 13 individually inspected generated downloads from `/private/tmp`: the site JavaScript bundle, downloaded terms PDF, recursive SGLang tree JSON, and problem-spec/leaderboard JSON pairs for the five named problems. None had an open file handle, and `/tmp` was not a Git checkout. No project sources were removed. Logical bytes retired: 5,192,493. Observed physical free bytes rose from 153,818,632,192 to 153,823,834,112 (+5,201,920); concurrent disk activity can affect this observation. Research findings and source URLs are preserved here.

## Official sources

- Rules: https://openkernelchallenge.dev/rules
- Agent/CLI/API documentation: https://openkernelchallenge.dev/llms.txt
- Legal terms: https://openkernelchallenge.dev/terms.pdf
- Live problems, caps and dates: https://openkernelchallenge.dev/api/v1/problems
- Serving base, patch constraints and launch: https://openkernelchallenge.dev/api/v1/serving
- Problem specifications: https://openkernelchallenge.dev/api/v1/problems/muse_moe_w4a16 (replace the problem name for the other tasks)
- Overall board: https://openkernelchallenge.dev/api/v1/leaderboard
- Pinned source: https://github.com/sgl-project/sglang/tree/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f

Additional root research scratch retired: `/tmp/okc-site-20261024.js` (446,642 logical bytes), individually inspected with no open handle and outside Git. Observed free bytes: 153,143,562,240 before, 153,144,008,704 after (+446,464); concurrent activity may affect physical free space.
