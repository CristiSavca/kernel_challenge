# Open Kernel Challenge notes

Refreshed against the official site, terms and API on 2026-11-03. Re-read the live rules and limits before using a day's runs.

## What wins the prize

The $100,000 is the total serving-track cash pool: first $50,000 plus an H200, second $30,000, third $10,000, fourth $3,000, fifth $2,000, and sixth through tenth $1,000 each. Kernel boards have no cash prizes. The overall kernel board's top 50 get serving-queue priority when the kernel phase closes.

The top 50 are a queue-priority group, not a qualification cutoff. Terms section 6 allows entering either track independently; no minimum kernel rank, speedup or number of solved problems is stated for serving entry. Serving requires its own compliant SGLang patch.

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

Live limits on 2026-11-03 are 20 scored submissions and 40 checks per kernel per UTC day, across all five problems. The prose docs still contain older caps, so use `okc problems --json` for the current account's real allowance. Caps reset at 00:00 UTC, now 7 pm Eastern.

Use `okc check <file> --json` for experiments, retain its profile and job ID, then submit a passing candidate to the board. Repeating identical code returns its previous result. A kernel run has 10 minutes for projections or 15 minutes for the other inspected tasks, including compilation. `prepare(weights)` is untimed, capped at 300 seconds. Serving allows 1 scored and 3 short check runs a day; the scored run can take up to 90 minutes.

## Implementation guardrails

Updates verified November 3: source must be readable, with no embedded/prebuilt GPU binaries or compressed, encoded or obfuscated code (November 2 clarification). The latest kernel guidance permits approximate math accurate to BF16, BF16/FP16 split partial sums, and FP16 attention accumulation, while preserving the required computation and checking bounds. Consult the live restrictions before using any newly allowed precision behavior. Our two submitted candidates retain their originally validated precision behavior.

The live llms.txt kernel restrictions also explicitly permit BF16 **or FP32**
activations wherever SGLang keeps BF16. This permits the reviewed CUDA experts'
fused FP32 epilogues. The serving restriction separately still says BF16; do not
assume a kernel-track permission applies to serving. Recheck the live serving
API before export because its published patch size and prose have differed.

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

## November 3 public-source and Modal setup

The RTX 5080 PC is not yet set up. The user authorized using free Modal H200
credits. Modal 1.6.3 is installed in a 53 MiB project venv and authenticated to
cristisavca44. The first H200 request was denied before allocation because the
account lacks a payment method. The account dialog says a refundable $0.50
verification unlocks the full $30 monthly Starter allowance; current credits
are $1. The user must complete that payment step. CLI usage remains $0 and the
failed Modal app is stopped with zero tasks. No paid plan was enabled.

The bounded runner and exact next commands are in tools/MODAL.md. It has four
passing CPU lifecycle tests; live CUDA validation is still pending access.
Selected public candidates, results and original licenses occupy 1.5 MiB under
third_party; hashes were verified. tools/PUBLIC_RESEARCH.md records provenance
and excludes the encoded dense source. No full third-party app build, model
weights or local CUDA image was downloaded. No project cleanup was needed in
this pass; free space observed 41 GiB initially and 45 GiB later reflects other
concurrent disk activity, not removals by this task.

## Native-entry evidence review (November 3 evening)

All four native candidates passed their official private checks with no flags,
rejections or runtime errors. An independent review matched all 43 numerical
cases and 60 timed cases to the TASK declarations, including parameters and
order. Recorded preparation times stayed within 300 seconds (MoE 140.60 s,
projections 75.56 s, decoder 113.66 s).

Passing-check snapshots, active submissions and scored-upload snapshots have
identical SHA-256 hashes for each problem. Canonical server content hashes also
match between each private check and its scored receipt, including decoder
`4b216098993e4112`, now confirmed in mode `run`. Complete upstream MIT/Apache
notices are present; original NVIDIA BSD text is retained with CUTLASS source.
All 18 official evaluator files remain byte-identical to the initial commit,
and applicable local evaluator hashes match those reported by the checks.

Private speedups: attention 2.6425×, experts 2.3605×, projections 1.4774×,
whole decoder 2.1541×. The new scored jobs are respectively
`8f9e5971f95841a0`, `3c15b408fdb04c44`, `3caa02bd72da4c49` and
`4b216098993e4112`. Their outcomes were pending at this initial review;
completed scored results are recorded below.

## Completed scored submissions (November 3, 9:20 pm Eastern)

Fresh official status responses confirm all four native entries passed:
attention **2.611970×**, experts **2.356104×**, projections **1.486827×** and
whole decoder **2.150186×**. All 43 numerical tests and 60 timed cases passed
with no flags or rejections. Independent review matched the required case
parameters and weights, source hashes and evaluator snapshots. Preparation
times were within 300 seconds: experts 137.81 s, projections 73.86 s and
decoder 114.46 s. All 18 official evaluator files remain unchanged.

Every native job is confirmed as our current entry on its official board.
At 2026-11-04 02:20 UTC the live ranks were attention 272, experts 335,
projections 341, decoder 386, and overall **386 of 1,946** with four problems
solved. These are provisional positions, below the top-50 priority cutoff.
The kernel results do not establish full 8×H200 serving throughput or prize
placement. The separate serving patch remains an unvalidated draft.

All 32 experiment source snapshots match their recorded hashes, all recorded
jobs are terminal, and the owned result watcher exited successfully. Exact
results and profiles remain under `experiments/`; the compact final audit is
`experiments/native-scored-audit.json`. No paid Modal GPU was used.
