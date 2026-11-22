# V100 tree verification with larger physical KV pages

V100 keeps V99’s native eight-row speculative decoder and lets eligible full-attention prompt chunks use Hopper’s paged tensor-memory accelerator. Larger pages can improve prompt throughput at high concurrency. They also change prefix-cache granularity, and committing a branched decode now costs up to two GPU launches. The net gain requires an official run. [Full export](patch.diff) · [Six-file change from V99](delta.diff)

## Preserving decode and prefix-cache semantics

V99 can permute individual physical slot IDs after target verification because each token occupies its own page. With 160-token pages, the attention backend derives every slot from a physical page number and its in-page offset. Reordering individual IDs would make later reads disagree with the request table.

V100 leaves that table intact. Each GPU block handles one request, one layer’s K or V buffer, and a slice of its channels. It reads all the accepted source rows that need to move, synchronizes its threads, and writes those bytes into their original sequential destination slots. This handles overlapping sources and destinations even when a tree branch crosses a page boundary. Full and sliding-window pools use their own physical addresses, including the live full-to-sliding-window mapping. The root stays in place. A wholly accepted linear chain moves zero KV bytes.

The draft regenerates its KV in those sequential destinations from the verified hidden rows in commit order. The scheduler publishes the new sequence length after the target copies and draft writes. Before graph capture, each tensor-parallel rank checks its local pool and warms the copy kernel with zero committed rows. A group vote selects the ordinary linear fallback if a rank cannot use the new path. The one-token-page path retains its existing slot permutation. [Cache ownership](kv_layout_review.md) · [Worker and two-rank setup tests](worker_check.json) · [Independent review](independent_review.md)

## Why 160 tokens

The pinned FA3 implementation can use paged tensor-memory loads for eligible long, full-attention FP8 prompts when physical page size is a multiple of its 160-key tile. Short queries and sliding-window layers keep their applicable native dispatch. At chunk size 16,320, each full prompt chunk occupies 102 pages. The prefill graph ceiling stays at 16,384. Two allocator and metadata changes from V60 support page widths that are not powers of two. The tree verifier uses quotient and remainder to locate each key. [Pinned FA3 dispatch](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_api.cpp) · [Page geometry](../v60_fa3_paged_tma/README.md)

Use V99’s flags with `--page-size 160`, `--chunked-prefill-size 16320`, `--max-prefill-tokens 16320` and `--cuda-graph-max-bs-prefill 16384`. All source paths and launch overrides passed the published local restrictions. The patch is 232,061 bytes across 41 allowed files. Its SHA-256 is `1775f98fc566f1c9bce3daa1e64dcb81e5e67c46927fa4213a099db2c0f88170`. [Validation](validation.json) · [Serving rules](../../RULES.md)

## Correctness checks

- The production compactor passed 48 cases in Triton’s CPU interpreter with 7,286,016 bytes checked. Independent tests checked 64,880,640 bytes, exercised all eight commit lengths over 32 consecutive steps, and compared full physical pages after shared-prefix reuse and request-slot reuse. [Interpreter](kv_interpreter.json) · [Independent oracle](independent_kv_commit.json)
- Tree attention produced identical results to the page-one reference for 552,960 output values across fragmented page layouts. A second oracle compared 4,608 query rows to dense full and sliding-window attention, including poisoned evicted pages. Maximum relative L2 error was 0.001944. [Page-one comparison](independent_paged_attention.json) · [Dense reference](paged_attention_cpu.json)
- The allocator and metadata kernels passed 792 allocations, 396 draft extensions and 1,584 target decode or verification cases over 11 page widths. All 26 offline Hopper specializations compiled without spills. Actual GPU execution, output agreement, cache hit rate, memory and latency are still pending. [Allocator results](allocator_metadata_cpu.json) · [SM90 compiler results](sm90.json)

- Five tests ran against SGLang’s native Rust radix cache with 160-token pages. They checked exact physical slot ownership across shared branches, host eviction and reload, cache-event hashes, and reuse after sliding-window eviction. For a 2,048-token window, the allocator retained a page-aligned tail of 2,080 tokens. The combined native suite passed 947 tests with one upstream test marked ignored. [Rust fixture](radix_page160.rs) · [Native test results](radix_page160.json)

## Measurement order

The V64 scored baseline peaked at 20,547.4 raw tokens/s/GPU. Its four paired stock throughputs have a geometric mean of 9,996.0. On that same stock curve, a 4× score requires a highest candidate rung of at least 39,984 tokens/s/GPU, since no interpolated throughput can exceed the candidate’s highest measured rung. This is a necessary threshold. The four individual stock-speed points determine the score. [V64 result](../../runs/abd217baf0ed49fd.json) · [Calculation](frontier_budget.json)

V100c isolates paging and the associated chunk settings on the scored V64 model at 32 users. V100c remains the paging-only control. The November 19 runner now selects [V111](../v111_paged_prefill_verify/README.md) at 32 users while V87 is queued or running and V64’s scored receipt passes. A passing V87 selects the matched V89 decode experiment. V100 itself is frozen separately. Compare V100c with the V64 32-user rung for raw throughput, prompt and later-turn latency, cache hit rate, correctness and peak memory. Larger pages can introduce fragmentation and shorten a reusable prefix at the page boundary. [V100c control](../v100c_v64_paged_prefill_control/README.md) · [Armed runner](../../runs/nov19_decode_runner.launch.json)
