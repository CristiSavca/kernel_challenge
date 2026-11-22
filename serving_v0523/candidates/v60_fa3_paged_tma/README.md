# V60 Paged FP8 attention with Hopper TMA

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`
- V48b parent `e59d8efcd1a544567686708cf4dd48361103e7640af95ff4d75ee556c4121a8f`
- V48b comparison run `8ebed2bced8942ff`
- Patch SHA-256 `8c152846270bd35aa7d58d8cfb6ea46f6cc5b81aa10f03a680bb4487000be79c`
- Patch size 12,095 bytes across six Python files

V60 keeps V48b's four files byte for byte. Two kernel files broaden
support for non-power-of-two KV pages. The allocator rounds the Triton
vector extent up while retaining each store's live-token mask. The
normal decode metadata kernel already divides by the actual page size.
Its wrapper now accepts any positive page size. Draft-extend metadata
retains its shift for power-of-two pages and uses page division for
other sizes, including floor division for signed slot values.

Use these flags for the private H200 check.

```text
--flag page-size=160
--flag chunked-prefill-size=16320
--flag max-prefill-tokens=16320
--flag cuda-graph-max-bs-prefill=16384
--flag swa-full-tokens-ratio=0.25
```

The scheduler requires positive chunk sizes to be multiples of the KV
page size. A 16,320-token chunk fits 102 pages. The explicit prefill graph
limit retains the 16,384-token bucket, whose synthetic sequence spans
103 pages. Deriving the limit from the new chunk size would end the
512-step bucket series at 15,872. Graph replay binds block tables from
live cache lengths and directs padded writes to the sink page.
The full and SWA pool sizers round down to complete pages and each pool
reserves a separate sink page. Compared with page size 1, live tails
can occupy up to 159 extra slots per request. A prefix lookup can also
reuse fewer tokens because it matches complete pages.

## Activation

The pinned SGLang build selects FA3 from sgl-attn
`f89bc2306632d1ec5f97b014dded4254f5b4a907`. The Hopper E4M3 head-dim-64
large-query kernel uses a 192-by-160 query-by-key tile. FA3 selects paged
TMA when the page size is divisible by 160, the maximum query length
times the query-to-KV head ratio exceeds 192, and the attention uses
no left padding, fused KV append or local mask. The target writes
its KV cache before attention, so full attention qualifies from 25
query tokens at the per-rank GQA ratio of 8. E4M3 inputs, descales and
attention coverage are unchanged. [Pinned tile selection](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/tile_size.h#L46-L52),
[pinned TMA dispatch](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_api.cpp#L379-L401)

DFLASH target verification has eight queries per request, or 64 effective
queries. It uses the one-warp-group 64-by-128 tile and fails the strict
TMA query threshold. Its one paged attention call still covers the full
live context or the original sliding window, with native SplitKV and
packed GQA. Local SWA keeps the non-TMA path while its mask is active.
The bundled BF16 FA4 draft uses its generic paged cp.async loader when
page width differs from the selected tile. Its compact draft-window
option is disabled by the V48b launch configuration.
[Pinned one-warp-group heuristic](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/heuristics.h#L8-L13)

At 74,160 cold tokens on TP8, the 16 full-attention layers account for
about 180 TFLOP per GPU out of about 208 TFLOP of cold attention. The
TMA path replaces per-row page-table loads and address calculations
with a page lookup and bulk loads per eligible KV tile. The work
estimate describes exposure to this change. Kernel and end-to-end
speedups need a GPU measurement. Warm prefix-cache hits reduce the
prefill opportunity, and wider pages can change cache eviction and
request packing.

## Verification

The full patch passed application and reverse checks on the pinned base,
V48b hunk parity, syntax and whitespace validation. The longest added line
is 86 bytes. Executing the indexing source with NumPy-backed, bounds-checked
pointers passed 2,160 allocator and 108 decode/draft metadata cases across
nine page sizes. Cases covered zero-length requests, partial and multiple
pages, 16,320 and 16,384 tokens, long context, random physical pages and
SWA mappings. Independent review confirmed the live replay bounds, TMA
predicate and cache geometry, and passed 91,624 signed page-index cases.

Source review covered the CUDA paged allocator, SWA freeing, Rust radix
cache, target and draft page tables, and host KV transfer. HiCache retains
its existing register fallback when the page width cannot tile the TMA
transfer stage. GPU compilation, FA3 and FA4 graph capture and replay,
token agreement, peak memory, TTFT and speed remain unverified.

The temporary worktree `okc-v60.q3cbpd` and the CPU indexing harness
occupied 173,956 KiB. The exact frozen patch reversed cleanly in the
worktree before removal. Available shared-volume space increased from
34,290,912 KiB to 34,471,804 KiB, an observed difference of 180,892 KiB.
