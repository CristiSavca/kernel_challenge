# How much draft page-table work is available

The 16-user V41 client trace contains 208 completed requests. For the first proposed eight-token block in each, the median full-width lookup would build 85,421 physical page IDs. A tile-aligned suffix needs a median 2,125.5 IDs. Across those shapes the count falls from 17,237,738 to 441,578, a 97.44% reduction in live address entries. The graph reserves its normal backing tensor. These are counts for one representative decode block per request. [Calculation for all 767 requests](independent_geometry.json) · [V41 client trace](../../../experiments/modal_node/20261113T204343Z-bench/results__20261113T204343Z-bench__v41__load.json)

| Client load | Completed requests | Median full page count | Median suffix page count | Fewer live page entries |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 39 | 97,077 | 2,117 | 97.77% |
| 4 | 104 | 81,699.5 | 2,128 | 97.28% |
| 16 | 208 | 85,421 | 2,125.5 | 97.44% |
| 32 | 416 | 85,817 | 2,124 | 97.43% |

V41's server log identifies an eight-row DFlash2 block, FA4 and a shared, noncompact request map. The V68 H200 receipt confirms page size one, a context limit of 131,072, an FP8 target cache and a BF16 draft cache. The release creates a BF16 draft pool for the FA4 path when the target uses FP8. The read-only receipts omit per-call V188 dispatch and GPU timings. [V41 server](../../../experiments/modal_node/20261113T204343Z-bench/results__20261113T204343Z-bench__v41__server.log) · [V68 private check](../../runs/04b01f958aa5487c.json)

The native SM90 FA4 tuner gives a 128-by-128 attention tile at head dimension 128. The proposal keeps up to 2,047 rows to the left, all eight active rows and up to 127 earlier rows to align the start with an original key tile. The earliest and latest attended physical key and each split's physical tile boundaries agree in 53,168 query interval and 299,070 split interval comparisons against the release implementation. Tests include 3,323 distinct prefixes, batch sizes 1, 4, 16 and 32, packed and unpacked query indexing, and the maximum context boundary. [Pinned-source replay](independent_geometry.py)

The FA4 wrapper derives its maximum paged length from the physical pool capacity. Its generic SM90 split selector bounds the local estimate by the left window, the right window and a query tile. With this checkpoint’s local width and eight query heads per KV head, the scheduler sees 33 key tiles with either page table. It chooses 33, 33, 8 and 4 splits for captured batches of 1, 4, 16 and 32 respectively. The savings come from gathering and writing fewer physical addresses and launching fewer metadata thread blocks. All attention dot products, split partials and merges remain. H200 measurements must establish the share of a decode cycle spent building this table. [Range and split results](independent_geometry.json)
