# V231 encodes cold-prefill queries in the Muse normalization kernel

V231 removes one full query conversion pass per eligible target layer. On V229, Muse writes a BF16 query after RMS normalization and optional RoPE. The FA3 prefill callback then reads it and casts it to FP8. V231 preserves both native BF16 rounding points and sends the resulting BF16 bits through the backend's existing 65,536-entry conversion table before storing the query. The attention callback receives the same encoded bytes. K and V remain BF16 at the callback boundary. [Official patch](patch.diff) · [Four-file change from V229](delta.diff) · [V229](../v229_sharded_decode_serving/README.md)

At the public 16,384-row chunk, each rank has 1,024 query elements per row across 62 target layers. Removing the BF16 write and read eliminates **3.875 GiB of logical intermediate traffic per GPU and 62 separate conversions per chunk**. The new kernel also issues up to 0.969 GiB of logical byte lookups into the already resident 64 KiB table. Both paths write and subsequently read the FA3 FP8 query. Hardware caching and kernel occupancy determine actual DRAM traffic and latency. The 74,160-token first turn recorded for scored V64 corresponds to 17.54 GiB of removed query-intermediate traffic per rank across five chunks. [Trace accounting](trace.json) · [V64 turn geometry](../v170_long_prefill_attention_feasibility/check.json)

| Archived concurrency | Prefill calls of at least 4,096 rows | Live rows | Calls with multiple requests |
|---:|---:|---:|---:|
| 1 | 17 | 268,481 | 0 |
| 4 | 36 | 578,151 | 6 |
| 16 | 75 | 1,228,800 | 15 |
| 32 | 153 | 2,471,245 | 34 |

The table aggregates the repetitions in the archived V41 scheduler log. It establishes the frequency of compatible row counts. V231 has no physical H200 measurement of output agreement, throughput, peak memory or time to first token. V64 holds the official four-load score of 1.4119×. [Full trace and source hash](trace.json) · [Scored V64](../../runs/abd217baf0ed49fd.json)

## Execution and validation

The producer runs on the normal Hopper FA3 target with the native Triton normalization operator, a named E4M3 cache and 64-channel heads. It supports ordinary EXTEND with at least 4,096 captured rows, including the Breakable prefill graph. At replay, the original callback narrows to the live row count, applies K/V scales and writes the live physical slots. Single-request dense K/V, multi-request FA3 varlen and the paged fallback accept the exact encoded query. Full prefill graphs, alternative normalization operators, instrumented modules, deterministic mode, CP, MXFP8, short inputs and unsupported shapes retain V229's conversion. [Production replay](independent_reach.md) · [Admission cases](dispatch.json)

- The Triton interpreter ran the V229 and V231 producers on identical projection inputs. All 11,107,328 emitted FP8 query bytes and 3,146,624 BF16 key bytes agreed in 30 cases covering optional RoPE, NaNs, infinities, signed zeros, noncontiguous row pitch, poisoned output guards and 4,096 full-width rows. [Interpreter results](interpreter.json) · [Reproduction](kernel.py)
- Twenty release SM90 kernel specializations assembled offline. The public large kernel used zero spill bytes. Without RoPE it used 124 registers, up from 88. With RoPE it used 143, down from 147. Wider 128- and 256-channel stress variants could spill, which informed the 64-channel admission condition. [Compiler and matched V229 register counts](sm90.json)
- Sixteen paired Breakable CPU replays compared 153,815,856 FA3 operand and physical K/V bytes, including 67,248,128 identical FP8 query bytes at entry to FA3. Graph output retained BF16 values that E4M3 cannot represent. Replays covered full/sliding layers, dense/varlen/paged calls, unequal prefixes, nonunit K/V scales, captured padding and hook fallback. The separate admission test exercised four eligible states and 71 declines. [Independent replay](independent_reach.json) · [Admission replay](dispatch.py)
- The official CLI export is 384,603 bytes across 53 permitted paths. The full and delta patches reconstructed the same frozen tree and reversed to their respective parents. The six inherited flags parsed correctly in all 720 orderings, and the audit found zero newly undefined Python names. [Rules and hashes](validation.json) · [CLI flags](flags.json) · [Global names](global_names.json)
