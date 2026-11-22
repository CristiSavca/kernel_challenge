# V100c physical-page control on scored V64

This control keeps V64's Humming experts and sequence-parallel prefill. Its two extra files enable non-power-of-two pages in the allocator and in decode and draft metadata. All nine V64 patch sections are byte-identical to the scored source. Both extra files have the same Git blob IDs as V60. The official exporter produced 49,063 bytes across 11 files, SHA-256 `335845f55f6e584e938dcb5bba72c18f89677f83f78a1374692525306cd635ed`. [Patch](patch.diff) · [Validation](validation.json)

For the 32-user comparison, use these eight flags. A 16,320-token chunk occupies 102 physical pages. The 16,384-token graph limit retains the existing capture bucket.

```text
--flag swa-full-tokens-ratio=0.25
--flag moe-runner-backend=humming
--flag speculative-moe-runner-backend=marlin
--flag enable-layernorm-sp
--flag page-size=160
--flag chunked-prefill-size=16320
--flag max-prefill-tokens=16320
--flag cuda-graph-max-bs-prefill=16384
```

On this source, V64's 54 CPU integration cases passed with 324 pre-normalization calls. An additional CPU fixture executed V60's production indexing and allocator bodies, checking 25,209 signed indices and 108 request extensions using nine page widths and both prefill lengths. Page geometry can also affect prefix-cache reuse, request packing and idle capacity. A matched H200 run is needed to measure the net result. [Page fixture](page_check.py) · [V60 design](../v60_fa3_paged_tma/README.md)

At 32 users, scored V64 recorded 20,547.4 tokens/s/GPU, a 92% cache hit rate, 4.2 accepted tokens per step, and a median decode rate of 43.3 tokens/s/user. The first-turn and later-turn p90 times to first token were 110.0 seconds and 0.696 seconds respectively. This reference load had one repetition. Use those measurements to distinguish prompt gains from a loss of prefix reuse or a decode regression. [Scored V64 result](../../runs/abd217baf0ed49fd.json)
