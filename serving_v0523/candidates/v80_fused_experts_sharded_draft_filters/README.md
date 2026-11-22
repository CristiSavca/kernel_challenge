# V80 fused target expert staging and sharded draft filters

V80 combines [V77's fused expert routing and input staging](../v77_fused_expert_staging/README.md) with [V78's sharded DFLASH convolution coefficient generators](../v78_dflash_sharded_conv_filters/README.md) on the eight-row V76 target verification tree. This candidate isolates both changes at the existing width.

The target path removes one serial kernel boundary per eligible routed layer. The draft path reduces the ten coefficient generators' persistent BF16 weights by 280 MiB per TP8 rank and introduces ten all-gathers per eligible draft pass. Repeated scans of routing metadata, route imbalance, gather latency and allocator behavior all affect the net result.

The [official export](patch.diff) is 149,302 bytes, SHA-256 `11a1995768ccc8419fe17caea989b27352596eb533ac73ec24d7df03b5a3e9c6`, against pinned SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`. The [validation receipt](validation.json) shows byte-for-byte patch-section parity for all 28 files. Twenty-three match V76, two match V77, and three match V78. All 25 changed Python sources compile. The patch passed both forward and reverse application and the longest added line is 106 bytes.

Both component CPU suites ran on the combined source. The target test passed 192 layouts, 48 grouped finalizer cases and 55,733 rounded per-route values. The draft test passed 8 factory settings, 144 constructors, 726 checkpoint cases and 1,008 convolution prepare/finish pairs, along with the inherited draft context tests. [Target CPU receipt](cpu_expert.json) and [draft CPU receipt](cpu_draft.txt).

Use V78's seven launch flags at a matched load:

```text
--flag swa-full-tokens-ratio=0.25
--flag moe-runner-backend=humming
--flag speculative-moe-runner-backend=marlin
--flag enable-layernorm-sp
--flag speculative-dflash-tree
--flag speculative-dflash-tp-context-projection
--flag speculative-dflash-tp-conv-projection
```

V80 is a reserve candidate. End-to-end speed, device correctness, replay safety, first and later TTFT, exact output lengths and peak memory require official measurement.
