# V253 backs the private draft cache with sliding-window slots and groups host copies

V253 combines V249's private draft storage with V154's page-first host backup on the hardware-tested V68 control. Five draft layers keep their native BF16 K and V values for every materialized row. The target allocator's sliding-window map supplies their physical addresses, including after eviction and host restore. The backup uses the original transfer stream and coalesces consecutive host destinations for each K or V plane. A gap, duplicate or reversal closes the current copy. [Official export](patch.diff) · [V249 ownership and guards](../v249_private_draft_swa/README.md) · [Copy implementation](delta.diff)

At the slot counts recorded on V68, the five draft layers would allocate **6.116 GiB less device KV payload per GPU**. Their host payload would fall by about **40.64 GiB per rank**. FA's additional full-width graph tables and write locations cost about 48 MiB for one backend at batch 32 and context 131,072. The target budget stays conservative. Final device peak, pinned host residency, transfer counts, first-token latency and model throughput require a hardware run. [Capacity calculation and assumptions](../v249_private_draft_swa/capacity.json) · [Passing V68 H200 control](../../runs/04b01f958aa5487c.json)

## What ran

The native draft allocation, write and restore test checked 20,480 BF16 words across six ordinary and accepted-prefix cases. It exercised the native translator and host transfer registration, 27 eligibility fallbacks and recovery from a partially completed fused write. A second fixture drove the captured and eager FA4 paths through 25 layer calls with changed slot ownership. All 3,200 inspected BF16 values matched at the FA4 boundary. It recovered all 1,791 old rows needed by a split prefix and checked restore ordering through target event 61. These fixtures execute production source bodies on a CPU. Device attention arithmetic, real graph replay and H200 DMA remain open. [V253 draft replay](draft_swa_replay.json) · [V253 FA4 and transfer replay](independent_fa.json)

The joint test took the actual five-layer sidecar returned by the cache builder, selected written slots after recovery, and passed their data through the stock and candidate C++ copy selectors. The CPU transport model checked every restored value and the surrounding guard regions. At native width the 64 pages contained 81,920 BF16 values in each case. [Runnable joint test](copy_replay.py) · [Results](integrated_copy.json)

| Host destination order for 64 pages | V249 copy descriptors | V253 copy descriptors |
| --- | ---: | ---: |
| Consecutive | 128 | 2 |
| Ten consecutive runs | 128 | 20 |
| Fully fragmented | 128 | 128 |

Both implementations moved the same bytes. A separate replay of the production copy helper ran 567 cases for each of CUDA 12.7, CUDA 12.9, CUDA 13 and ROCm against simulated transfer APIs. The two staging controls and V253 produce the identical C++ header. The earlier offline Hopper builds therefore cover this header. [Transfer compatibility](copy_compatibility.json) · [Hopper objects](../v154_coalesced_hicache_backup/sm90.json)

## Qualification

The official CLI exported 99,080 bytes on the public SGLang pin. Both component orders, forward and reverse reconstruction, source restrictions and all four hardware-tested V68 flags passed the public contract checks. The runtime selects the new storage from the loaded model, backend and allocator. Archived traces informed the analysis only. V253 is a focused V68 control, separate from the broader V248 kernel changes. H200 throughput, output agreement, peak memory and TTFT have yet to be measured. [Source, flags and component verification](validation.json) · [Trace limits](../v251_hicache_residency/README.md) · [Published rules](https://openkernelchallenge.dev/api/v1/serving)
