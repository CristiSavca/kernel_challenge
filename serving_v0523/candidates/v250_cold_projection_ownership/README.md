# V250 tests cold-prompt projection ownership on V68

**Keep V68's token ownership for the long-prompt pooler, router and unpooler.** V64 already divides the prompt among its TP ranks before these three operations. On a 16,384-token prompt, eight GPUs each process 2,048 rows. Sharding their output columns would preserve the node-wide FLOP count and add communication. V228 saves replicated work in target verification, where ranks receive the same input rows. V179 and V242 exchange expert inputs or routing metadata after the two input projections have run. [Passing V68 result](../../runs/04b01f958aa5487c.json) · [V228](../v228_sharded_verify_projections/README.md) · [V242](../v242_coalesced_prompt_routing/README.md)

The V68 SP entry, pooler, two-factor router, grouped expert input gather, return collective and unpooler ran at three real prompt lengths in Gloo process groups of two, four and eight. All 42 rank-cases observed owner rows at all four matrix calls, and all 989,296 returned BF16 components matched. The fixture uses smaller feature dimensions, CPU FP32 accumulation for BF16 inputs, and a controlled expert body to isolate ownership and transport. The tensor sizes below are calculated separately at the published model's full feature dimensions. [Executable fixture](check.py) · [Results](check.json)

| Per rank and sparse layer at 16K/TP8 | Result |
| --- | ---: |
| Pooler, two router products, unpooler | 137.44 + 17.18 + 137.44 GFLOP |
| Nominal unique cold-weight bytes that output-column ownership could save | 119 MiB |
| Extra received bytes for a direct column-sharded implementation with a reusable full reduced expert result | 267.75 MiB |
| Extra received bytes if unpool inputs must also be gathered | 379.75 MiB |

The communication estimates use one new hidden-input gather shared by the router and pooler and owner transposes for all three output tensors. The direct implementation's saved weights correspond to 26.0 µs at 4.8 TB/s. Its extra receive payload alone corresponds to 312–442 µs even allowing the full published 900 GB/s bidirectional NVLink figure as receive bandwidth. Both are peak-bandwidth conversions. Allocation, concurrent kernels, cache hits and real collectives change elapsed time. [H200 specifications](https://www.nvidia.com/en-us/data-center/h200/) · [Byte accounting](check.json)

The weight metadata lists exactly 130,039,296 FP32 elements, equal to the size of the 62 router matrices and their correction biases. The aggregate establishes neither individual tensor values nor that low BF16 factors vanish. A finite, exactly representable two-factor example swaps the eighth selected expert when its low factor is removed. A paired projection that retains both factors is a separate hypothesis. [Pinned public checkpoint metadata](checkpoint_metadata.json) · [Top-eight example](check.json)

The available H200 projector control covers at most 256 rows, below the native 2,048 owner rows. In the separate eight-H200 V41 trace, a 5% whole-trace improvement confined to eligible long MoE layers would require 1.688 ms per layer at 16 users or 1.559 ms at 32. No new submission or hardware allocation was made for V250. [Timing and reach](../v197_grouped_prompt_gate_up/independent_reach.json)
