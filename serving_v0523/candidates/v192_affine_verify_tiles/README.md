# V192 · Direct FP8 cache reads in target verification

V192 changes the way the target reads long, full-attention FP8 KV history. On Hopper, physical rows certified as consecutive can enter a pipelined Tensor Memory Accelerator (TMA) load. Each 128-key tile can start at a different physical slot. The candidate retains V190’s five launch flags. V68 is the measured ancestor. A private 16-user V68 check reached **17,512.9 tok/s/GPU**; end-to-end performance for V190 and V192 is unmeasured. [V68 judge receipt](../../runs/04b01f958aa5487c.json) · [V192 manifest](validation.json)

## What executes

The existing page-one metadata kernel examines the physical ID of every live row as it writes the ordinary page table. For each complete group of 128 rows it writes the first physical slot if and only if all slots are positive and consecutive. A partial or fragmented tile receives −1. The companion table takes one int32 per 128 page slots and is allocated with the capture buffers. Replay refreshes the live entries as request assignments and lengths change. There is no new kernel launch or physical KV copy. [Relative patch](delta.diff) · [Graph lifetime checks](independent_graph.json)

For a full-attention DFLASH target verify, each request/partition intersection checks its tile origins and current cache capacity on the device. When all pass, the program reads K and V with two-dimensional tensor maps against the original FP8 cache bases. Each bank uses its own physical row stride. An intersection containing a rejected tile runs the indexed page-table loop. Query packing, tile order, causal visibility, descales, FP8 QK inputs, BF16 probability/value products, FP32 state, partition merge and output order follow V190. The draft, SWA attention, prompt paths and native FA fallback retain their earlier routes. The fast path requires the ordinary captured target metadata, Hopper FA3, page size one, explicit compatible bank shapes/strides, and hardware-appropriate nondeterministic dispatch. [Official export](patch.diff) · [Source checks](verify.py)

## Evidence

* Offline Triton 3.8 compiled 32 SM90 specializations across all four target batch sizes, with and without K descales. The eight descriptor specializations emitted four static pipelined TMA instructions, used 209–225 registers and had zero spill stores or loads. A cubin cannot establish GPU time. [SM90 receipts](sm90.json)
* The production Triton interpreter exercised affine, exchanged-interior, shuffled, zero-slot, partial, shortened, remapped and descriptor-capacity cases with separate pitched K/V banks. The test reproduced all 819,200 FP32 partial components and 327,680 BF16 output components from the V190 kernels. A separate dense calculation checked 7,680 output values. The fixture decodes the CPU interpreter’s BF16 storage before its NumPy dot operation. Neither CPU comparison reproduces GPU MMA reduction order. [Test and method](interpreter_check.py) · [Results](interpreter.json)
* An independent source and graph replay oracle checked 720 changing batches, including 4,261 intersections. The full candidate and its three-file delta each rebuild the same tree and apply in reverse. [Independent evidence](independent_graph.json) · [Manifest](validation.json)

## How often the direct path can run

A replay combined 767 recorded public V41 request lengths with the pinned page-one allocator. The request grouping, append order and returned free-list order were modeled. The archive does not record physical cache addresses on the serving judge. The percentages below are the share of logical history tiles in wholly certified intersections. [Replay source](independent_allocator.py) · [Results](independent_allocator.json)

| Modeled free-list history | 1 user | 4 users | 16 users | 32 users |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 94.92% | 82.65% | 36.67% | 3.81% |
| Shuffled 4,096-row runs | 94.57% | 54.05% | 13.01% | 0.02% |
| Shuffled 256-row runs | 93.71% | 33.61% | 12.88% | 0.00% |
| Shuffled 32-row runs | 0.00% | 0.00% | 0.00% | 0.00% |

The target kernel already reads each logical K and V once per request/head/tile and reuses them across the query block. TMA can reduce per-row gather and issue overhead. Full KV payload and model compute remain necessary. In the independent H200 attention-kernel check, a different native implementation reduced baseline-overhead-corrected full verification from 205.2 µs to 101.4 µs at 16 users. That 103.8 µs difference across 16 global layers would be 1.66 ms per round **if** it transferred to serving. The V68 check reported acceptance 4 and 48 tok/s median user decode, an illustrative 83.3 ms period. These measurements were taken from different programs and cannot establish V192’s benefit or four-load score. [Archived shape and serving metrics](hardware_context.json)

A further allocator sensitivity isolated runs of four or more certified tiles inside a long intersection. Eligible shares at 16/32 users rise to 97.6%/97.5% in the fresh case and 94.9%/95.2% in the 4,096-row case. Sustaining a pipelined kernel across those runs would require carrying the original FP32 online softmax state across boundaries. V192 ships whole-intersection dispatch. A matched H200 V190/V192 comparison would resolve the cost of descriptor reads, fragmented fallback and per-intersection voting. [Run-length model](segment_sensitivity.py) · [Results](segment_sensitivity.json)
