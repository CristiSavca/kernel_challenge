# V68 independent serving review

Reviewed patch SHA256
`6248a58842e1a22c1cda5a957d5ab464a6ec824efcdf8308bdcc9c6a1046e505`.
The 85,080-byte patch contains 15 permitted source files and a maximum
added line length of 99 characters. Its hunks match V67 plus the six V66
changes relative to V64. The three V67 capacity paths and six V66 decode
paths are disjoint. V68 uses the same four mutable flags as V65 and the
stock DFLASH block length. Dispatch uses hardware, phase and tensor
geometry. The [V65 rules audit](../v65_grouped_sp_verify_attention/rules_audit.md)
records the live rule verification. [Official serving rules](https://openkernelchallenge.dev/rules)

## Decode and attention

V66 validates global IDs at their input width, constructs local IDs and a
stable suffix on the device, and restores original routing slot order.
The suffix preserves Humming's last-expert boundary on a fixed physical
arena. All-empty ranks schedule zero grouped tiles and contribute zero
through finalization. The packed E2M1/E4M3 bank, FP32 full-K accumulation,
BF16 scale rounding and stock SiLU remain in the same arithmetic sequence.
Independently executed 3,036 final planner cases, including 506 with extreme
signed IDs, with exact slot maps, bounds and unique stores. [V66 review receipt](../v66_humming_grouped_decode/independent_review.json)

The MoE sequence goes from about 11 GPU launches to six for the judged
shapes, counting the removed Torch route-map operators and scale passes.
Grouped and indexed Humming choose the same WGMMA tiles in the supported
small-M tiers. Packed-weight bytes and per-expert tile padding are therefore
the same. A speedup depends on TMA transport, launch savings and pipeline
efficiency after compaction and activation gathering.

V67 increases request-map and FA3/FA4 verification page capacity on the KV
axis. V66's route count comes from query rows. With block length eight and
32 requests, top-k eight uses 2,048 physical slots and fits the grouped
path. V62 schedules only the live key lengths within the wider buffers and
preserves causal and sliding-window visibility. BF16 probabilities widen the
pinned FA3 E4M3 probability format and change numerical rounding. Query
lengths and route counts beyond the supported geometry use the incumbent
attention or indexed expert paths.

## Validation and open measurements

On the composed V68 source, the independent rerun passed V66's 600 planner,
48 combined arithmetic and 71 dispatch cases, and V62's 110 attention and
890 long-context ownership cases. Each CPU suite models GPU operations.
H200 compilation, dynamic graph replay and output agreement remain pending.

At the 2,048-slot limit, the target grouped decode arenas occupy 48 MiB,
matching V64's indexed arenas for those dimensions. Route metadata adds
small allocations. Verification attention, capture pools, KV and allocator
reserve all contribute to the measured peak. The judge still must establish
the 2 GB peak increase limit, output agreement and requested token counts,
and first-token latency within twice stock at every scored load. Serving
throughput remains unmeasured.
