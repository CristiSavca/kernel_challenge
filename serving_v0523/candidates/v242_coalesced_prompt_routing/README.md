# V242 coalesces long-prompt routing collectives

V241's sequence-parallel expert path gathers routing scores, gathers expert IDs, and gathers a per-rank eligibility vote. V242 transmits all three in one rank-ordered byte payload. It removes **two TP collectives per eligible sparse layer, or 124 per 62-layer prompt pass**. Both the W13/W2 projections and the original expert-output reduction still run. [Production delta](delta.diff)

The transport preserves the full FP32 routing-score representation and the original integer IDs. The default FP32/int32 geometry uses one transpose to restore both contiguous tensors. A change to eligibility on any rank selects the parent's dense expert path on every rank. Before calling a fallback expert or its hooks, V242 recreates independent backing allocations containing every padded row and exposes only the true token slices. Changed routes and hooks are sampled on each replay. Eager short prompts and full capture use the inherited route.

## Cost and reach

For 16,384 tokens, eight routes per token and TP8, each rank formerly received 917,532 bytes of other ranks’ routing data and now receives 917,616 bytes. The potential gain is collective startup and sequencing. The additional local packing and unpacking account for approximately 2.25 MiB of logical reads and writes per rank. The canonical packet/transpose window holds an estimated 1.13 MiB more explicit live metadata than the parent's score/ID/vote results. On a fallback, cloning both full planes can briefly hold a 2 MiB total of original and cloned route data, up to 1 MiB over the parent, with no additional collective. Graph allocator reservation, transport scratch and actual peak require an H200 measurement. [Memory and operation counts](work.json)

In the archived physical V41 trace, 75 prompt calls at 16 users and 150 at 32 users met the grouped prompt size threshold. If every one also enters the peer path, V242 would remove 9,300 and 18,600 collectives respectively. A 5% whole-trace throughput increase would require a **net** saving of roughly 844 or 779 microseconds per eliminated collective. The historical trace does not measure the cost of these particular gathers. V242 also moves communication from the captured segment to its eager callback and adds local copies. There is no measured H200 speedup or evidence of a fourfold serving improvement from this change. [Trace lineage](../v197_grouped_prompt_gate_up/independent_reach.md)

## Verification

The two-file change against V241 and its full patch reconstructed the same tree and reversed cleanly to the donor and public pin. The official CLI export is 399,067 bytes, SHA-256 `51550e155265ffc2a7a6ba97c1512e4243168071963cfcad95f0c959cce0874b`. The eight inherited flags and current source rules were checked. [Validation and full flag map](validation.json)

Real Gloo process groups at TP2, TP4 and TP8 ran 112 paired rank-layer cases through the model's sequence-parallel entry, grouped expert adapter and finalizer. All 4,353,728 returned BF16 components matched. Another 84 rank cases preserved every bit of the routing fields across canonical, alternate and noncontiguous dtypes, including special FP32 encodings. Hook tests covered separate allocation addresses, exact padding contents, storage sizes and strides. CPU implementations supplied the matrix products, finalization, peer address transport and output reduction. [Distributed driver](distributed.py) · [Results](gloo.json)

The pinned Breakable callback executed 72 changing-route replays on fixed CPU tensor addresses, including 48 rank-local or peer vetoes. Capture issued zero transport calls and each replay issued one modeled route gather. [Graph driver](graph_replay.py) · [Results](graph.json)

No judge job or cloud hardware was used. Reproduce with CPU PyTorch using `distributed.py /path/to/sglang` and `graph_replay.py /path/to/sglang` after `verify.py` has reconstructed the frozen trees.
