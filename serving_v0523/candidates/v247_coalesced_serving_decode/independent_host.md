# V247 host dispatch and replay

The four suites ran on tree `0f51d90866886150d3ec712cdcb965271c7cd5ed`, exported as `91f9e78ff697d73a6bc60d27f2a9360cc7fcb653cb5d6b2ccc1ea3ea87c52159`. They exercise production host entry points with CPU tensors. CUDA transport, GPU kernel execution and timing remain unmeasured.

| Coverage | Result | Reproduction with an applied checkout |
| --- | --- | --- |
| Muse attention, Breakable replay, physical cache writer, mixed prompt and verification tail | 20 paired cases, 12 direct prompt allocations, 83,356,928 physical/logical cache bytes checked. The 434,176 specialized-tail BF16 values passed the logical dense reference tolerance. [Receipt](independent_storage_graph.json) | `python ../v246_direct_mixed_prompt_storage/independent_graph.py --stage "$stage" --candidate .` |
| Legacy attention provider | 72 destination copybacks checked across 77,199,360 BF16 components, including 36 retries after an unsupported output argument. [Receipt](independent_storage_legacy.json) | `python ../v246_direct_mixed_prompt_storage/independent_legacy.py "$stage"` |
| Coalesced expert routes with real TP2, TP4 and TP8 Gloo | All 4,353,728 BF16 values matched V241 across 112 rank-layer cases. Includes stale or unsuitable peer workspace and fallback storage inspection. [Receipt](independent_peer_gloo.json) | `python ../v242_coalesced_prompt_routing/distributed.py "$stage" .` |
| Coalesced expert routes across Breakable replays | 24 accepted and 48 rank-vetoed replays recovered matching scores and IDs. [Receipt](independent_peer_graph.json) | `python ../v242_coalesced_prompt_routing/graph_replay.py "$stage" .` |

Run from this candidate directory with CPU PyTorch and a Git checkout containing the frozen V241 and V247 trees.
