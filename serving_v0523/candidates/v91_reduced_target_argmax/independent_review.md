# Independent review of V91

Compiled the production selector and ID unpack Triton definitions directly from Python AST. The reviewed source has SHA-256 `382e9be2f9acf20668de2a0107367b98e5538180047ca87cb4467a53efce5755`. Triton 3.8.0 on CUDA 12.9 produced 12 stock/candidate specializations and two unpack specializations for Hopper. [AOT digest and resource use](independent_sm90.json)

For floating and integer softcap arguments, the stock and candidate use the same FP32 divide, libdevice `__nv_tanhf` and multiplication. Each matched alignment case has the same blocked lane layout, tie-left reduction region and sequence of 16 butterfly shuffle offsets and masks. This order matters when a tile contains NaN. An invalid or minus-infinity tile winner receives the baseline shard-start ID and minus-infinity score. The candidate canonicalizes both signed zeros before key packing. The global masked key compares by descending score, then ascending global ID. The unpack TTIR truncates to unsigned 32 bits, inverts, and zero-extends to 64 bits. Both row alignment specializations generated the same 5,648-byte cubin using ten registers.

The fast path requires the existing deferred target-logit path, supported greedy mode, FP32 contiguous last dimension, a shared coordinator for TP and speculative result synchronization, and a multi-rank NCCL group. The rank offset uses the same equal shard width as the preceding code. The candidate reduces one signed 64-bit key per row, so the same ID reaches each rank before the corresponding redundant broadcast is skipped. The ordinary synchronization path remains active for other topologies and sampling modes.

The kernels were compiled offline. H200 CUDA graph replay, NCCL end-to-end timing, and model-level agreement await a private run.
