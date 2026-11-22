# V178 direct FP8 prompt K/V

V178 prepares the current prompt's K/V in dense FP8 storage that both the physical cache writer and FlashAttention 3 can consume. The previous prefix is gathered from the physical cache, using only the visible left halo for a sliding-window layer. The writer still publishes every current row to its allocated slot. The candidate inherits all four measured V64 flags and changes two target-attention source files. [Export validation](validation.json) · [V64 delta](delta.diff)

The retained implementation is **V178b**. During graph preparation, the pinned runtime can execute a synthetic batch with zero slot IDs while CUDA capture is inactive. V178b requires the runtime's live replay row count or ordinary scheduler metadata before it changes K/V. [V178b](../v178b_live_direct_prompt/README.md)

For a modeled 74,160-token cold prompt, full-and-sliding dense packing would gather 1.945 GB per GPU across the layers and chunks. Direct current-row conversion leaves 0.768 GB of past K/V to gather. The difference avoids 2.354 GB of logical reads and writes relative to packing all rows. V64 performs page-one attention directly and does not incur that packing expense. The performance comparison with V64 depends on the speed of dense FP8 FA3 on H200. [Geometry](geometry.json) · [CPU comparisons](host.json)

An alternative that distributes prompt tokens among two TP peers has essentially the same maximum per-rank QK/AV work with balanced row ownership. It adds 7.062 GB of query and output transfers per rank and 1.456–1.628 GB of K/V transfers for the same modeled turn. The pinned Muse sequence-parallel configuration also rejects context parallelism. [Reproducible derivation](geometry.py)
