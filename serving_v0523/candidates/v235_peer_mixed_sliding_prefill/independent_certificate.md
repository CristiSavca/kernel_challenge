# V235 proposal ownership and accepted KV writes

The frozen production bridge admitted one, two, three and four verification tails. On each admission, the target received the proposal’s exact tokens, positions and reserved slots. The mixed target batch acquired its causal certificate only after the proposal ticket was consumed. Neither source batch acquired one. A mask, a top-two tree, a ragged layout, pre-existing speculative metadata, and a simulated remote veto each declined for all four cohort sizes, preserving the ticket. [Test and code paths](independent_certificate.py)

Twelve separate CPU settlement comparisons covered all accepted-prefix lengths from one through eight. Across ordinary and remapped physical slots, exact and padded row counts, a partial fused write, the per-layer path and a nonunit scale, all 460,800 BF16 pool words matched the normalized-feature reference. The test checked 336 rejected tail rows and confirmed publication of accepted lengths and bonus tokens. The CUDA writes were simulated on CPU. [Frozen results and source digests](independent_certificate.json)

The paired V233 and V235 target graph replay checks cache-slot reuse by a subsequent ordinary request. [Attention and slot reuse](independent_causal_attention.md)
