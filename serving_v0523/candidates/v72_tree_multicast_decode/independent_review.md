# V72 independent composition review

Reviewed the final SHA-256
`e90c47ab73e66158b678893bb6fc7950c3b52f38cc3213ac6d7b7716d1b19705`.
The patch contains 128,042 bytes across 23 permitted source files. Its longest
added line is 106 characters. Comparing normalized patch sections independently
confirmed 21 sections equal their V68, V69, V70 or V71 parent. The remaining
sections combine the two distinct `EnvBool` declarations and add the prefix
tile fast path. The preserved `391c1430` patch matches the union before that
attention change. [Final patch](patch.diff) and [export validation](validation.json).

## Expected activation

- The V44 judge startup records FA3 target attention, page size one, 32
  requests, FP8 E4M3 target KV, BF16 draft KV, legacy SWA pools and a folded
  CUDA graph for the draft selector. The DFlash2 checkpoint's documented
  selector has K=16 and its own 2,048-token attention window. The separate
  `speculative_draft_window_size` flag defaults to `None`, and the pinned
  normalizer only changes it when explicitly set. The draft therefore shares
  the target request map with the proposed flags. [V44 startup](../../runs/c3b7bd142c264853.log),
  [DFlash2 checkpoint](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/blob/main/README.md).
- Muse's decoder supplies D64, GQA 8 and a zero attention logit cap. The eight
  verify rows give 64 packed query lanes per KV head. The pinned static
  verification default and the judge's TP8/EP8 topology satisfy the FA3
  backend guard. The scheduler allocates pools before initializing either
  attention backend, and enables tree metadata before graph capture. SGLang
  collects the new tree flag automatically from the `Spec` namespace.
  [Muse text configuration](https://huggingface.co/docs/transformers/en/model_doc/muse_spark).
- Ran the exact final backend and worker eligibility bodies with a mock of
  that profile. Both enabled the tree. Five worker mutations each preserved
  linear verification, covering compact cache, out-of-range selector width,
  SWA ring, separate decode attention, and a different request map. Live use
  also requires a captured draft graph and the ordinary greedy request
  eligibility checks. The H200 startup marker is `DFLASH selector tree
  prepared for greedy CUDA graph batches.`

## Composition and prefix masking

The target verifies the same eight rows, selecting its own greedy result for
each accepted child. The V69 Humming CTAs share A only for paired N tiles on
the same live expert and K extent. The V71 collective uses the same BF16
result, FP32 accumulation and existing symmetric pull plane as stock. Its
borrowed outputs are consumed on the forward stream before the V70 commit
uses that stream to permute page-one slot IDs. Existing full-to-SWA IDs stay
paired. The post-commit event fences scheduler and HiCache access to the shared
map. The tree adds an auxiliary gather of up to 20 MiB for 32 requests and five
BF16 hidden taps. This amount alone cannot establish peak GPU memory.

The final fast path skips ancestry work only when `tile < prefix // TILE_N`.
Every key in those tiles precedes the speculative suffix. The query's depth
still sets the sliding-window lower bound. Mixed tiles and the first fresh
tile execute the full ancestry and depth checks. An independent CPU comparison
covered 1,680,000 visibility and position checks with tile widths 64 and 128,
tree widths 2 through 8, full and SWA modes, and tile and 131K boundaries.
The component test also compared attention outputs before and after the patch.
[Tree component](../v70_dflash_selector_tree/README.md) and [collective review](../v71_direct_multicast_workspace/independent_review.md).

## Rules and remaining validation

The live rules bundle re-read for this review had SHA-256
`71ecccaaf78c1993beb26fe217a9e4c2f1b25f8725895b65879a156e493e5d18`.
V72 stays within the source limits. The tree opt-in uses ordinary request
semantics and runtime capabilities, with all target proposals evaluated and
visible keys preserved. The V68 flags remain `swa-full-tokens-ratio=0.25`,
`moe-runner-backend=humming`, `speculative-moe-runner-backend=marlin` and
`enable-layernorm-sp`. Tree use additionally supplies `speculative-dflash-tree`.
Judge flag acceptance, H200 JIT, live tree replay, target output agreement
and requested token counts, peak memory within +2 GB, TTFT within 2× stock
and the actual acceptance and throughput effect still need official evidence.
[Live challenge rules](https://openkernelchallenge.dev/rules).
