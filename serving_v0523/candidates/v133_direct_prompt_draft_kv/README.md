# V133 direct prompt writes to draft KV

V133 keeps the two full-size matrix multiplications from V120b and removes a cache copy after the second one. During an eligible prompt append, the existing stacked normalization and rotary kernel writes all five layers of BF16 draft K/V directly into the physical slots supplied by the worker. It handles uneven prompt tails and preserves V120b's sequence-parallel input projection. The full candidate uses V120b's eleven launch flags, including the existing opt-in direct-write flag. The official export is 101,396 bytes. [Full patch](patch.diff) · [Three-file change from V120b](delta.diff) · [Export and flag checks](validation.json)

## Work removed

The prompt appender trims graph padding and translates slots through its normal allocation plan. Each full prompt then runs one stacked KV GEMM with the same dimensions and weights as V120b. Each normalization program looks up its physical slot, checks the current pool bound, skips the native reserved slot zero, and writes the original BF16 values. The prefix-valid decode writer retains its own length masks and slot semantics. The production path keeps the original callbacks for graph capture, custom pool writers, debugging, unsupported storage, and unavailable native stores.

For 16,320 live rows, five layers, and one local 128-channel BF16 KV head, the parent writes and rereads 39.84 MiB of intermediate keys and values per GPU. V133 removes 79.69 MiB of logical traffic and up to five scatter launches for that append. The full projection stays live until the direct write completes. An eligible append also avoids growing the helper's persistent projection and K/V workspaces. Caching allocator reuse, actual peak VRAM, TTFT and end-to-end throughput require physical measurement. [Scored V64 draft-cache dtype](../../runs/abd217baf0ed49fd.log) · [Parent projection](../v120b_native_muse_prompt/README.md)

## Validation

Independent CPU tests matched 2,709,504 prompt cache and guard bytes against the native writer's slot behavior and 1,483,968 legacy BF16/FP8 cache bytes. Both paths rejected all 60 invalid-slot examples. Of 67 production-worker prompt fixtures, 17 used the new path without growing a workspace, 49 took a fallback and one prompt was empty. A separate run exercised 144 prompt projection and logits calls across tensor parallel sizes 2, 4 and 8. [Byte and worker checks](prompt_check.json) · [Distributed projection](prompt_projection.json) · [Independent review](independent_review.md)

All 22 specializations assembled for Hopper without register spills. The one-head, full-rotary direct BF16 prompt kernel used 26 registers per thread. The export applies to the pinned SGLang source, reverses cleanly and reconstructs the same tree from V120b plus the delta. V133 is held for a matched hardware comparison against V120b. [Hopper build](sm90.json) · [Serving rules](../../RULES.md)
