# V88 fused draft head on measured V64

V88 adds the V85a vocabulary projection to V64’s native eight-position DFLASH2 linear selector. V64 scored 1.4119× across the four official loads with +1.9 GB/GPU peak memory. Its scored startup log confirms the selector was folded into the draft CUDA graph. V88 isolates the benefit and memory cost of the draft head against that measured control. [V64 scored result](../../runs/abd217baf0ed49fd.json).

The optimized draft head computes every vocabulary dot product at BF16 input and FP32 accumulation precision, rounds the result to BF16, then retains tile-level candidates. Shards share score bits and IDs with one gathered buffer. V88 retains the stock selector, target verifier, KV cache and acceptance logic. Vendor GEMM accumulation order and ties between equal draft scores can change individual proposals. The original target still verifies them.

Use V64’s four recorded flags and the head switch:

```text
--flag swa-full-tokens-ratio=0.25
--flag moe-runner-backend=humming
--flag speculative-moe-runner-backend=marlin
--flag enable-layernorm-sp
--flag speculative-dflash-fused-candidate-head
```

The final switch defaults to off. With it enabled, the captured sampler reaches the fused head before walking its linear proposal lattice. Eager selector proposals can also use it. The kernel file is byte-identical to V85a, whose 16 SM90 specializations compiled without spills. Eight CPU capture-hook cases covered both switch settings at batch sizes 1, 4, 16 and 32. The remaining fixtures covered 48 tiled projections, 65,536 BF16 encodings, 60 distributed transport invocations, 14 dispatch boundaries and 54 V64 integration cases. [Validation](validation.json), [SM90 compiler evidence](../v85a_bounded_vocabulary_tiles/sm90_aot.json).

The official four-user check crashed during draft CUDA graph capture because the model module lacked the is_cuda import. It produced no throughput rungs. V88a repairs the source and reproduces the original failure on the host. [Official crash](../../runs/30bb36feb528416f.log) · [V88a repair](../v88a_draft_head_capture_fix/README.md)
