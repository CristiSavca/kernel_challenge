# V84 routed decode and committed draft context

V84 combines three changes on V80's native eight-node verification tree. During target decode, the pooler and router compute their owned output columns and pack them into a shared gather. Its unpack restores the full expert input in token order and full FP32 logits in canonical expert order. The stock routing operator selects the experts. Token-owned staging applies the full pre-expert RMSNorm once per participating input and writes each selected local destination in stable route order. The packed W4A16 experts and original-slot output combination follow their existing paths.

After target acceptance, V82's guarded draft projector consumes the actual tree permutation and commit lengths. It computes the complete context projection on tiles containing committed nodes. The prefix-valid cache writer sees the same accepted context in request order. These stages run on distinct sides of the target verify/accept boundary. The published CPU results exercise individual kernel and dispatch contracts on the combined source. Collective progress, CUDA graph replay and complete-model numerics require an H200 check.

The benchmark comparison is V84 versus V80 at equal load. Use V80's seven flags plus the compact context option:

```text
--flag swa-full-tokens-ratio=0.25
--flag moe-runner-backend=humming
--flag speculative-moe-runner-backend=marlin
--flag enable-layernorm-sp
--flag speculative-dflash-tree
--flag speculative-dflash-tp-context-projection
--flag speculative-dflash-tp-conv-projection
--flag speculative-dflash-compact-context
```

The target-side changes may save normalization or router work while increasing route-planner or gather latency. The compact draft projector can lose to the dense vendor GEMM when few row tiles are eliminated. V84 needs measured committed throughput, first/later-turn agreement, exact output lengths, peak memory and TTFT across the scored loads. Its end-to-end speed is unknown. [V81](../v81_token_owned_expert_inputs/README.md), [V82](../v82_committed_draft_context/README.md), [V83](../v83_sharded_router_pooler/README.md).

The [validation receipt](validation.json) identifies the frozen patch and source parent for every changed path. The sole shared file combines the independent environment switches. The eight-slot runtime preserves V80's graph request width.

Frozen patch SHA-256 `8dafee38e53c17a484217016c9f0858b9f2b9bee1969af0818552e97818d298e`; 168,614 bytes, 30 allowed files, 27 Python sources compiled, longest added line 107 bytes. The local official export matched the staged Git diff, and forward and reverse application checks passed.

Combined-source CPU runs passed 588 routed-norm/staging cases, 18,442,925 BF16 values and 16 eligibility decisions; 175 committed-context row plans, 36 poisoned numerical projections and 19 fallback decisions; 192 inherited staging and 48 finalizer cases; 1,008 prepare/finish pairs and 726 checkpoint cases for the ten draft filters. Tree regression coverage included 1,400 proposals, 9,800 acceptance/slot permutations, 707 graph metadata resets, 72 causal/short-window attention cases and 32 backend eligibility checks.

The combined V83 transport test preserved 175,808 raw words and passed 16 router comparisons on PyTorch 2.5.1 CPU. Maximum absolute FP32 difference was 5.96e-7. [Independent integration review](independent_review.md).
