# V203 gathers vocabulary winners for mixed verification

**Scheduler correction, November 19.** On the production mixed-batch entry, V202–V206 run the target outside the owner-projection context. V203–V206 also leave the sharded verification-head flag at its default. The earlier component tests enabled those features through a separate helper. Queued V204/16 measures mixed batching and expert changes with the full vocabulary gather and raw context features. V209 connects both optimizations to the production scheduler. [Scheduler replay and source](../v209_mixed_prompt_context_entry/README.md)

V203 computes every target-vocabulary score for prompts and running requests in the same target forward. It gathers the full softcapped vocabulary for the ordinary prompt sampler. For each eligible greedy verification row, every GPU contributes its winning score and token ID. The verifier resolves the global winner and runs the existing acceptance and prefix-valid cache writers. The fast path checks the actual shard width, representable token IDs and a finite FP32 softcap. It retains the first valid index on ties, including rows with NaNs. Requests with sampling adjustments, beam state, grammars or other unsupported features follow their original serving routes. [Patch](patch.diff) · [Eligibility and compiler review](independent_compiler.md)

At the largest possible mixed call in the archived 32-user schedule, one prompt shared a target forward with 31 eight-row verification chains. The local matrix on each rank still produced 25,154,976 bytes of FP32 scores. The full-vocabulary gather destination fell from 201,239,808 bytes to 808,192 bytes for the prompt and 15,872 bytes for the tail winners. These are tensor sizes. They exclude allocator caching, existing graph buffers and other live model allocations. [Per-load sizes](head_budget.json)

| Users | Candidate mixed calls in V41 replay | Full-vocabulary verification output avoided per rank over the trace |
| ---: | ---: | ---: |
| 1 | 0 | 0 |
| 4 | 70 | 1.06 GiB |
| 16 | 99 | 7.86 GiB |
| 32 | 59 | 7.67 GiB |

V41 did not run this schedule. Its logged request chronology gives opportunities under the candidate's graph and continuation conditions. V203 still pays for local vocabulary projection, cross-rank winners and all target attention and expert work. [Archived trace assumptions](../v202_mixed_target_serving/phase_budget.md)

The real mixed sampler and verifier agreed with the full-vocabulary reference in four- and eight-process Gloo tests covering 1,867,500 dense logits, 352 tail rows, NaNs, saturation ties and padded ranks. Adjusted requests exercised ordinary sampling. Twelve offline SM90 builds reported zero spills. The 48-case graph and model replay verified raw and full-vocabulary routing and retained all 57,600 target-tap values. The 96-round scheduler replay tested publication, rejection and the next decode step. [Distributed results](independent_gloo.md) · [Graph entry](graph.json) · [End-to-end scheduler fixture](rounds.json)

The official export is 277,284 bytes and uses the same six flags as V202. The full and relative patches reverse on the pinned SGLang tree. All 720 flag orderings passed, and the public positional field order was checked against the pin. V203 has no official throughput or output-agreement result. V64 remains the measured four-load reference at 1.4119×. [Manifest and flags](validation.json) · [CLI replay](flags.json) · [V64 score](../../runs/abd217baf0ed49fd.json)
