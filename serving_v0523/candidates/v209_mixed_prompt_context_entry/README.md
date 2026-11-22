# V209 restores two optimizations on the serving scheduler’s mixed forward

V206 admitted mixed rounds and ran their target, but its scheduler built a ForwardBatch and passed it directly to the TP worker. That bypassed both optimizations in the alternate DFlash worker helper. The target saw the default full-vocabulary setting and gathered all five raw draft-context taps. V209 sets the vocabulary intent on the actual mixed ForwardBatch and enters the sequence-owner projection scope around the actual target call. Its one-file delta applies directly to V206 with the same six launch flags. [Delta](delta.diff) · [Full official export](patch.diff) · [Validation](validation.json)

The projection evaluates the released draft FC and RMSNorm on each sequence owner before gathering the normalized features. The head still scores the full vocabulary on the group. It gathers complete logits for the ordinary prompt sampler and only score/token pairs for eligible greedy verification rows. Mixed settlement retains the prompt and accepted-prefix draft-KV writers. The scope explicitly disables paired raw KV because a running request may reject part of its proposal.

Full and Breakable capture only the layer body on this route. On each forward the outer eager Muse model consumes its returned taps and the current serving batch. Full copies live metadata and imports the captured body’s sequence-sharding verdict. Breakable’s prepared serving view copies the mixed-vocabulary flag. Eager fallback uses the same outer model. Graph configurations that capture the entire outer model fail mixed admission. Entering the context at scheduler replay cannot alter a previously captured body. The production projector also declines during active CUDA capture.

The sharded-logit gate checks target/head pre-hooks, post-hooks, global hooks and replaced or compiled forward methods. Writers and hooks that need the full vocabulary receive the existing mixed layout. Head eligibility participates in the existing cross-rank cohort signature. If ranks disagree, they select ordinary standalone verification before any combined target launches. The independent context gate coordinates rank-local hooks on the FC and RMSNorm. [Actual scheduler replay](entry_scheduler.py) · [Eligibility checks](head_eligibility.json)

## Scheduler evidence

The paired fixture ran the exact V206 and V209 mixed-forward source through scheduler selection, proposal construction, TP target, Muse outer model, logits processor, prompt sampling, acceptance, physical cache writes, publication and the next filtered batch. It used real four- and eight-process Gloo, synthetic BF16 model inputs and CPU emulation of device kernels, capture backends and memory pools. Across 168 complete rounds, all 84 comparisons matched IDs, acceptance, future lengths and the expected KV cache contents. It checked 10,330,560 physical cache words per TP4 rank and 20,064,960 per TP8 rank, including rejected slots. Eight additional distributed rounds declined inconsistent head availability or asymmetric head hooks with the original proposal intact. [Paired results](entry_scheduler.json)

On the TP8 Full fixture, two prompts and two running requests supplied 1,064 target rows. V206 returned synthetic hidden width 40 and vocab width 43 and gathered local tensors shaped 18 × 6. V209 returned width 8 context and width 6 local scores, then gathered 2 × 6 prompt scores and 16 × 2 tail pairs. Both committed the same five running tokens. Width-sensitive head and global hooks observed all 43 vocabulary values for each output row on both versions. The suite also exercised short rows, disabled projection, unsupported normalization, local and rank-asymmetric draft-FC hooks, wrapped/compiled heads and separate head availability.

The ordinary scheduler fixture additionally completed 96 mixed rounds with 1,008 committed tokens and six native standalone verification fallbacks. The unchanged production logits eligibility logic accepted 64 tested configurations and selected the full-vocabulary path in 1,872. [Ordinary replay](ordinary_scheduler.json) · [Producer guard matrix](head_eligibility.json)

## Performance exposure

At a full 16,384-row mixed batch the BF16 target-tap payload per rank drops from 1,280 MiB to 256 MiB. At TP8 the ideal per-rank receive-volume saving on that all-gather is 896 MiB. The released 40,960-to-8,192 projection requires about 11.00 TFLOP per rank when replicated. Computing one eighth of its rows reduces this to 1.37 TFLOP per rank before padding. The independent vocab saving for 31 running requests at width eight is 200,431,616 bytes of full-vocabulary tail output, with 15,872 bytes of gathered tail pairs per rank.

| Concurrent users | Reconstructed mixed opportunities | Owner-context opportunities | Raw → normalized context gathered per rank | Full tail vocab output avoided per rank |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 → 0 | 0 |
| 4 | 70 | 42 | 9.17 → 1.83 GiB | 1.06 GiB |
| 16 | 99 | 58 | 6.64 → 1.33 GiB | 7.86 GiB |
| 32 | 59 | 45 | 12.12 → 2.42 GiB | 7.67 GiB |

The table applies the candidate’s admission geometry to archived V41 request chronology. Those runs used a different scheduler. Real admission frequency, CUDA/NCCL behavior, per-rung throughput and TTFT still need measurement. At one user, this change has no opportunity to combine two request roles. [Trace and assumptions](../v202_mixed_target_serving/phase_budget.json) · [Vocabulary materializations](../v203_mixed_sharded_verify_head/head_budget.json)

The official export contains 294,834 bytes across 42 paths. Its SHA-256 is `5f99408f91c0b7c3ed04a9be3a7e85080f3f27a63bdb4215a0ecebcf0f429ff7`. Full and relative patches strictly apply and reverse on the current official pin. The live rules check covered path, suffix, content and flag restrictions. No official job was submitted for this component. [Manifest](validation.json)
