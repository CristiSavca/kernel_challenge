# V99 adaptive DFLASH proposals

V99 aims to increase the number of greedy tokens committed per target pass. V93 already constructs a seven-token chain and a branching tree from the same draft forward. V99 chooses a proposal for each request using its recent verification results. Both proposals use the checkpoint’s eight positions and the same eight target rows. The first eligible step uses V93’s tree. [Full patch](patch.diff) · [Five-file change from V93](delta.diff)

## What one verification reveals

If the target accepts two proposed tokens and emits a bonus, the verifier now knows three consecutive greedy tokens. An alternative that misses the second token would have committed two, including its own bonus. One that matches all three would have committed at least four, provided there is room in the block. An alternative that first misses the third token ties. The same comparison works at any depth, with both rewards capped by the physical block width. The target can therefore establish which proposal would have won even when the winning margin is unknown.

The selecting tensor-parallel rank updates one signed 32-bit score for each request. A step moves its prior score one unit toward zero, adds two units for a chain win or subtracts two for a tree win, then clamps the result to the block width in either direction. A positive score selects the chain. Ties decay past wins and losses. This is a heuristic for predicting the next step. Changes in the request’s token distribution and the extra GPU launches can reduce throughput.

The existing tensor-parallel broadcast distributes all four planes of the selected tree. The target evaluates its tokens, positions and ancestor mask. Its greedy predictions control acceptance and the bonus token. The comparison assumes the same target argmax at a shared logical prefix. The judge still has to establish output agreement under real GPU arithmetic. The score contains no token IDs and resets for new prefills, ineligible decodes and reused request slots. The ordinary path handles sampled requests, grammars and logit modifications. The feature requires the parent’s supported DFLASH backend and an explicit launch flag. [Request-state review](lifecycle_review.md) · [Rules](https://openkernelchallenge.dev/rules)

## Evidence

- An independent oracle checked 12,832 exhaustive comparisons. Selection, target acceptance and score update then ran together on CPU for 1,008 requests at widths two through eight. The tests include duplicate tokens, multiple valid children, a full block, a mismatch at the root, and poisoned output past the real commit length. [Oracle and results](independent_counterfactual.json)
- Two real Gloo ranks exercised the worker’s selection and broadcast with deliberately different local proposals. The target received the selecting rank’s tokens, parents, positions and ancestry for all eight eligible batches. Four ineligible batches reset their staged metadata. CPU request-state tests covered 160 overlap cycles and 128 direct filter or merge operations. [Distributed test](worker_gloo.json) · [Lifecycle test](lifecycle_cpu.json) · [Eligibility tests](worker_check.json)
- Fourteen Triton specializations for Hopper compiled without spills. The inherited tree suite passed 1,400 proposals, 9,800 acceptance and slot permutations, 707 replay resets, 72 attention cases and 32 fallback cases. The official export contains 218,514 bytes across 38 permitted files. Its SHA-256 is `b481fac41f1b182a0552ebb381e3305f201a13b9005639906094c6655c595bb4`. [Compiler results](sm90.json) · [Export verification](validation.json)

A synthetic example illustrates why either policy can win. With a 60/40 draft selector and 60% target agreement on the leading choice, the tree commits an expected 3.216 tokens and the chain 2.458. At 94% agreement the tree commits 3.831 and the chain 6.507. These figures enumerate constructed target continuations. No serving acceptance distribution has been measured for V99. [Reproducible example](spine_example.json)

## Official comparison

The earlier November 19 allocation can spend its third private slot on V99 at 16 users if the exact V93 check passes and its shared dependencies remain eligible. The only change to V93’s launch configuration is `--speculative-dflash-adaptive-tree`. Otherwise the runner retains its existing width-12 V79a experiment under the same slot limit. It also retains the V98 versus V93 prefill comparison. [Armed runner](../../runs/nov19_decode_runner.launch.json)

Compare committed length and tokens/s at matched concurrency, plus first-turn and later-turn latency, output agreement, lengths and peak memory. V64’s measured four-load score is 1.4119×. At 16 users it averaged four committed tokens per step. Even perfect eight-token commits only double that decode rate if step time stays fixed. Reaching a 4× four-load score requires gains elsewhere as well. [Official V64 result](../../runs/abd217baf0ed49fd.json)

The November 18 10:45 UTC runner update assigned V99’s conditional slot to V101b, which retains V99’s binary preference whenever an intermediate branching policy lacks a strict advantage. [V101b](../v101b_endpoint_guarded_spine/README.md) · [Updated allocation](../../runs/nov19_decode_runner.launch.json)
