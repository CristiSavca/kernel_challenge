# V160 combines prompt processing and speculative decode verification

V160 lets a waiting prompt and the running decode requests share one target-model forward on a tensor-parallel server. It uses the real draft tokens, evaluates the target over every candidate position, and runs the native prompt sampler and speculative verifier. V64 holds the measured four-load score of **1.4119×**. V160 has CPU integration evidence. Its eight-H200 throughput, token agreement, peak memory and time to first token still require a judge run. [V64 score](../../runs/abd217baf0ed49fd.json) · [V160 patch](patch.diff)

Enable the candidate with `--speculative-dflash-mixed-prefill` alongside the four flags in the manifest. The flag defaults to off. SGLang's actual argument parser accepted the bare flag and retained the existing 546 positional fields in their original order. The official diff is 190,775 bytes on the published base. Both the full patch and the delta from V159 apply forward and reverse to the recorded Git trees. All 30 paths, added lines, file modes, prohibited strings and launch flags passed the published source checks. [Manifest and launch flags](validation.json) · [CLI replay](cli_cpu.json) · [Official rules](https://openkernelchallenge.dev/api/v1/serving)

## How a request moves through the scheduler

1. After preparing a prompt, the scheduler checks the entire running group for free KV capacity, eligible greedy sampling and compatible captured graphs. Tensor-parallel ranks compare their decisions before reserving the ordinary decode slots. Combined target rows must fit the sum of the separate padded graph sizes. The outer scheduler batch counts every prompt and verification row. [Admission changes](scheduler_delta.diff)
2. The overlap scheduler resolves the running requests' latest tokens and lengths. On CUDA with GPU-only lengths it queues a pinned copy of one 64-bit length per running request before launching the draft. The bridge waits for that copy's event and verifies that it belongs to this proposal. It then builds the actual causal target input from the live proposal. Each rank checks the physical row mapping, prefix bounds and captured graph keys before entering the target. [Readback changes](readback_delta.diff) · [Ordering and lifetime tests](readback_cpu.json)
3. On agreement, the target runs once. Prompt sampling, tail acceptance, draft-cache writes and host output processing use their native role-specific paths. A late veto sends the same proposal through ordinary verification and then runs the prompt. Both routes publish the real bonus tokens and committed lengths once, in the scheduler's original request order. The next iteration can filter requests and gather its inputs by physical request slot. [Scheduler execution changes](execution_delta.diff) · [V159 target bridge](../v159_mixed_verify_forward/README.md)

## What the checks establish

The end-to-end CPU replay exercised 96 admitted rounds through normal scheduler selection, target-row preparation, verification and publication. It checked 1,008 accepted tail tokens with shuffled request slots, stale host length estimates and both prefill graph backends. A separate 216-round scheduler test covers late vetoes, subsequent ordinary decode, both CPU-length paths and separate prompt/verify graph outcomes. The admission fixture tests 47 rejection conditions and six agreement patterns each with real two-, four- and eight-process Gloo groups. CUDA kernels, graph replay and transfer timing are emulated in these tests. [End-to-end results](independent_round.json) · [Scheduler and fallback results](execution_cpu.json) · [Admission and distributed results](scheduler_admission_cpu.json)

| Concurrent users in archived V41 run | Prompt passes with running decode | Whole groups that fit the graph budget | Running request appearances within those groups |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 |
| 4 | 112 | 66 | 164 |
| 16 | 238 | 99 | 1,305 |
| 32 | 347 | 55 | 1,150 |

The archived eight-H200 V41 run used different prompt settings. Its 220 qualifying passes show when a separate verification launch could have been avoided under those schedules. The two rank agreements, draft and head work, new memory traffic and changes to admission timing all affect the realized gain. The one-user trace had no opportunity to combine a prompt with another request. With the one-user ratio held at V64's value, the other three loads would need a further **4.01× geometric-mean gain over V64** to reach a 4× overall score. [Historical trace and graph geometry](../../../experiments/iterations/20261118-serving-mixed-graph-capacity.json)
