# V309 bounds the compact prompt expert workspace

V309 reserves 264.109 MiB of BF16 expert workspace for an eligible 16,384-token TP8 prompt with top-eight routing and 4,096-wide input and intermediate tensors. V308 reserves 384 MiB. Both execute one fused projection wave for all five near-balanced routing counts archived at 8K, 12K and 16K. Both execute two waves for the archived skewed 16K prompt with 30,697 local routes. The route counts come from generated inputs previously run on a single H200. The comparison executes the candidate planners on the host. [Recorded routes and branch replay](physical_replay.json)

The allocation uses only tensor dimensions and the fraction of experts owned by the rank. It reserves a whole-token margin plus four times the square root of the equal-share route count, rounded up before multiplication. That allowance controls the first wave's capacity. The original GPU planner processes every overflow wave, so the row estimate cannot discard a route. More skew can cost extra projection launches. [Changes from V308](delta.diff)

| TP8 case | V308 arena | V309 arena | V309 fused route capacity | V309 waves if every route is local, fused / binding declined |
| --- | ---: | ---: | ---: | ---: |
| 8,192 prompt tokens | 192.164 MiB | 133.797 MiB | 8,556 | 8 / 12 |
| 16,384 prompt tokens | 384 MiB | 264.109 MiB | 16,896 | 8 / 12 |
| 24,576 prompt tokens | 384 MiB | 384 MiB | 24,569 | 9 / 13 |

At 16,384 tokens the 16,224-route fixture uses 253.5 MiB for V204's live fused inputs and activations. V309 reserves 10.609 MiB more on that fixture. V309's all-local case runs eight fused waves to V308's six. Eager warm-up also runs all scheduled waves. When the existing conditional-graph probe declines or a fused preparation is absent, V309 keeps V308's original reservation and schedule. If the later tensor-map binding fails, the ordinary projection uses the smaller allocation with its own safe row capacity. [Scenario sizes](geometry.json) · [Binding and allocation instrumentation](arena_replay.json)

The largest BF16 workspace reservation remains 384 MiB. The output, sorting temporaries, larger wave metadata, graph objects, other model buffers, allocator rounding and allocator reservations contribute to whole-server peak memory. V309 has no H200 correctness, full-server peak, first-token latency or throughput measurement. Its changed path serves large prompts. [Full patch](patch.diff) · [Matching V204 launch flags](flags.json)

## Verification

- 144 host cases exercised the actual PyTorch workspace allocation, fused and declined bindings, eager and modeled captured execution, 1/8/64-way ownership and all-local, all-remote, repeated-expert and ragged routes. They matched 42,624 BF16 output words. V308's 180-case joined regression also passed on V309, matching 53,280 words. [Numerical and storage replay](arena_replay.json)
- 27,841 admitted shape combinations obeyed the 384 MiB ceiling, whole-token storage bound, int32 key-span limit and 16-wave upper bound for each projection layout. The mixed scheduler preserved 1,008 committed tokens over 96 rounds. [Capacity sweep](geometry.json) · [Mixed regression](mixed_regression.json)
- The official CLI export is 285,174 bytes across 44 allowed source paths. Applying and reversing both patches recovered the published base, V308 and V309 trees exactly. Only the Python wave orchestration file changes relative to V308. [Reconstruction and public contract validation](validation.json) · [Published rule snapshot](public_contract.json)
