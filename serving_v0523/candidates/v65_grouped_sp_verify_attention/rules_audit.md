# V65 serving rules and source audit

Reviewed 2026-11-16 UTC against the live official rules and the serving
CLI help. The public rules page renders client-side. Its own live bundle
`/assets/index-DITIunFP.js` had SHA256
`71ecccaaf78c1993beb26fe217a9e4c2f1b25f8725895b65879a156e493e5d18`.
The serving restrictions cover lossless execution, output agreement, fixed
flags, reviewable source, a 2 GB peak increase per GPU, and first-token time
within twice stock at each load. The organizers may rerun held-out workloads
and review entries for workload specialization and work shifted outside timed
requests. [Official rules](https://openkernelchallenge.dev/rules)

## Patch and flags

- Reviewed SHA256 `80e41ae32e748857053f553359f4d555cbe983534cff465f4910551723c30337`.
  The 62,373-byte patch changes eleven plain-text Python files in permitted
  `python/sglang/` paths. The longest added source line is 98 characters.
  The V64 hunks and both V62 attention hunks match their frozen inputs.
- The intended flags are `moe-runner-backend=humming`,
  `speculative-moe-runner-backend=marlin`, `enable-layernorm-sp`, and
  `swa-full-tokens-ratio=0.25`. Each is mutable under the official CLI's
  pinned serving template. V65 uses the stock model and draft checkpoints,
  context length, KV dtype, requested token counts, and target sampling. The SWA
  ratio affects allocation capacity while the attention mask sets visibility.

## Precision and dispatch

- Hopper indexed and grouped Humming reuse a single bank of original E2M1
  weights and E4M3 block scales. The scale transform matches Marlin's
  suppression of E4M3 subnormals. FP32 full-K accumulation disables Stream-K
  partial reductions. BF16 activation and output boundaries, global scaling,
  and original-slot per-expert RMSNorm and weighting are preserved. Grouped
  dispatch uses expected routes per expert and actual arena size checks.
  Cubin precompilation depends on matrix geometry during model load, and the
  runner executes every live request's routes. [Humming 0.1.12](https://pypi.org/project/humming-kernels/0.1.12/)
- Sequence parallelism runs only on supported ordinary prefill. The rank
  gathers restore token order before routing finalization and DFLASH auxiliary
  materialization. The reduction guard retains the original all-reduce when
  the incumbent could use FP32 or a different accelerated accumulator.
  Ordinary NCCL BF16 reduce-scatter and per-rank projection GEMM shapes can
  change rounding order. Numerics need comparison on the judge image.
- The attention change runs on compatible Hopper FA3 DFLASH target verify
  shapes. It reads live E4M3 Q/K/V with unchanged descales, all causal and
  sliding-window keys, and FP32 scores, maxima, partial reductions and merge.
  Pinned FA3 casts probabilities to its E4M3 MMA operand type. V65 uses BF16
  probabilities and widens E4M3 values exactly to BF16. This increases operand
  precision and can change results. Native FA3 handles unsupported masks,
  modes and geometry. The guard follows hardware, dtype, layout, and
  supported query geometry. [Pinned FA3 probability conversion](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1359-L1363)

The SP and attention guards apply to distinct model forward modes. Backend
selection uses the launch configuration and tensor geometry, and each
forward reads that invocation's routed tokens and attention positions.

## Judge gates

The source audit supports proceeding to an H200 private check for cubin
startup, actual capture and replay, first/later-turn output agreement and
token counts, peak GPU memory, TTFT and throughput.
The 384 MiB balanced prefill arenas and roughly 5.2 MiB live verify-attention
scratch are allocation estimates for separate phases. The combined graph-pool
and KV-cache peak needs measurement. First-token limits apply separately at
all scored loads. Serving score and final organizer review remain pending.
