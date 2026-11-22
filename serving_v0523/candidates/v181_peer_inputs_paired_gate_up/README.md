# V181 joins peer prompt inputs to a fused expert projection

V181 combines two successive stages of prompt MoE execution. Each rank publishes its BF16 prompt shard, then reads the rows needed for its local experts. The grouped gate/up projection consumes the gathered buffer, computes both halves in one Humming output tile and writes the activated BF16 intermediate directly to W2. The existing sparse return sends the scaled expert contributions back to their owners. Both changes retain their independent admission and fallback paths. [V177 arithmetic and compiler checks](../v177_grouped_gate_up_epilogue/README.md) · [V179 peer transport and admission](../v179_peer_prompt_inputs/README.md)

For a 16,384-token prompt with uniform top-eight routing across eight ranks, the peer-input model removes 137.43 MiB of logical tensor traffic per eligible rank and layer. The modeled remote payload remains 74.13 MiB. At 16,384 local routed rows with 4,096 channels, the fused projection removes an additional 512 MiB of intermediate writes and reads and one activation launch. V179 reserves up to 16 MiB of symmetric storage per rank at that prompt size. V177 cuts projection workspace by 128 MiB for that many local routes. Allocation lifetimes, cache hits, NVLink barriers and throughput determine physical peak memory and time. V181 has no H200 throughput or token-agreement result. [Peer-input traffic](../v179_peer_prompt_inputs/traffic.json) · [Projection traffic](../v177_grouped_gate_up_epilogue/traffic.json)

A distributed CPU replay used an actual Gloo process group at both TP4 and TP8. Each rank traced the same tensor storage across the production peer read, the grouped gate/up adapter and the original sparse return, and the full model fixture compared the returned BF16 values with its owner reference. A second startup exercise declined peer access on one rank and verified group-wide fallback. CPU implementations supplied symmetric-memory transport, GPU GEMMs and finalizer arithmetic. The frozen CUDA and Triton implementations have separate machine-code tests. [Distributed replay](joint.json) · [Test source](joint_gloo.py)

| Ranks | Joined rank calls | BF16 components on joined returns | Routed rows reaching paired W13 from peer storage |
| ---: | ---: | ---: | ---: |
| 4 | 4 | 393,600 | 8,200 |
| 8 | 8 | 787,200 | 8,200 |

The official 220,541-byte export reproduces the same tree in both donor orders. All eight donor source files have identical contents to their frozen parents. The seven launch flags are inherited from V176. The source and flags passed the public static restrictions. Run `python verify.py SGLANG` on the integrated tree and `python joint_gloo.py SGLANG` with CPU PyTorch and Gloo to reproduce the checks. [Official patch](patch.diff) · [Manifest and hashes](validation.json) · [Reconstruction](verify.py)

Four archived H200 measurements timed the full Marlin expert operation at 7.188–7.394 ms for 16,224 local routes. The archives do not isolate the paired gate/up intermediate or time V181’s grouped Humming parent. [Hardware measurement scope](physical_comparators.md)
