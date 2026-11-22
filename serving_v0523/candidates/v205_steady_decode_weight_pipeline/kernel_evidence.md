# V205 grouped decode down projection

V205 places four ordered WGMMA instructions in each group throughout the 4,096-wide K reduction. It retains the incoming BF16 activations, prepared packed W2 bank, E4M3 scales, TMA descriptors, four K-warps and final in-block reduction. Each of the four warpgroups now makes 32 waits per complete K reduction, compared with 48 in V204. The number of native GEMM launches, requested weight bytes and mathematical operations is identical. [Actual tuner configurations](configs.json) · [CUDA machine-code audit](sm90a.json)

| Tuner row tile | Relevant users with eight draft rows | Outstanding BF16/bit-operation sites after two HGMMA, V204 → V205 | Total hot-loop sites, V204 → V205 | Shared bytes |
|---|---|---:|---:|---:|
| 8 | 1, 2, 4, 8 | 380 → 452 | 1,142 → 1,145 | 91,136 |
| 16 | 16 | 435 → 596 | 1,140 → 1,141 | 107,520 |
| 24 | 32 | 399 → 434 | 1,168 → 1,169 | 123,904 |

The opportunity is to execute operand preparation before the matrix operations finish. The count is static, not elapsed time. All six matched SM90a cubins retained 32 HGMMA sites per four-stage hot loop, 96 registers, three barriers and zero spills. Longer retention of the current activation tile can delay a producer. H200 throughput and token agreement remain unmeasured. [Matched SASS and lifetime results](sm90a.json)

The producer begins with three published stages. At current stage s, its next release-dependent transfer writes s+3 into the four-slot ring. The consumer therefore may wait for s+1 between the third and fourth K16 slice while continuing to hold s. All 512 math workers release s only after their fourth K16 instruction pair drains, protecting the live shared A addresses. The final K slice skips the unused next-stage prefetch. Subsequent expert jobs retain the vendor barrier phases, rank-local expert offsets and reduction. [Production changes](delta.diff)

The portable C++ run exercised the submitted method with delayed matrix reads at all four K-warp offsets. It preserved all 13,824 FP32 outputs across 864 cases and 1,062,912 delayed register-source checks. Startup generated and registered the three full-pair cubins. An unsuitable full-pair binary fell back to the original single-pair implementation, and a second rejection selected ordinary Humming. The original W2 fallback cubins and all four W13 prompt/decode cubins matched their archived binary hashes. [Arithmetic replay](cpu.json) · [Startup and fallback](startup.json) · [W13 binary comparison](gate_identity.json)

For offline reproduction on the pinned SGLang tree and Humming 0.1.12 wheel, place them in SCRATCH/sglang and SCRATCH/humming and run `python render.py SCRATCH`. Use the [SM90a compilation script](../v169_grouped_expert_tile_cohorts/compile_sm90a.sh) on SCRATCH/paired and SCRATCH/paired_twice. Run `python isa.py SCRATCH`, `python startup.py SCRATCH`, and `python cpu_check.py SCRATCH/sglang`. For the W13 hash comparison, run the [actual W13 renderer](../v197_grouped_prompt_gate_up/render.py) on SCRATCH, compile SCRATCH/final and run `python gate_identity.py SCRATCH`. [Official export and rules](validation.json)
