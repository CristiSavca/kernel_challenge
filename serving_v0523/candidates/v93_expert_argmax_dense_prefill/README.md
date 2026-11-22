# V93 fused expert decode, target selection and dense FP8 prefill

V93 is the integrated 16-user private experiment on frozen V89. The full patch is 208,583 bytes across 36 source paths. The judge accepted it as job `75557c21595347de` at 07:23 UTC on November 18. [Submission receipt](../../runs/v93_nov18_check16.submit.txt) · [Exact patch](patch.diff)

| Component | Change in the serving path | Cost to measure |
| --- | --- | --- |
| V90 | Each eligible routed down projection applies the original rounded gate, up and SiLU in shared memory and writes the projected rows. One separate activation launch and its global intermediate disappear. | Each group of output columns evaluates the activation again. |
| V91 | Each rank evaluates every target vocabulary logit and its softcap. A signed 64-bit MAX all-reduce identifies the greedy token across ranks. | Performance depends on target-selection time and the NCCL collective. |
| V92 | Large full-attention prefill calls gather the actual cached FP8 bytes once for dense FA3. The runtime retains page-one KV storage and prefix reuse. | Packing adds a launch and up to 32 MiB of transient storage per eligible invocation. |

The combined tree matched an independent reconstruction from V89 and the three component deltas. The official export applied and reversed cleanly on the pinned SGLang source. All 31 modified Python files compiled. The ten launch overrides matched zero restricted patterns. Source review covered precisions, attended keys, routed experts, greedy decisions and the fallback paths. The combined source passed the inherited proposal, tree acceptance, replay and attention checks and the new expert, distributed greedy selection and FP8 pack tests. [Validation](validation.json) · [Combined tests](combined_cpu.json) · [Independent review](independent_review.md)

The H200 check will determine whether the combination improves the official 16-user result and passes peak memory, first-token latency, output length and agreement. V64 scored 1.5919× on this rung and 1.4119× over the four loads. Its private 16-user check indicated 1.412×, illustrating the uncertainty of comparing single-run checks with scored results. The four-load 4× objective requires a further 2.83× over V64. A 16-user result supplies one load point. [V64 scored run](../../runs/abd217baf0ed49fd.json) · [V64 private check](../../runs/ac89fddc4ee147ab.json) · [Serving rules](../../RULES.md)

V94 separately shares the activation across up to eight cooperating Hopper blocks. The first check uses V90's block-local path because an eight-block cluster trades seven redundant SiLUs for seven peer copies per activation vector and 65 cluster synchronizations per live tile at K=4096. [V94 comparison](../v94_clustered_expert_down/README.md)
