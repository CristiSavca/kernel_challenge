# V237 graph output and fallback

V237 matched V235 in 70 paired CPU cases. Eligible target verification carried the captured FP32 multiplication scale through the normal logits processor and the full-graph runner's output slice. The test retained eight rows of a twelve-row capture, changed the processor's live scale, and then ran the actual DFLASH consumer branch. Greedy decisions matched. Sampling, penalties, logit bias, a custom processor, grammar and logprob requests recovered the complete baseline distribution. The shared graph output stayed bitwise unchanged on every fallback. [Results and source hashes](independent_graph.json)

The V217 capture setup admitted an uninstrumented target and declined one with a registered logits-processor hook. An overridden fused-gather checker observed the original scaled tensor. Mixed and eager forward, a replaced logits method, zero, absent and subnormal scales produced ordinary untagged logits. A positive unit scale also passed. [Reproducible replay](independent_graph.py)

This review used real CPU tensors through the exported Python methods, with stand-ins for the projection, transport and GPU launches. It checks graph provenance and consumer behavior. CUDA arithmetic and end-to-end serving time require separate evidence.
