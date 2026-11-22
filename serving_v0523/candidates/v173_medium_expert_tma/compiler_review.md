# Native Hopper build

NVIDIA CUDA 12.9.86 compiled the ordinary TVM-FFI wrappers of both V168 and V173 for SM90 and SM90a on an offline Linux aarch64 host. V173 exports prepare, run and run_medium from the TMA module and run from the standalone medium module. All eight module/architecture/variant builds succeeded, with zero stack and spills in every emitted device function.

| Projection | Registers per thread W13 / W2 | Static shared bytes per CTA W13 / W2 | Smallest MMA backedge, sites W13 / W2 |
| --- | --- | --- | --- |
| V168 and V173 low-load stream | 47 / 47 | 41,248 / 34,224 | 651 / 354 |
| V173 light medium-density stream | 39 / 36 | 41,248 / 34,224 | 652 / 354 |
| V173 heavy medium-density vector path | 121 / 118 | 0 / 0 | 318 / 165 |

Each of the nine original TMA module functions and eight original standalone medium module functions has identical encoded instructions on its matched architecture. The V173 heavy kernel has the original W13 330- and 318-site inner MMA backedges and W2 200-, 177- and 165-site inner backedges, including their scheduling words, bit for bit. The padded planner expands from 192 to 200 instruction sites at the original 23 registers. Light transfer uses the descriptor-driven producer. Its smallest consumer backedge contains 16 W13 or 8 W2 HMMA sites. Vector and streamed loops cover different tile widths, so their raw instruction counts cannot give a latency ratio. There was no physical hardware timing.

To reproduce the build in a new Linux scratch directory at /tmp/v173, reconstruct the V168 and V173 official patches against the pinned SGLang commit as separate source trees. Under prior/ and new/ respectively keep the three expert headers at their original python/sglang/kernels/jit/csrc/muse_spark paths. Create pinned.tar with git archive from the base commit 297e54fd105e974cdf401e5b4d20253a22c16f90 of python/sglang/kernels/jit/include and python/sglang/kernels/jit/csrc/elementwise/activation.cuh. The saved compile_sm90.sh gives the exact two wrappers, flags and toolkit/include locations used. Run it, then run python3 compiler_check.py /tmp/v173. The sm90.json receipt records per-section machine-code and per-file digests, input headers, register allocation, spills and arithmetic backedges. Compile durations in ptxas logs and temporary paths in host objects can vary across reruns.
