# V169 grouped scheduler ABI

Humming 0.1.12 receives contiguous BF16 row subviews for each cohort. Its grouped-contiguous launcher takes the current input’s row count and an int32 expert-offset vector of length E+1. It computes successive expert lengths by offset differences and uses the current cohort row count minus the last expert offset for the last expert. Repeated offsets represent empty experts. The TMA descriptor uses the sliced tensor’s data pointer. The real model’s contiguous, aligned row pitch preserves the required alignment at any whole-row cut.

A CPU fixture compiled those two native count expressions from the pinned header and ran them on the production V169 plans. In 170 groupings it verified 1,095,332 live rows, 145 two-cohort calls, 218 empty-last-expert cases and 139 cuts at row indices not divisible by sixteen. Every local route appeared exactly once and preserved its stable order within its expert. Widths 136, 152 and 176 all ran. [Fixture](independent_vendor.py) · [Results and source digests](independent_vendor.json)

Changing M changes the transposed WGMMA instruction width. The native K-loop extent, operand precision and scale/epilogue ordering stay fixed. CUDA instruction semantics and these host checks do not establish bitwise equality between different WGMMA shapes or measure the time saved. Device correctness and throughput remain open.
