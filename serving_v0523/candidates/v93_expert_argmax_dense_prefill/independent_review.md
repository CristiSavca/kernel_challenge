# Independent V93 composition and dispatch review

Starting from the pinned SGLang Git index, I applied V89 and then the frozen V90, V91 and V92 component deltas. The resulting full Git diff is byte-identical to V93 with SHA-256 `995c3f3c78750bbf99e6c3db3f8ab0fa8353be87cbffb07b1d15c49e6a4c0fdb`. This checks the entire source tree and hunk ordering, including shared base files.

V90’s activated-down environment option defaults to true. With V89’s Humming and sequence-parallel settings, eligible grouped experts enter its guarded W2 path and retain the original slot-ordered output. V92 has a shape- and capability-guarded call at the existing full-attention FA3 EXTEND site and needs no launch flag. Its FP8 pack is upstream of target verification and leaves the model’s KV allocation intact. [V90 audit](../v90_activated_expert_down/independent_review.md), [V92 audit](../v92_dense_fp8_prefill/independent_review.md).

V91 needs one additional flag, `--speculative-dflash-reduced-target-argmax`, on top of V89’s nine flags. The worker constructs its normal speculative synchronizer from `get_dp_tp_group()`. The new selection path requires the same group object as the target TP coordinator, world size above one and an NCCL process group. It then passes that coordinator’s `device_group` to the integer MAX. Only the resulting replicated decision suppresses the later broadcast in both tree and linear verification. Sampling, grammar, penalties and logprob requests retain the existing selection path. [V91 audit](../v91_reduced_target_argmax/independent_review.md).

The component evidence covers isolated source, compiler and CPU behavior. V93 still requires an H200 check on the combined patch for output agreement, throughput, memory and TTFT.
