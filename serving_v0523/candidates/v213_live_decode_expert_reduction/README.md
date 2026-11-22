# V213 live partials for grouped expert decode

V213 reduces synchronization and padded work in V211's ordinary Humming expert projections. It applies during native target verification, including single-request verification. Its official patch SHA-256 is `6223b3cf65a3ac64ef46c45adb574621a8568e358b02c21d0815aa869f6c233f` at tree `0cb08b65dd155026a1c48dfaabf6a48f5afe4c6c`. The published-rule and bidirectional reconstruction checks passed with V211's launch flags. [Official patch](patch.diff) · [Delta](delta.diff) · [Validation](validation.json)

## Reduction and work

Humming assigns two K warpgroups to the gate/up output and four to the down output. Its ordinary reducer stages partial sums in shared memory and synchronizes all 512 math threads multiple times. V213 lets each nonowner publish its live FP32 fragments together. After one math-group barrier the owner computes `p0+p1` for gate/up or `(p0+p2)+(p1+p3)` for down. This preserves the original reduction tree and avoids writing an intermediate partial back through shared memory. A four-float fragment covers two adjacent token rows. A final odd live row can share one fragment with its padded neighbor. [Submitted reducer](delta.diff) · [Native reducer comparison](partials.json)

The producer completes its current job before the epilogue and waits for math completion before it can refill the stages for another job. The new exchange uses a suffix of that retired shared stage arena. Static size and offset assertions protect the live scheduler metadata and reserve a disjoint prefix for the original output writer. Both the W13 BF16 rounding and SiLU epilogue and the W2 consumer keep their original order. The fast path requires the ordinary single-CTA, full-K, non-stream-K schedule and falls back through the earlier paired kernels if the compiler, selected image or layout fails the checks. [Guarded constructors](delta.diff) · [Startup fallback trials](startup.json)

At medium load, the same persistent CTA additionally selects an 8-, 16- or 24-row tensor instruction once per expert block from the scheduler's *actual* remaining live rows. The native 8-row schedule remains active at 1–8 users. The new owner reduction still saves shared traffic and synchronization there. V208 targeted 176-row prompt CTAs with one K owner. V213 handles the two- and four-way K merge in decode and leaves every V208 prompt CUDA source byte-identical. V173 and V174 use a separate packed projection and tensor-core implementation. [Prompt identity](prompt_identity.json) · [Earlier medium implementation](../v174_multirow_expert_tma/README.md)

| Users | Native M tile | Modeled padded MMA work removed | Nominal FP32 shared partial transfer per rank per layer, V211 → V213 | Packed bank requests |
|---:|---:|---:|---:|---:|
| 1 | 8 | 0% | 8.97 → 2.26 MiB | 193.8 MiB |
| 4 | 8 | 0% | 25.52 → 7.18 MiB | 551.2 MiB |
| 16 | 16 | 49.01% | 78.63 → 22.50 MiB | 849.2 MiB |
| 32 | 24 | 52.98% | 119.96 → 42.50 MiB | 863.8 MiB |

The model assumes eight verification rows per user, distinct uniformly selected top-eight experts per token and 32 local experts of 256 on each rank. It counts shared stores and reads at the FP32 K merge. Packed weights and scales still make the same sweeps. At 16 users the modeled 849.2 MiB corresponds to about 185.5 μs at 4.8 TB/s, if all those requests reach HBM. The measured related V68 control reached 17,512.9 tokens/s/GPU at 16 users. There is no H200 timing, physical cross-width rounding test or competition token-agreement result for V213. Neither fewer tensor rows nor fewer barriers alone predicts that serving rate. [Model](work.json) · [Matched serving control](../v68_grouped_decode_verify/README.md)

Applying both schedules to two **observed H200 route histograms** gives a second comparison. At 32 verification rows the rank had 39 local routes. The issued tensor rows stay at 128, while the modeled shared partial transfer falls from 20.00 to 7.50 MiB. At 256 rows it had 354 live routes, including two experts with 73 and 61 rows. Issued tensor rows fall from 792 to 456, a 42.42% reduction, and shared partial transfer falls from 123.75 to 57.19 MiB. The actual archived kernels ran in 128.576 and 214.304 µs and reached 74.43% and 75.06% of sustained DRAM throughput. Those counters belong to a separate standalone fused implementation with cache flushing. Keeping its measured bytes and assuming 4.8 TB/s gives 95.90 and 161.24 µs, or only 1.34× and 1.33× against its timings at ideal bandwidth. This comparison illustrates the importance of the weight traffic and supplies no directly measured V213 or full-serving speedup. [Recorded-histogram replay](physical.json) · [Counter archive](../../../experiments/iterations/20261104-moe-hardware-profile-recovered.json)

## Compiler and correctness checks

NVCC 12.9.86 compiled six actual SM90a modules from Humming 0.1.12. Each selected K loop retains 32 HGMMA sites, eight groups of four, and shared-stage releases after sites 8, 16, 24 and 32. All six modules retain 96 registers/thread, the original 82,944–123,904 bytes of shared memory, and zero reported spills and stack. The compiled FP32 add mode matches the incumbent. The extra paths increase some cubins substantially, up to 114,536 bytes, and can affect instruction-cache behavior. The complete 24-row W2 reduction now uses one shared merge barrier per tile in place of eight. Entry and output publication barriers remain. [Machine evidence](sm90a.json)

- The CPU fixture executed the actual Humming reducer and the submitted reducer with all 512 logical threads in randomized order between barriers. It compared 1,708,032 live FP32 words across 792 cases. A linear K-group sum changed 90,989 results on these inputs. The original binary tree matched bit-for-bit, including 105,843 zeros and 164,754 nonfinite outputs. [Coroutine fixture](partials.py) · [Results](partials.json)
- For every native shape, the emitted K-slice arithmetic is byte-identical to V211 after substituting the selected vendor FMA type. Every narrow FMA body matches the vendor's native instruction for that width. An independent map followed 181,248 live components through Humming’s XOR-4 register transpose into its shared writer. It checked each against the producer and owner masks. [K ordering](k_order.json) · [Writer map](writer.json) · [PTX accumulator layout, Figure 152](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n16)
- The actual grouped adapter passed 144 end-to-end, 1,800 route-planning and 168,471 BF16 comparisons with selected, missing and disabled variants. A separate full-pair-to-SiLU fixture exercised the existing Pytorch-facing gate/up numerical boundary. [Dispatch and finalization](decode.json) · [BF16 activation](stock_gate_boundary.json)

To reproduce, prepare `$scratch/sglang` with the official applied patch and extract the pinned Humming 0.1.12 wheel into `$scratch/humming`. Run the renderer with CPU PyTorch and Jinja. Use the paths for the local CUDA 12.9 toolchain and its host compatibility headers in the compile command.

```sh
python render.py "$scratch"
sh compile.sh "$cuda_bin" "$scratch/humming/include" "$compat_include" "$scratch"/{base,live}/*.cu
python isa.py "$scratch"
python k_order.py "$scratch"
python partials.py "$scratch"
python writer.py "$scratch"
python work.py
python verify.py "$scratch/sglang"
```

The isolated source and compiler scratch were retired after matching the export, checking Git state and confirming there were no open handles. The local compiler VM is stopped. [Host cleanup](host_cleanup.json) · [Guest cleanup](guest_cleanup.json)
