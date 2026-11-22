# V52 dense prefill route offsets

V52 is V44 with one change to dense prefill routing. The stable sort returns
both IDs and slot indices. A CUDA lower-bound search finds the offsets for
local expert IDs 0 through the expert count, and one `.tolist()` transfers
those offsets for the eager GEMM loop. Adjacent offsets give the per-expert
route counts. The first offset skips the inactive prefix.

For nonempty int32 input, CUDA `bincount` reads the minimum and maximum back
to the host before launching its histogram. CUDA `searchsorted` on same-dtype
contiguous tensors launches on the current stream. The physically sorted IDs
avoid passing a `sorter` tensor, which would trigger its own scalar transfers.
Stable `argsort` already calls the same sort and discards the values. This
removes two serial device-to-host transfers per dense MoE layer, leaving the
transfer needed for the eager loop. The pinned SGLang source declares PyTorch
2.14.1. PyTorch 2.12.0 has the same two host reads in `bincount`.
[CUDA bincount 2.14.1](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/SummaryOps.cu#L234-L261),
[CUDA bincount 2.12.0](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/cuda/SummaryOps.cu#L234-L260),
[CUDA searchsorted](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/Bucketization.cu#L79-L159),
[sorter validation](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/BucketizationUtils.h#L95-L129),
[stable argsort](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/Sorting.cpp#L991-L996).

- SGLang base `297e54fd105e974cdf401e5b4d20253a22c16f90`.
- V44 patch SHA-256 `0822e1279e1710bbb30f2e0e541dcf0055a105e7bc1032ae5b081638c1d27b4c`.
- V52 patch SHA-256 `7198bb4cf02b9044afbfbda738b42bf125b2feffcdf590f7f63e7a8825594241`.
- Patch size 20,386 bytes across four allowed source files. The other three
  patch entries match V44 byte for byte.
- Comparison flag `--flag swa-full-tokens-ratio=0.25`.

CLI export, pinned-base application, reverse application, whitespace,
source rules, and Python syntax checks passed. PyTorch 2.14.1 CPU execution
of the extracted V44 and V52 code matched exact slot order, expert IDs, route
counts, and totals on 21,905 inputs. Cases included all combinations through
length seven of three experts and the inactive ID, empty and all-inactive
inputs, repeated and skewed experts, noncontiguous input, and up to 16,384
tokens with eight routes each. Thirty BF16 CPU comparisons also matched the
full dense body bit for bit with mocked CUDA dequantization and activation.

Judge throughput is unmeasured. The dense GEMM time dominates the operations
changed here, so the expected serving gain is modest. The sorted IDs remain
live in V52, adding 512 KiB for 16,384 tokens with eight int32 route IDs each.

## Scratch cleanup

Removed the verified temporary path `/private/tmp/okc-v52.wAKXLk` after checking
its Git state, open files, and saved patch. It held an alternate-object sparse
checkout and a temporary CPU-only PyTorch install. The directory occupied
728,244 KiB and shared-volume free space increased by 748,292 KiB between
measurements.
