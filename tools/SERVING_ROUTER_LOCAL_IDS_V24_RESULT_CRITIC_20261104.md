# V24 independent terminal review — November 4

**Verdict: close the fixed attempt as failed/incomplete.** Run
`20261104T194033Z-c834b1b0` stopped at the frozen activation assertion. It did
not record a numerical mismatch, reach paired timing, or establish a speedup.
Keep the isolated candidate and evidence; no retry, changed criterion, serving
merge, or public promotion follows this result.

The recovered checkpoint SHA is
`ab6969de676de7cab5c7347237d68e640414d9659dc059403138670a3dd03092`.
Its six uploaded source hashes, image, policy and exact descriptor sequence
match the request and manifest. The archived worker is `65a2bdcc…`; archived
runner is `3c18303d…`. Both compiled modules used their distinct role/content
namespaces and exact owned headers. The 24 checked source dependencies match;
both builds recorded 438 dependencies and the same common-dependency digest.
Compiled binaries and complete dependency lists were not recovered, so these
are source-bound worker records rather than independent local binary checks.

| Evidence | Independently verified result |
|---|---|
| Fixed numerical cases | All 432 exact descriptors and checks pass: 384 nonempty, 48 empty; nine T values, eight ranks, six patterns |
| Numeric/ownership checks | Exact FP32 score bytes, global/local IDs, parent packed bits, unchanged inputs and disjoint input/output storage |
| Padding | All eight ranks pass active outputs, untouched padded scores and explicit −1 sentinels |
| Legacy optional API | All eight ranks preserve original mapped IDs, scores and packed bytes |
| T16 raw activation | Stock: router, casting copy, index; candidate: router only |
| T256 activation | Not reached |
| Streams / graphs / paired timing | Not reached; no result or speed gate can be computed |

The activation function required exactly two stock kernels and one candidate
kernel. The retained T16 error JSON instead contains three stock CUDA events:
`routing_kernel`, `direct_copy_kernel_cuda` with cast loads/stores, and
`index_elementwise_kernel`. The candidate has one routing event. All four
events have positive duration. The exception occurs before appending a
successful activation record, explaining the empty activation list. The raw
error agrees across checkpoint, stdout and stderr. Profiler duration sums are
not substituted for the planned warmed, alternating graph measurements.

The extra copy has a concrete source explanation. The exact stock chain does
`mapping[ids].contiguous()` with int32 IDs. The reported Torch version is
2.14.1+cu130; its matching primary source converts int32 advanced indices to
int64 before indexing, and the CUDA index kernel reads int64 indices.
This supports attributing the observed casting copy to index promotion.
It is not a direct trace of the copy's operands or proof of the installed
binary's exact implementation. Sources:
[PyTorch v2.14.1 index preparation](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/TensorAdvancedIndexingUtils.h#L93-L96),
[CUDA index reads](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/IndexKernel.cu#L99-L101).

**Review miss:** I accepted the exact two-kernel stock inventory without checking
this Torch index conversion. The failed assertion is overly restrictive; it
does not demonstrate an algorithm defect. That finding does not retroactively
pass the frozen screen. Retain the original source-review record alongside
this correction. High-level backend selection, SP, stream/graph ownership and
full-serving compatibility remain outside the completed GPU evidence.

The worker exited 1 after 30.768 seconds. Checkpoint recovery occurred inside
the live app context; the manifest records sandbox termination. The saved
19:42:07 provider receipt shows this app stopped with zero tasks and billed
cost zero. Root made no retry.

I independently reversed the current runner's three saved-floor literals and
reconstructed the archived runner exactly. The sole change raises the debit
from $4.21235928 to $4.22000000, bound to billing receipt `2ae77bac…`; current
runner SHA is `d5bff0165db68b7de45b1117f6b2c401372f2674dc593db16f969434d3ee6ed4`.
The previous 16 journal rows remain byte-identical to commit `9591ce1`; only
this run's full $1.78 reservation was appended. All 17 reservations total
$32.74. Unchanged corrections A=$3.74 and E=$9.51 leave $1.29 working allowance
after the saved $4.22 observation and separate $5 headroom. This is a dated
conservative accounting result, not permission to bypass fresh checks. No
refund, journal rewrite, guard change, or retrospective source change occurred.

The active MoE247888, decoder97af and attention87a9 file hashes remain exact.
The machine-readable companion is
`experiments/iterations/20261104-router-local-ids-result-critic.json`; it pins
all evidence and retains full raw event names and build records. No provider,
GPU or official calls were made during this review.
