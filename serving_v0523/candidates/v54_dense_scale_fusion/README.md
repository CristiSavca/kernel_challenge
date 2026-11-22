# V54 fuse dense expert output scaling

Pinned SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`. Parent candidate
V50 SHA-256 `7ba1708e049dbfae89eac613efe0bdeb9b7c8ccbcf0c11859ccad49f7f4441ae`.
This patch SHA-256
`44e7e293f7183361e9448740da93d15ae66265dde0a862c7f1b6240ff2e16bd7`.
Compare at V50's flags, including `swa-full-tokens-ratio=0.25`.

V50 scales the BF16 GEMM1 gate/up rows and GEMM2 output rows in two separate
PyTorch operations. V54 does those multiplications in the existing CUDA SiLU
and mapped Triton finalizers. Both kernels multiply in FP32, round to BF16 as
the standalone store did, and read that rounded value into their existing
arithmetic. The new CUDA specialization uses scalar `mul.rn.f32` to retain
subnormals under the activation library's fast-math compilation. The Triton
finalizer keeps zero in masked columns, including when a scale is nonfinite.
Existing activation and finalizer callers specialize with row scaling disabled.
The graph bridge, route map, expert weights, original slot order and collectives
are the same as V50.

At 16,224 local routes, the two standalone scaling passes read and write about
761 MiB per MoE layer. The fused form removes those full-row transfers and adds
small scale loads in their consumers. Expected direct gain is roughly 0.5–1%
of overall score if the removed traffic was bandwidth-limited. CUDA
compilation, graph replay, memory and throughput are still unmeasured.

Verification ran the exact V50 and V54 eager Python bodies with CPU BF16 GEMMs
and modeled CUDA and Triton operations. Compact down-projection rows after
scaling and combined outputs agreed bitwise for 60 cases. Cases covered dropped,
repeated and unbalanced routes, both RMSNorm orders, strided inputs and
positive, negative and zero scales. A separate test compared every BF16 input
encoding against 33 scale encodings, 2,162,688 pairs in total, with exact
non-NaN bits and NaN classification against PyTorch's standalone multiply.
The Triton AST confirmed that the new scale block leaves the original slot
mapping, norm and accumulation arithmetic intact. The official CLI exported
the 35,001-byte patch, and an exact pinned-base cached apply and whitespace
check passed. Five Python sources compiled locally. The patch covers seven
allowed files with a maximum added line length of 121 bytes.

Use a free check on the exact patch before scoring. Inspect JIT startup,
correctness and peak GPU memory as well as throughput against a measured V50
comparison at the same load.

Independent source review found no blocking issue. After comparing the frozen
patches and inspecting both temporary Git indexes, their stash lists and open
files, the V50/V54 scratch directory `okc-v50.c4u2B7RR` was retired. It occupied
3,556 KiB. Available space on the shared volume rose from 35,992,256 KiB to
35,995,804 KiB, an observed gain of 3,548 KiB.
