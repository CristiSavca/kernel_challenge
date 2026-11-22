#!/bin/sh
# Run inside the CPU-only AOT VM from the rendered scratch directory. Supply
# the original SGLang JIT include directory as ./include.
set -eu
kit="${CUDA_HOME:-$HOME/nvcc126/toolchain/usr/local/cuda-12.9}"
ffi="${TVM_FFI_INCLUDE:-$HOME/nvcc126/ffi/tvm_ffi/include}"
compat="${CUDA_COMPAT_INCLUDE:-$HOME/nvcc126/compat}"
for entry in donor golden; do
    "$kit/bin/nvcc" -std=c++20 -O3 --expt-relaxed-constexpr -DSGL_CUDA_ARCH=900 \
        -arch=sm_90a --ftz=false --prec-div=true --prec-sqrt=true -I. -Iinclude \
        -I"$ffi" -I"$compat" --compiler-options=-fPIC -Xptxas=-v \
        -c "$entry.cu" -o "$entry.o" > "$entry.log" 2>&1
    "$kit/bin/cuobjdump" -sass "$entry.o" > "$entry.sass"
    "$kit/bin/cuobjdump" -xelf all "$entry.o"
done
sha256sum donor.1.sm_90a.cubin golden.1.sm_90a.cubin > cubin.sha
