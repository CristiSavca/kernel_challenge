#!/bin/sh
set -eu
work=$1
includes=$2
cd "$work"
bin=$HOME/nvcc126/toolchain/usr/local/cuda-12.9/bin
for name in GroupedResidualPromptGateUpKernel GroupedMulticastPromptGateUpKernel; do
    stem=images/$name
    "$bin/nvcc" -std=c++17 --ptxas-options=--register-usage-level=10,-v --use_fast_math \
        --diag-suppress=39,161,174,177,940,1444 -I"$HOME/nvcc126/compat" -I"$includes" \
        -gencode=arch=compute_90a,code=sm_90a -cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda \
        --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi,-fopenmp,-lgomp \
        "$stem.cu" -o "$stem.cubin" > "$stem.log" 2>&1 || { cat "$stem.log"; exit 1; }
    "$bin/cuobjdump" -sass "$stem.cubin" > "$stem.sass"
done
"$bin/nvcc" -std=c++20 -O3 -arch=sm_90a -DSGL_CUDA_ARCH=900 --expt-relaxed-constexpr -Iinclude -I"$HOME/nvcc126/ffi/tvm_ffi/include" \
    -I"$HOME/nvcc126/compat" --compiler-options=-fPIC -c wrapper.cu -o wrapper.o > wrapper.log 2>&1 || { cat wrapper.log; exit 1; }
nm -g wrapper.o > wrapper.symbols
