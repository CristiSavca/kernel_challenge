#!/bin/sh
# Run in the compiler guest in the directory emitted by render.py.
set -eu
cd "$1"
for src in old/r*.cu new/r*.cu; do
    stem=${src%.cu}
    bin=$HOME/nvcc126/toolchain/usr/local/cuda-12.9/bin
    "$bin/nvcc" -std=c++20 -O3 --expt-relaxed-constexpr -DSGL_CUDA_ARCH=900 -arch=sm_90a \
        -Iinclude -I"$HOME/nvcc126/ffi/tvm_ffi/include" -I"$HOME/nvcc126/compat" \
        --compiler-options=-fPIC -Xptxas=-v -c "$src" -o "$stem.o" > "$stem.log" 2>&1 || { tail -30 "$stem.log"; exit 1; }
    "$bin/cuobjdump" -sass "$stem.o" > "$stem.sass"
    nm -g "$stem.o" > "$stem.symbols"
done
