#!/bin/sh
set -eu
cd /tmp/v173
toolkit=$HOME/nvcc126/toolchain/usr/local/cuda-12.9
jit=python/sglang/kernels/jit
mkdir -p common
tar xf pinned.tar -C common
for role in prior new; do
    mkdir -p "$role/out" "$role/$jit/csrc/elementwise"
    cp "common/$jit/csrc/elementwise/activation.cuh" "$role/$jit/csrc/elementwise/activation.cuh"
    for module in tma medium; do
        if [ "$module" = tma ]; then
            cat >"$role/out/$module.cu" <<'CU'
#include "csrc/muse_spark/expert_tma_route_pair.cuh"
#include <tvm/ffi/function.h>
using namespace sglang;
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, MuseTmaRoutePair::prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, MuseTmaRoutePair::run);
CU
            if [ "$role" = new ]; then
                echo 'TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_medium, MuseTmaRoutePair::run_medium);' >>"$role/out/$module.cu"
            fi
        else
            cat >"$role/out/$module.cu" <<'CU'
#include "csrc/muse_spark/expert_medium_route_pair.cuh"
#include <tvm/ffi/function.h>
using namespace sglang;
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, MuseMediumRoutePair::run);
CU
        fi
        for arch in sm_90a sm_90; do
            out="$role/out/$module.$arch"
            "$toolkit/bin/nvcc" -arch="$arch" -std=c++20 -O3 -DSGL_CUDA_ARCH=900 --use_fast_math --expt-relaxed-constexpr \
                -I"$role/$jit" -I"common/$jit/include" -I"$HOME/nvcc126/compat" -I"$HOME/nvcc126/ffi/tvm_ffi/include" \
                -Xptxas=-v -c "$role/out/$module.cu" -o "$out.o" >"$out.log" 2>&1
            "$toolkit/bin/cuobjdump" -sass "$out.o" >"$out.sass"
        done
    done
done
