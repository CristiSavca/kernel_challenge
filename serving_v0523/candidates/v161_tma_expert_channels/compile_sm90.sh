#!/bin/sh
set -eu
# Give an assembled, materialized V161 SGLang tree, a bounded empty output
# directory, the toolkit root and the TVM-FFI include directory on Linux.
stage=$1
out=$2
toolkit=$3
ffi=$4
jit=$stage/python/sglang/kernels/jit
mkdir -p "$out"
cat > "$out/kernel.cu" <<'CU'
#include "csrc/muse_spark/expert_tma_route_pair.cuh"
#include <tvm/ffi/function.h>
using namespace sglang;
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, MuseTmaRoutePair::prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, MuseTmaRoutePair::run);
CU
for arch in sm_90a sm_90; do
    "$toolkit/bin/nvcc" -arch="$arch" -std=c++20 -O3 -DSGL_CUDA_ARCH=900 --use_fast_math --expt-relaxed-constexpr \
        -I"$jit" -I"$jit/include" ${HOST_COMPAT:+-I"$HOST_COMPAT"} -I"$ffi" -Xptxas=-v \
        -c "$out/kernel.cu" -o "$out/$arch.o" > "$out/$arch.log" 2>&1
    "$toolkit/bin/cuobjdump" -sass "$out/$arch.o" > "$out/$arch.sass"
done
