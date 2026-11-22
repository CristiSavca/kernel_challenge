set -eu
# Run from an empty scratch directory with a patched SGLang checkout.
stage=$(cd "$1" && pwd)
jit="$stage/python/sglang/kernels/jit"
: "${CUDA_HOME:?set to a CUDA toolkit, tested with 12.9.86}"
: "${TVM_FFI_INCLUDE:?set to apache-tvm-ffi 0.1.11 tvm_ffi/include}"
export PATH="$CUDA_HOME/bin:$PATH"
cat > wrap.cu <<'WRAP'
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "muse_spark/expert_cluster_mlp.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, (MuseClusterMLP::run));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(supported, (MuseClusterMLP::supported));
}
WRAP
for arch in sm_90 sm_90a; do
  set -- -std=c++20 "-arch=$arch" -O3 --expt-extended-lambda --expt-relaxed-constexpr --use_fast_math \
    -DSGL_CUDA_ARCH=900 "-I$TVM_FFI_INCLUDE" "-I$jit/include" "-I$jit/csrc"
  if [ -n "${COMPAT_INCLUDE:-}" ]; then set -- "$@" "-I$COMPAT_INCLUDE"; fi
  nvcc "$@" -Xcompiler=-fPIC --compile wrap.cu -o "$arch.o"
  nvcc "$@" --ptx wrap.cu -o "$arch.ptx"
  ptxas -v "-arch=$arch" "$arch.ptx" -o "$arch.cubin"
  cuobjdump --dump-sass "$arch.cubin" > "$arch.sass"
done
