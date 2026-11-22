set -eu
jit=python/sglang/kernels/jit
src="$jit/csrc/muse_spark/expert_activated_down.cuh"
cat > down.cu <<WRAP
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "$src"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, (MuseActivatedDown::run));
}
WRAP
nvcc=toolkit/usr/local/cuda-12.9/bin/nvcc
"$nvcc" -std=c++20 -O3 -arch=sm_90 -DSGL_CUDA_ARCH=900 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Iffi/tvm_ffi/include -I"$jit/include" -Xptxas=-v -Xcompiler=-fPIC -c down.cu -o down.o
"$nvcc" -std=c++20 -O3 -arch=sm_90 -DSGL_CUDA_ARCH=900 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Iffi/tvm_ffi/include -I"$jit/include" --ptx down.cu -o down.ptx
toolkit/usr/local/cuda-12.9/bin/ptxas -arch=sm_90 -v down.ptx -o down.cubin
sha256sum "$src" down.o down.ptx down.cubin
