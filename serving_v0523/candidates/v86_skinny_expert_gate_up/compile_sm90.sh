set -eu
jit=python/sglang/kernels/jit
src="$jit/csrc/muse_spark/expert_skinny_gate_up.cuh"
cat > skinny.cu <<WRAP
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "$src"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, (MuseSkinnyGateUp::run));
}
WRAP
nvcc=toolkit/usr/local/cuda-12.9/bin/nvcc
"$nvcc" -std=c++20 -O3 -arch=sm_90 -DSGL_CUDA_ARCH=900 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Iffi/tvm_ffi/include -I"$jit/include" -Xptxas=-v -Xcompiler=-fPIC -c skinny.cu -o skinny.o
"$nvcc" -std=c++20 -O3 -arch=sm_90 -DSGL_CUDA_ARCH=900 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Iffi/tvm_ffi/include -I"$jit/include" --ptx skinny.cu -o skinny.ptx
/tmp/py/triton/backends/nvidia/bin/ptxas -arch=sm_90 -v skinny.ptx -o skinny.cubin
sha256sum "$src" skinny.o skinny.ptx skinny.cubin
