# Build every public FFI registration and both formats of the actual route packer.
set -eu
jit=python/sglang/kernels/jit
toolkit="$HOME/nvcc126/toolchain/usr/local/cuda-12.9"
export PATH="$toolkit/bin:$PATH"
cat > wrapper.cu <<'CU'
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include "decode_gate_up_launch.cuh"
namespace sglang {
using K = MuseDecodeGateUp;
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, (K::prepare<false>));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, (K::run<false>));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_prompt, (K::prepare<true>));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_prompt, (K::run<true>));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure, K::configure);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_wave, K::prepare_wave);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_wave, (K::run<true,true>));
}
CU
nvcc -arch=sm_90a -std=c++20 -O3 -DSGL_CUDA_ARCH=900 --expt-extended-lambda --expt-relaxed-constexpr \
  "-I$HOME/nvcc126/ffi/tvm_ffi/include" "-I$HOME/nvcc126/compat" "-I$jit/include" -Xcompiler=-fPIC -c wrapper.cu -o wrapper.o
python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
sym=subprocess.check_output(["nm","-g","--defined-only","wrapper.o"],text=True)
exports=sorted(row.split()[-1] for row in sym.splitlines() if " T __tvm_ffi_" in row)
assert set(exports)=={"__tvm_ffi_"+n for n in ["prepare","run","prepare_prompt","run_prompt","configure","prepare_wave","run_wave"]}
d=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
print(json.dumps(dict(arch="sm_90a",nvcc=subprocess.check_output(["nvcc","--version"],text=True).splitlines()[-2],exports=exports,wrapper_sha256=d("wrapper.cu"),header_sha256=d("decode_gate_up_launch.cuh"),object_sha256=d("wrapper.o"),executed_on_cuda=False),indent=2))
PY
