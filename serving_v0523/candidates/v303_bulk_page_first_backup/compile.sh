set -eu
jit=python/sglang/kernels/jit
toolkit="$HOME/nvcc126/toolchain/usr/local/cuda-12.9"
export PATH="$toolkit/bin:$PATH"
cat > wrapper.cu <<'CU'
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "kvcacheio/staged_write_back.cuh"
namespace sglang {
using Target = HiCacheStagedWriteBackKernel<128,4,2,1024>;
using Draft = HiCacheStagedWriteBackKernel<256,4,2,1024>;
TVM_FFI_DLL_EXPORT_TYPED_FUNC(target_mha, (Target::run_all_lf_pf_staged));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(target_bulk, (Target::run_all_lf_pf_bulk));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(draft_bulk, (Draft::run_all_lf_pf_bulk));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(target_mla, (Target::run_all_mla_lf_pf_staged));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(draft_mha, (Draft::run_all_lf_pf_staged));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(draft_mla, (Draft::run_all_mla_lf_pf_staged));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(unified_mha, (HiCachePageUnifiedWriteBackKernel<128,false>::run));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(unified_mla, (HiCachePageUnifiedWriteBackKernel<128,true>::run));
}
CU
for arch in sm_90 sm_90a; do
  nvcc "-arch=$arch" -std=c++20 -O3 --use_fast_math -DSGL_CUDA_ARCH=900 --expt-extended-lambda --expt-relaxed-constexpr \
    "-I$HOME/nvcc126/ffi/tvm_ffi/include" "-I$HOME/nvcc126/compat" "-I$jit/include" "-I$jit/csrc" \
    -Xcompiler=-fPIC -c wrapper.cu -o "$arch.o"
done
python3 - <<'PY'
import hashlib,json,pathlib,subprocess
source=pathlib.Path("python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
objects={}
for arch in ("sm_90","sm_90a"):
 p=pathlib.Path(arch+".o")
 names=subprocess.check_output(["nm","-g","--defined-only",p],text=True)
 exports=[i.split()[-1] for i in names.splitlines() if "__tvm_ffi_" in i and " T " in i]
 assert len(exports)==8 and all("__tvm_ffi_"+x+"_bulk" in exports for x in ("target","draft")),exports
 sass=subprocess.check_output(["cuobjdump","--dump-sass",p],text=True)
 syms=subprocess.check_output(["nm","-u",p],text=True)
 assert "cudaMemcpyAsync" in syms and "hicache_relayout_kernel" in sass
 objects[arch]=dict(bytes=p.stat().st_size,sha256=sha(p),exports=exports,has_native_relayout=True,
      cudaMemcpyAsync_external=True,sass_sha256=hashlib.sha256(sass.encode()).hexdigest())
print(json.dumps(dict(cuda=subprocess.check_output(["nvcc","--version"],text=True).splitlines()[-2],
 source_sha256=sha(source),wrapper_sha256=sha(pathlib.Path("wrapper.cu")),objects=objects,physical_GPU=False),indent=2))
PY
