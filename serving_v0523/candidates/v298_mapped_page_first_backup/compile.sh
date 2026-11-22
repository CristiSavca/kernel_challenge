set -eu
jit=python/sglang/kernels/jit
toolkit="$HOME/nvcc126/toolchain/usr/local/cuda-12.9"
export PATH="$toolkit/bin:$PATH"
cat > wrapper.cu <<'CU'
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include "kvcacheio/hicache.cuh"
namespace sglang {
using Target = HiCacheKernel<128,4,2,1024>;
using Draft = HiCacheKernel<256,4,2,1024>;
TVM_FFI_DLL_EXPORT_TYPED_FUNC(target_copy, (Target::run_all));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(draft_copy, (Draft::run_all));
}
CU
nvcc -arch=sm_90a -std=c++20 -O3 --use_fast_math -DSGL_CUDA_ARCH=900 --expt-extended-lambda --expt-relaxed-constexpr \
  "-I$HOME/nvcc126/ffi/tvm_ffi/include" "-I$HOME/nvcc126/compat" "-I$jit/include" "-I$jit/csrc" \
  -Xcompiler=-fPIC -Xptxas=-v -c wrapper.cu -o wrapper.o 2> assembler.log
cuobjdump --dump-sass wrapper.o > instructions.sass
python3 - <<'PY'
from pathlib import Path
import hashlib,json,re,subprocess
p=Path("wrapper.o");sass=Path("instructions.sass").read_text();log=Path("assembler.log").read_text()
exports=[line.split()[-1] for line in subprocess.check_output(["nm","-g","--defined-only",p],text=True).splitlines() if " T __tvm_ffi_" in line]
assert len(exports)==2
kernels={}
for name,body in re.findall(r'Function : (\S+)(.*?)(?=Function : |\Z)',sass,re.S):
 if 'transfer_all_layer' not in name:continue
 assert 'STG.' in body and 'LDG.' in body
 kernels[name]={inst:len(re.findall(r'\b'+inst+r'\.',body)) for inst in ('LDG','STG','LDL','STL')}
assert len(kernels)==4 and all(x["LDL"]==x["STL"]==0 for x in kernels.values()),kernels
print(json.dumps(dict(arch='sm_90a',compiler=subprocess.check_output(['nvcc','--version'],text=True).splitlines()[-2],
    source_sha256=hashlib.sha256(Path('python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh').read_bytes()).hexdigest(),
    wrapper_sha256=hashlib.sha256(Path('wrapper.cu').read_bytes()).hexdigest(),object_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),exports=exports,device_images=kernels,
    resources=[line.strip() for line in log.splitlines() if 'Used ' in line or 'stack frame' in line],gpu_execution=False),indent=2))
PY
