"""Render a weighted BF16 native CUDA entry from the exact PyTorch release sources."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen

REV = "5c4886908584029761b579af026dcfb627c84070"
BASE = "https://raw.githubusercontent.com/pytorch/pytorch/" + REV + "/"
ROOT = Path(sys.argv[1]).resolve()
STAGE = Path(sys.argv[2]).resolve()
SCRATCH = Path(sys.argv[3]).resolve()
SCRATCH.mkdir(exist_ok=True)
sources = {}


def read(path):
    with urlopen(BASE + path, timeout=30) as response:
        data = response.read()
    sources[path] = hashlib.sha256(data).hexdigest()
    return data.decode()


def between(text, start, stop):
    return text[text.index(start):text.index(stop, text.index(start))]


cuda = read("aten/src/ATen/native/cuda/layer_norm_kernel.cu")
device_utils = read("aten/src/ATen/cuda/DeviceUtils.cuh")
math_compat = read("c10/cuda/CUDAMathCompat.h")
constants = read("aten/src/ATen/native/cuda/thread_constants.h")
dispatch = read("aten/src/ATen/native/layer_norm.cpp")
for path in ("torch/headeronly/macros/Export.h", "torch/headeronly/macros/Macros.h",
             "torch/headeronly/util/bit_cast.h", "torch/headeronly/util/BFloat16.h"):
    target = SCRATCH / path
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_text(read(path))
assert "return C10_WARP_SIZE * 4;" in constants
assert "return ::rsqrtf(x);" in math_compat
assert "const dim3 threads(warp_size, num_threads() / warp_size, 1)" in cuda
assert "using acc_t = acc_type<T, true>" in cuda
assert "return at::_fused_rms_norm" in dispatch or "_fused_rms_norm" in dispatch

# CUDA uses a 32-thread warp and acc_type<BFloat16, true> is float. Use the
# release's conversion type, arithmetic, shuffles and generic kernel unchanged.
golden = """#include <cuda_runtime.h>
#define C10_USING_CUSTOM_GENERATED_MACROS
#include <torch/headeronly/util/BFloat16.h>
#include <type_traits>
#include <cstdint>
#define C10_WARP_SIZE 32
template<class T, bool> using acc_type = float;
namespace c10::cuda::compat {
__device__ inline float rsqrt(float x){ return ::rsqrtf(x); }
}
"""
golden += between(device_utils, "template <typename T>\n__device__ __forceinline__ T WARP_SHFL(", "#if defined(USE_ROCM)\ntemplate<>")
golden += between(cuda, "constexpr int vec_size", "// Checks alignment")
golden += between(cuda, "struct WelfordDataLN{", "template<typename T, typename T_ACC, bool rms_norm>\n__device__ __inline__ void compute_gI")
golden += """
template __global__ void vectorized_layer_norm_kernel<c10::BFloat16,float,true>(
  int, float, const c10::BFloat16*, const c10::BFloat16*, const c10::BFloat16*,
  float*, float*, c10::BFloat16*);
"""
(SCRATCH / "golden.cu").write_text(golden)
jit = "python/sglang/kernels/jit"
header = Path(jit) / "csrc/muse_spark/ranked_pool_norm.cuh"
shutil.copyfile(STAGE / header, SCRATCH / header.name)
spec = (ROOT / jit / "utils/compile/spec.py").read_text()
env = {"Tuple": tuple}
exec(between(spec, "_FFI_INCLUDES =", "def resolve_sources"), env)
humming = STAGE / "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
module_def = next(n for n in ast.parse(humming.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_ranked_pool_module")
call = next(n for n in ast.walk(module_def) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "load_jit")
options = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
assert options["cuda_files"] == ["muse_spark/ranked_pool_norm.cuh"]
assert options["extra_cuda_cflags"] == ["--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
(SCRATCH / "donor.cu").write_text(env["_wrapper_source"]((header.name,), options["cuda_wrappers"]))
(SCRATCH / "pins.json").write_text(json.dumps({"revision": REV, "sha256": sources}, indent=2) + "\n")
