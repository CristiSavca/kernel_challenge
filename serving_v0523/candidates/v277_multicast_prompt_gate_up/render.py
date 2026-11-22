"""Render the production gate/up constructor with the pinned Humming generator."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import json
import runpy
import sys

here = Path(__file__).resolve().parent
driver = here.parent / "v169_grouped_expert_tile_cohorts/vendor_codegen.py"
exec(compile(driver.read_text().split("sp=importlib.util.spec_from_file_location", 1)[0], str(driver), "exec"))
model = stage / "python/sglang/srt/layers/muse_spark"
jit = ModuleType("sglang.kernels.jit")
jit.__file__ = str(stage / "python/sglang/kernels/jit/__init__.py")
pkg = ModuleType("sglang.kernels")
pkg.jit = jit
sys.modules[pkg.__name__] = pkg
for name in ("residual_prompt_mma", "prompt_multicast"):
    mod = ModuleType("sglang.srt.layers.muse_spark." + name)
    mod.__file__ = str(model / (name + ".py"))
    exec(compile(Path(mod.__file__).read_text(), mod.__file__, "exec"), mod.__dict__)
    sys.modules[mod.__name__] = mod
compiler = ModuleType("humming.jit.compiler")
compiler.NVCCCompiler = type("Compiler", (), {})
sys.modules[compiler.__name__] = compiler
utils = ModuleType("sglang.kernels.jit.utils")
launcher = SimpleNamespace(configure=lambda *_: True, prompt_cluster_grid=lambda *_: 128)
utils.load_jit = lambda *a, **kw: launcher
sys.modules[utils.__name__] = utils
compile_kernel = runpy.run_path(str(model/"decode_gate_up.py"))["_compile"]
records = {}


def emit(kernel):
    name = type(kernel).__name__
    (target/(name+".cu")).write_text(kernel.code+"\nauto ptr = reinterpret_cast<void*>(&"+kernel.kernel_expr+");\n")
    records[name] = dict(block=kernel.block_shape, warp=kernel.warp_shape, multicast=kernel.multi_cast_size_a,
                         shared=kernel.estimated_smem_size, threads=kernel.num_threads, math_threads=kernel.num_math_threads)
    kernel.func = 99


OfflineRuntime.prepare = emit
OfflineRuntime.check_context = lambda self: None
with (patch.object(torch.cuda, "get_device_capability", return_value=(9,0)), patch.object(torch.cuda, "current_device", return_value=0),
      patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=132))):
    obj = LayerConfig(shape_n=8192, shape_k=4096, num_experts=32, a_dtype=dtypes.bfloat16, b_dtype=dtypes.float4e2m1,
                      bs_dtype=dtypes.float8e4m3, c_dtype=dtypes.bfloat16, weight_scale_group_size=16)
    config = scope["Sm90Heuristics"].get_config(obj, 65408, gemm_type=GemmType.GROUPED_CONTIGUOUS)
    config.update(use_stream_k=False, use_tma_c=False)
    args = (obj.to_str(), json.dumps(dict(gemm_type="grouped_contiguous", use_f16_accum=False)), json.dumps(config))
    plain, clustered = (compile_kernel(0,*args,True,mc) for mc in (False,True))
    assert plain is not None and len(plain)==4 and clustered is not None and clustered[-1]==128
    launcher.prompt_cluster_grid = lambda *_: 0
    assert compile_kernel(1,*args,True,True) is None and compile_kernel(1,*args,True) is not None
    assert compile_kernel(2,*args,False,True) is None
print(json.dumps(dict(variants=records, occupancy_fallback=True, decode_excluded=True), indent=2))
