"""Exercise the production constructor, success and fallback with the pinned Hopper generator."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import json
import runpy
import sys

here = Path(__file__).resolve().parent
vendor, stage, target = map(Path, sys.argv[1:])
driver = here.parent / "v169_grouped_expert_tile_cohorts/vendor_codegen.py"
exec(compile(driver.read_text().split("sp=importlib.util.spec_from_file_location", 1)[0], str(driver), "exec"))
model = stage / "python/sglang/srt/layers/muse_spark"
jit = ModuleType("sglang.kernels.jit")
jit.__file__ = str(stage / "python/sglang/kernels/jit/__init__.py")
pkg = ModuleType("sglang.kernels")
pkg.jit = jit
sys.modules[pkg.__name__] = pkg
res = ModuleType("sglang.srt.layers.muse_spark.residual_prompt_mma")
res.__file__ = str(model / "residual_prompt_mma.py")
exec(compile(Path(res.__file__).read_text(), res.__file__, "exec"), res.__dict__)
sys.modules[res.__name__] = res
compiler = ModuleType("humming.jit.compiler")
compiler.NVCCCompiler = type("Compiler", (), {})
sys.modules[compiler.__name__] = compiler
prepare = runpy.run_path(str(model / "prompt_weight_pipeline.py"))["_prepare"]
attempts, configs, denied, kernels = [], {}, set(), {}


def emit(kernel):
    name = type(kernel).__name__
    attempts.append(name)
    if name in denied:
        raise RuntimeError("unavailable image")
    source = kernel.code + "\nauto ptr = reinterpret_cast<void*>(&" + kernel.kernel_expr + ");\n"
    (target / (name + ".cu")).write_text(source)
    kernels[name] = kernel
    configs[name] = dict(block=kernel.block_shape, warp=kernel.warp_shape, multicast=kernel.multi_cast_size_a,
                         threads=kernel.num_threads, math_threads=kernel.num_math_threads,
                         native_estimated_shared=kernel.estimated_smem_size)
    kernel.kernel_id = {"WidePromptWeightPipeline":7, "ResidualPromptWeightPipeline":9, "PromptWeightPipeline":11}[name]


OfflineRuntime.prepare = emit
OfflineRuntime.check_context = lambda self: None
with patch.object(torch.cuda, "get_device_capability", return_value=(9,0)), patch.object(torch.cuda, "current_device", return_value=0):
    with patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=132)):
        obj = LayerConfig(shape_n=4096, shape_k=4096, num_experts=32, a_dtype=dtypes.bfloat16, b_dtype=dtypes.float4e2m1,
                          bs_dtype=dtypes.float8e4m3, c_dtype=dtypes.bfloat16, weight_scale_group_size=16)
        tuning = scope["Sm90Heuristics"].get_config(obj, 65408, gemm_type=GemmType.GROUPED_CONTIGUOUS)
        tuning.update(use_stream_k=False, use_tma_c=False)
        compute = json.dumps(dict(gemm_type="grouped_contiguous", use_f16_accum=False))
        results = []
        residual = "ResidualPromptWeightPipeline"
        wide = "WidePromptWeightPipeline"
        plain = "PromptWeightPipeline"
        ranges = None
        for context, failures in enumerate(([],[wide],[residual],[wide,residual],[wide,residual,plain])):
            attempts.clear()
            denied.clear()
            denied.update(failures)
            with patch.dict(prepare.__wrapped__.__globals__, _cluster_grid=lambda kernel,sms:128):
                result = prepare(context,obj.to_str(),compute,json.dumps(tuning))
            table = result.tolist() if result is not None else None
            results.append(dict(attempts=list(attempts), denied=failures, config=table))
            if context == 0:
                ranges = result.reshape(-1,4).tolist()
                for n,expected in ((1,9),(8192,7),(10240,9),(12288,7),(14336,7),(16384,7)):
                    rows = [r for r in ranges if r[0] < n <= r[1]]
                    assert len(rows)==1 and rows[0][2]==expected, (n,rows)
                    assert rows[0][3] == (128 if expected==7 else 0)
                assert ranges[0][0]==0 and ranges[-1][1]==1<<30
                assert all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
        assert [r["config"] is not None for r in results] == [True,True,True,True,False]
        assert list(configs)==[wide,residual,plain]
        attempts.clear()
        denied.clear()
        with patch.dict(prepare.__wrapped__.__globals__, _cluster_grid=lambda *a:(_ for _ in ()).throw(RuntimeError("no occupancy"))):
            assert prepare(8,obj.to_str(),compute,json.dumps(tuning)).tolist()==[0,1<<30,9,0]
            assert attempts==[wide,residual]
    denied.clear()
    with patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=131)):
        attempts.clear()
        assert prepare(10,obj.to_str(),compute,json.dumps(tuning)) is not None
        assert attempts == ["ResidualPromptWeightPipeline"]
print(json.dumps(dict(variants=configs, startup=results, odd_SM_fallback=attempts, local_row_ranges=ranges), indent=2))

