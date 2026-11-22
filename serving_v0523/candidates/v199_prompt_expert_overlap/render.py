"""Render Humming 0.1.12's actual FP4 and BF16 W2 kernels with full FP32 accumulation."""
from pathlib import Path
import sys
import json
here=Path(__file__).resolve().parent
root=Path(sys.argv[1])
sys.argv=[sys.argv[0],str(root/"humming"),str(root/"sglang"),str(root/"rendered")]
fixture=(here.parent/"v169_grouped_expert_tile_cohorts/vendor_codegen.py").read_text()
exec(compile(fixture[:fixture.index('sp=importlib')],str(here/"render.py"),'exec'))
stats={}
def save(self):
    name=f"{self.b_dtype.num_bits}_{self.block_shape[1]}_{self.num_stages}"
    path=target/(name+'.cu')
    path.write_text(self.code+"\nauto ptr = reinterpret_cast<void*>(&"+self.kernel_expr+");\n")
    stats[name]=dict(shared=self.estimated_smem_size,threads=self.num_threads,warp=self.warp_shape,block=self.block_shape,stages=self.num_stages)
OfflineRuntime.prepare=save
with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(multi_processor_count=132)):
    obj=LayerConfig(shape_n=4096,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,c_dtype=dtypes.bfloat16,weight_scale_group_size=16)
    cfg=scope["Sm90Heuristics"].get_config(obj,65408,gemm_type=GemmType.GROUPED_CONTIGUOUS)
    cfg.update(use_stream_k=False,use_tma_c=False)
    args=dict(shape_n=4096,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,c_dtype=dtypes.bfloat16,gemm_type=GemmType.GROUPED_CONTIGUOUS)
    HummingKernel(**(args|cfg),b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16)
    try:
        HummingKernel(**(args|cfg),b_dtype=dtypes.bfloat16)
    except AssertionError as exc:
        import traceback
        frame=traceback.extract_tb(exc.__traceback__)[-1]
        assert frame.name=="check_dtype" and "self.b_dtype.num_bits <= 8" in frame.line
        stats["native_bf16_refusal"]={"line":frame.lineno,"predicate":frame.line}
    else:
        raise AssertionError("unexpected published vendor BF16 B support")
    class DiagnosticBF16(HummingKernel):
        """Test native machine-template feasibility outside the published loader's dtype set."""
        def check_dtype(self):
            assert self.a_dtype==self.b_dtype==self.c_dtype==dtypes.bfloat16 and self.sm_version==90
    for n in (256,128):
        for stages in (3,4):
            c=cfg|dict(block_shape=(176,n,64),num_stages=stages)
            DiagnosticBF16(**(args|c),b_dtype=dtypes.bfloat16)
    import runpy
    from types import ModuleType
    jit=ModuleType("sglang.kernels.jit")
    jit.__file__=str(stage/"python/sglang/kernels/jit/__init__.py")
    pkg=ModuleType("sglang.kernels")
    pkg.jit=jit
    sys.modules[pkg.__name__]=pkg
    adapt=runpy.run_path(str(stage/"python/sglang/srt/layers/muse_spark/prompt_weight_pipeline.py"))["adapt_prompt_weights"]
    def pipeline(self):
        adapt(self)
        (target/"pipeline.cu").write_text(self.code+"\nauto ptr = reinterpret_cast<void*>(&"+self.kernel_expr+");\n")
    OfflineRuntime.prepare=pipeline
    HummingKernel(**(args|cfg|{"use_tma_bs":False,"use_cp_async":False}),
                  b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16)
print(json.dumps(stats,indent=2))
