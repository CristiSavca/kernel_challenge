"""Render actual native V68 grouped SM90 decode geometries and equal-K four-band variants."""
from hashlib import sha256
import json
from pathlib import Path
import sys

vendor, stage, target = map(Path,sys.argv[1:])
here=Path(__file__).resolve().parent
driver=here.parent/"v169_grouped_expert_tile_cohorts/vendor_codegen.py"
exec(compile(driver.read_text().split("sp=importlib.util.spec_from_file_location",1)[0],str(driver),'exec'))
result={}
def dump(k):
    name=f"{phase}_{width}_{kind}"
    src=k.code+"\nauto ptr = reinterpret_cast<void*>(&"+k.kernel_expr+");\n"
    (target/(name+'.cu')).write_text(src)
    result[name]=dict(source_sha256=sha256(src.encode()).hexdigest(),block=k.block_shape,warp=k.warp_shape,
        threads=k.num_threads,math_threads=k.num_math_threads,shared=k.estimated_smem_size,stages=k.num_stages)
OfflineRuntime.prepare=dump
with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(multi_processor_count=132)):
    for phase,n in [('w2',4096),('w13',8192)]:
        obj=LayerConfig(shape_n=n,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,
            b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,
            c_dtype=dtypes.bfloat16,weight_scale_group_size=16)
        for lo,hi,original in scope['Sm90Heuristics'].get_configs(obj,gemm_type=GemmType.GROUPED_CONTIGUOUS):
            if lo>=256: continue
            width,bn,bk=original['block_shape']
            for kind,factor in [('native',1),('wide',2),('wide3',2)]:
                cfg=original|dict(block_shape=(width,bn*factor,bk),warp_shape=(width,original['warp_shape'][1]*factor,64),
                    use_stream_k=False,use_tma_c=False,num_stages=3 if kind=='wide3' else original['num_stages'])
                try:
                    HummingKernel(shape_n=n,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,
                        bs_dtype=dtypes.float8e4m3,c_dtype=dtypes.bfloat16,weight_scale_group_size=16,gemm_type=GemmType.GROUPED_CONTIGUOUS,**cfg)
                except (AssertionError,ValueError) as exc:
                    result[f'{phase}_{width}_{kind}']=dict(rejected=str(exc),block=cfg['block_shape'],warp=cfg['warp_shape'])
print(json.dumps(result,indent=2))
