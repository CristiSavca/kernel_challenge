"""Render the actual new scheduler and check exact profile decisions including occupancy failures."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

here=Path(__file__).resolve().parent
fixture=here.parent/"v274_four_block_prompt_multicast/render.py"
text=fixture.read_text().split("records=[]",1)[0].replace('here.parent/"v270_wide_prompt_weight_reuse"', 'here.parent/"v270_wide_prompt_weight_reuse"')
text=text.replace('"WidePromptWeightPipeline4":4,','"WidePromptWeightPipeline4":4,"OrderedPromptWeightPipeline":8,')
exec(compile(text,str(fixture),"exec"))
stage=Path(sys.argv[2])
module=ModuleType("sglang.srt.layers.muse_spark.prompt_width_order")
module.__file__=str(stage/"python/sglang/srt/layers/muse_spark/prompt_width_order.py")
exec(compile(Path(module.__file__).read_bytes(),module.__file__,"exec"),module.__dict__)
sys.modules[module.__name__]=module
cases=[]
with patch.object(torch.cuda,"get_device_capability",return_value=(9,0)):
    for i,(pair,quad,order,fail,expect) in enumerate(((132,128,128,(),8),(132,132,128,(),4),(132,124,124,(),2),
            (132,128,124,(),2),(132,128,0,(),2),(132,128,128,("OrderedPromptWeightPipeline",),2),
            (130,132,128,(),8),(128,124,124,(),8),(128,120,120,(),2),(132,0,0,(),2),(0,128,128,(),9))):
        with patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(multi_processor_count=132)):
            dtype=ns["dtypes"]
            obj=ns["LayerConfig"](shape_n=4096,shape_k=4096,num_experts=32,a_dtype=dtype.bfloat16,b_dtype=dtype.float4e2m1,bs_dtype=dtype.float8e4m3,c_dtype=dtype.bfloat16,weight_scale_group_size=16)
            tuned=ns["scope"]["Sm90Heuristics"].get_config(obj,65408,gemm_type=ns["GemmType"].GROUPED_CONTIGUOUS)
            tuned.update(use_stream_k=False,use_tma_c=False)
            probes=[]
            def capacity(kernel,sms):
                typ=type(kernel).__name__
                value=order if typ=="OrderedPromptWeightPipeline" else quad if kernel.multi_cast_size_a==4 else pair
                probes.append((typ,kernel.multi_cast_size_a,value))
                if not value: raise ValueError("no device capacity")
                return value
            attempts.clear(); denied.clear(); denied.update(fail)
            with patch.dict(prepare.__wrapped__.__globals__,_cluster_grid=capacity):
                tab=prepare(i,obj.to_str(),json.dumps(dict(gemm_type="grouped_contiguous",use_f16_accum=False)),json.dumps(tuned))
            table=tab.reshape(-1,4).tolist()
            def select(rows):
                return next((k,g) for lo,hi,k,g in table if lo<rows<=hi)
            assert select(16384)[0]==expect,(i,table,attempts)
            if expect==8:
                grid=min(order,pair-pair%4);limit=256*(pair*grid//(32*(pair-grid))-31)
                assert select(limit)==(8,grid) and select(limit+1)==(2,pair)
                assert select(8192)==(8,grid) and select(4096)==(9,0)
            cases.append(dict(pair=pair,quad=quad,ordered=order,failed=list(fail),selected=select(16384),probes=probes))
print(json.dumps(dict(production_geometry=meta,cases=cases,patch_sha256=__import__('hashlib').sha256((stage.parent/"patch.diff").read_bytes()).hexdigest(),GPU=False),indent=2))
