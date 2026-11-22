"""Render V274's real constructors and test two/four-block occupancy and compiler fallbacks."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys

here=Path(__file__).resolve().parent
parent=here.parent/"v270_wide_prompt_weight_reuse"
fixture=parent/"candidate.py"
# Install the same real public Humming 0.1.12 generator and SGLang modules as the parent.
ns=dict(__name__="v274_render",__file__=str(fixture))
exec(compile(fixture.read_text().split("def emit(kernel):",1)[0],str(fixture),"exec"),ns)
runtime,prepare,torch=ns["OfflineRuntime"],ns["prepare"],ns["torch"]
images=Path(sys.argv[3])
kernels={}
meta={}
attempts=[]
denied=set()
def emit(kernel):
    name=type(kernel).__name__
    if name=="WidePromptWeightPipeline":
        name+=str(kernel.multi_cast_size_a)
    attempts.append(name)
    if name in denied:
        raise RuntimeError("compile failure")
    code=kernel.code+"\nauto ptr = reinterpret_cast<void*>(&"+kernel.kernel_expr+");\n"
    (images/(name+".cu")).write_text(code)
    kernels[name]=kernel
    kernel.kernel_id={"WidePromptWeightPipeline2":2,"WidePromptWeightPipeline4":4,"ResidualPromptWeightPipeline":9,"PromptWeightPipeline":11}[name]
    meta[name]=dict(multicast=kernel.multi_cast_size_a,warp=kernel.warp_shape,block=kernel.block_shape,threads=kernel.num_threads)
runtime.prepare=emit
runtime.check_context=lambda self:None
records=[]
with patch.object(torch.cuda,"get_device_capability",return_value=(9,0)):
    for i,(sms,n,pair,quad,fail,want) in enumerate((
        (132,4096,128,128,(),4),(132,4096,128,132,(),4),(132,4096,128,124,(),2),
        (132,4096,126,128,(),2),(130,4096,128,128,(),4),(128,4096,128,0,(),2),
        (132,4096,0,128,(),9),(131,4096,128,128,(),9),
        (132,4096,128,128,("WidePromptWeightPipeline4",),2),
        (132,4096,128,128,("WidePromptWeightPipeline2",),9),
        (132,4096,128,128,("ResidualPromptWeightPipeline",),4),
        (132,4096,128,128,("ResidualPromptWeightPipeline","PromptWeightPipeline"),None),
        (132,3840,128,128,(),2)
    )):
        props=SimpleNamespace(multi_processor_count=sms)
        with patch.object(torch.cuda,"get_device_properties",return_value=props):
            dtype=ns["dtypes"]
            obj=ns["LayerConfig"](shape_n=n,shape_k=4096,num_experts=32,a_dtype=dtype.bfloat16,b_dtype=dtype.float4e2m1,
                                  bs_dtype=dtype.float8e4m3,c_dtype=dtype.bfloat16,weight_scale_group_size=16)
            tuned=ns["scope"]["Sm90Heuristics"].get_config(obj,65408,gemm_type=ns["GemmType"].GROUPED_CONTIGUOUS)
            tuned.update(use_stream_k=False,use_tma_c=False)
            tracks=[]
            def occupancy(kernel,_sms):
                assert _sms==sms
                size=kernel.multi_cast_size_a
                tracks.append(size)
                val=pair if size==2 else quad
                if not val: raise ValueError("no resident cluster")
                return val
            denied.clear()
            denied.update(fail)
            attempts.clear()
            with patch.dict(prepare.__wrapped__.__globals__,_cluster_grid=occupancy):
                table=prepare(i,obj.to_str(),json.dumps(dict(gemm_type="grouped_contiguous",use_f16_accum=False)),json.dumps(tuned))
            selected=None if table is None else next(r for r in table.reshape(-1,4).tolist() if r[0]<16384<=r[1])
            assert (None if selected is None else selected[2])==want,(i,attempts,selected)
            if want in (2,4):
                assert selected[3]==pair
            if 4 in tracks:
                assert tracks==[2,4] and pair and pair%4==0
            records.append(dict(sms=sms,channels=n,pair_capacity=pair,quad_capacity=quad,failed_compiles=fail,
                                occupancy_queries=tracks,selected=selected,constructors=list(attempts)))
            if i==0:
                actual_meta=meta.copy()
                primary={x:(images/(x+".cu")).read_bytes() for x in kernels}
for name,source in primary.items():
    (images/(name+".cu")).write_bytes(source)
print(json.dumps(dict(production_geometry=actual_meta,cases=records,device=False),indent=2))

