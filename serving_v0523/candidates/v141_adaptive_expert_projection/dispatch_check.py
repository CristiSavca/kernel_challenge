"""Exercise the production setup and shape dispatch with real CPU Torch metadata."""
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
pdir=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("old",pdir/"v129_routed_projection_pair/dispatch_check.py")
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
stage=Path(sys.argv[1])
path=stage/"python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
original=parent.function
preparations=[]
for enabled,ready in product((False,True),repeat=2):
    def augmented(p,name,scope):
        scope["envs"].SGLANG_ENABLE_MUSE_SPARK_COMPACT_PREFILL=NS(get=lambda:False)
        scope["envs"].SGLANG_ENABLE_MUSE_SPARK_MEDIUM_ROUTE_PAIR=NS(get=lambda:enabled)
        loads=[]
        def lookup():
            loads.append("medium")
            return object() if ready else None
        scope["_medium_route_pair_module"]=lookup
        native_loads=[]
        tiny_module=scope["_route_pair_module"]
        def tiny_lookup():
            native_loads.append("tiny")
            return tiny_module()
        scope["_route_pair_module"]=tiny_lookup
        f=original(p,name,scope)
        def invoke(layer):
            layer.hidden_size=4096;layer.intermediate_size_per_partition=4096;f(layer)
            assert bool(loads)==(bool(native_loads) and enabled)
            assert layer.humming_medium_route_pair==(bool(loads) and ready)
            preparations.append((layer.humming_route_pair,layer.humming_medium_route_pair))
        return invoke
    with patch.object(parent,"function",augmented):
        parent.startup(path)

class Stop(Exception):pass
def stop(*args,**kwargs):raise Stop("indexed")

cases={}
for t,top in [(0,8),(8,8),(32,8),(33,8),(128,8),(256,8),(257,8),(293,7),(2048,1)]:
    for mode in ("normal","medium-off","tiny-off","compiled","unaligned","strided-hidden","strided-ids","id64","width"):
        H=128
        x=torch.zeros((t,H),dtype=torch.bfloat16)
        if mode=="unaligned": x=torch.zeros(t*H+1,dtype=x.dtype)[1:].view(t,H)
        if mode=="strided-hidden":x=torch.zeros(t,2*H,dtype=x.dtype)[:,::2]
        ids=torch.zeros(t,top,dtype=torch.int32)
        if mode=="id64":ids=ids.long()
        if mode=="strided-ids":ids=torch.zeros(t,2*top,dtype=ids.dtype)[:,::2]
        def run(*args):
            assert args[0] is x and args[1] is ids
            assert args[8].shape==(ids.numel(),H) and args[9].shape==(ids.numel(),H)
            assert args[8].dtype==args[9].dtype==torch.bfloat16
            if len(args)==11:
                assert args[-1].dtype==torch.int32 and args[-1].shape==(32,t*top+1)
                raise Stop("medium")
            assert len(args)==12
            raise Stop("tiny")
        model=NS(humming_route_pair=mode!="tiny-off",humming_medium_route_pair=mode!="medium-off",
                 num_local_experts=32,global_num_experts=256,intermediate_size_per_partition=H,
                 humming_indexed_tunings={k:[(10**6,{"block_shape":(8,64,128)})] for k in ("w13","w2")})
        for proj,dim in (("w13",2*H),("w2",H)):
            setattr(model,proj+"_weight",torch.empty(32,H//16,dim*2,device="meta",dtype=torch.int32))
            setattr(model,proj+"_weight_scale",torch.empty(32,H//16,dim+(mode=="width"),device="meta",dtype=torch.float8_e4m3fn))
            setattr(model,proj+"_global_scale",torch.empty(32,device="meta",dtype=torch.bfloat16))
        f=original(path,"humming_expert_outputs",dict(torch=torch,moe_align_block_size=stop,
                   _route_pair_module=lambda:NS(run=run),_medium_route_pair_module=lambda:NS(run=run)))
        with patch.object(torch.compiler,"is_compiling",return_value=mode=="compiled"):
            try:
                out,orig=f(model,x,ids);assert not t and orig is ids and out.shape==(0,top,H);got="empty"
            except Stop as selected:got=selected.args[0]
        eligible=(mode not in ("compiled","width") and x.is_contiguous() and x.data_ptr()%4==0
                  and ids.dtype==torch.int32 and ids.is_contiguous())
        expected=("empty" if not t else "tiny" if eligible and t*top<=256 and mode!="tiny-off" else
                  "medium" if eligible and 256<t*top<=2048 and mode!="medium-off" else "indexed")
        assert expected==got,(t,top,mode,expected,got)
        cases[got]=cases.get(got,0)+1
print(json.dumps(dict(startup_cases=len(preparations),dispatch_cases=sum(cases.values()),branches=cases,
    independent_jit_states=sorted(set(preparations)),physical_gpu=False),indent=2))
