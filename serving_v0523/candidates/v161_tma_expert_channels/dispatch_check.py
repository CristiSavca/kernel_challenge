"""Test normal preparation, conditional descriptor creation and live sparse dispatch."""
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("route129",here.parent/"v129_routed_projection_pair/dispatch_check.py")
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
stage=Path(sys.argv[1]); path=stage/"python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
real=parent.function
startup=[]
for available,success in ((False,False),(True,False),(True,True)):
    def contextual(path,name,scope):
        base=scope["torch"]
        for key in ("SGLANG_ENABLE_MUSE_SPARK_COMPACT_PREFILL","SGLANG_ENABLE_MUSE_SPARK_MEDIUM_ROUTE_PAIR"):
            setattr(scope["envs"],key,NS(get=lambda:False))
        def prepare(g,gs,d,ds,maps):
            assert maps.shape==(512,) and maps.dtype==torch.uint8 and maps.device.type=="cpu"
            assert gs.dtype==ds.dtype==torch.uint8 and g.dtype==d.dtype==torch.int32
            return success
        mod=NS(prepare=prepare)
        base.empty=torch.empty
        scope["_tma_route_pair_module"]=lambda:mod if available else None
        original=real(path,name,scope)
        def call(layer):
            layer.hidden_size=4096;layer.intermediate_size_per_partition=4096
            with patch.object(type(layer.gate_up_proj),"data_ptr",lambda self:id(self),create=True):
                original(layer)
                keep=layer.humming_route_maps
                assert (keep is not None)==(available and success and layer.humming_route_pair)
                if keep is not None:
                    assert keep[0] is mod and len(keep[2])==4
            startup.append(keep is not None)
        return call
    with patch.object(parent,"function",contextual):
        parent.startup(path)

class Select(Exception):pass
def select(name):
    def run(*args,**kw):
        if name=="tma":
            assert args[-1] is cached and args[-2]>=1
        raise Select(name)
    return run

branches={}
for rows in (0,1,8,32,33,128,256,257):
    H, E, top=128,2,8
    x=torch.zeros(rows,H,dtype=torch.bfloat16);ids=torch.zeros(rows,top,dtype=torch.int32)
    for mode in ("normal","no-descriptors","address-w13","address-ds","new-shape","value-update","compiled","id64","wrong-width"):
        env=dict(torch=torch,_medium_route_pair_module=lambda:NS(run=select("medium")),
                 _route_pair_module=lambda:NS(run=select("vector")),moe_align_block_size=select("indexed"))
        model=NS(humming_route_pair=True,humming_medium_route_pair=True,num_local_experts=E,global_num_experts=256,
                 intermediate_size_per_partition=H,humming_indexed_tunings={k:[(10**6,{"block_shape":(8,64,128)})] for k in ("w13","w2")})
        banks=[]
        for proj,n in (("w13",2*H),("w2",H)):
            w=torch.empty(E,H//16,2*n,dtype=torch.int32)
            s=torch.empty(E,H//16,n,dtype=torch.float8_e4m3fn)
            banks.extend((w,s))
            setattr(model,proj+"_weight",w);setattr(model,proj+"_weight_scale",s)
            setattr(model,proj+"_global_scale",torch.empty(E,dtype=torch.bfloat16))
        cached=torch.zeros(512,dtype=torch.uint8)
        model.humming_route_maps=(NS(run=select("tma")),cached,tuple((b.data_ptr(),b.shape) for b in banks))
        if mode=="no-descriptors":model.humming_route_maps=None
        if mode=="address-w13":model.w13_weight=banks[0].clone()
        if mode=="address-ds":model.w2_weight_scale=banks[3].clone()
        if mode=="new-shape":model.w13_weight=banks[0].reshape(E,H//8,-1)
        if mode=="value-update":banks[0].fill_(37)
        if mode=="wrong-width":model.w2_weight_scale=banks[3][...,:-1]
        f=real(path,"humming_expert_outputs",env)
        with patch.object(torch.compiler,"is_compiling",return_value=mode=="compiled"):
            try:
                out,orig=f(model,x, ids.long() if mode=="id64" else ids)
                assert not rows and out.shape==(0,top,H); got="empty"
            except Select as it: got=it.args[0]
        expected=("empty" if not rows else "indexed" if mode in ("compiled","id64","wrong-width") or rows*top>2048
                  else "medium" if rows*top>256 else "tma" if mode in ("normal","value-update") else "vector")
        assert got==expected,(rows,mode,got)
        branches[got]=branches.get(got,0)+1
print(json.dumps(dict(preparation_cases=len(startup),successful_layouts=sum(startup),
                      dispatch_cases=sum(branches.values()),branches=branches,hardware=False),indent=2))
