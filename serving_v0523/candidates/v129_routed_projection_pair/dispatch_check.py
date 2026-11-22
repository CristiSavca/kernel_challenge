"""Exercise V129's production preparation, indexed dispatch, and ordinary V64 fallback."""
from pathlib import Path
from itertools import product
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import json
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v98_graph_grouped_prefill"))
from entry_check import function
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v117_tiled_route_staging"))
from dispatch_check import Record


def dispatch(path):
    results = []
    for (m, topk), (enabled, mode) in product(
        [(0,8),(1,1),(8,8),(17,7),(32,8),(33,8),(128,8),(256,8)],
        [(True,"normal"), (False,"normal"), (True,"compiled"), (True,"odd"),
         (True,"unaligned-row"), (True,"strided-ids"), (True,"nonint-ids")],
    ):
        width, inner, experts, total = 128, 256, 32, 256
        ops = []
        recorder = lambda name: Record(name, ops)
        h = torch.zeros(m,width,dtype=torch.bfloat16)
        if mode=="odd":
            h=torch.empty(m*width+1,dtype=h.dtype)[1:].view(m,width)
        if mode=="unaligned-row":
            h=torch.empty(m,width*2,dtype=h.dtype)[:,::2]
        ids = torch.zeros(m,topk,dtype=torch.int32)
        if mode=="strided-ids":
            ids=torch.zeros(m,topk*2,dtype=ids.dtype)[:,::2]
        if mode=="nonint-ids":
            ids=ids.long()
        tunings = {k:[(10**6,{"block_shape":(8,64,128)})] for k in ("w13","w2")}
        layer = NS(humming_route_pair=enabled, num_local_experts=experts,global_num_experts=total,
                   intermediate_size_per_partition=inner,humming_indexed_tunings=tunings)
        for name,n,k in [("w13",2*inner,width),("w2",width,inner)]:
            setattr(layer,name+"_weight",torch.empty(experts,k//16,2*n,dtype=torch.int32,device="meta"))
            setattr(layer,name+"_weight_scale",torch.empty(experts,k//16,n,dtype=torch.float8_e4m3fn,device="meta"))
            setattr(layer,name+"_global_scale",torch.empty(experts,device="meta"))
        def align(*args,**kwargs):
            ops.append(("align",args,kwargs))
            return ids.flatten(),ids.flatten(),torch.empty(1,dtype=torch.int32)
        scope=dict(torch=torch,triton=NS(next_power_of_2=lambda x:1<<(x-1).bit_length()),
                   HummingMethod=NS(forward_layer=recorder("humming").run), moe_align_block_size=align,
                   silu_and_mul=recorder("silu").run, _scale_live_routes=recorder("scale"),
                   _route_pair_module=lambda:recorder("pair"))
        f=function(path,"humming_expert_outputs",scope)
        with patch.object(torch.compiler,"is_compiling",return_value=mode=="compiled"):
            out,original=f(layer,h,ids)
        assert out.shape==(m,topk,width) and original is ids
        qualifies=(enabled and 0<m*topk<=total and h.is_contiguous() and h.data_ptr()%4==0 and
                   ids.is_contiguous() and ids.dtype==torch.int32 and mode!="compiled")
        names=[k for k,_,_ in ops]
        assert names==([] if not m else ["pair"] if qualifies else ["align","humming","scale","silu","humming","scale"]), (m,mode,names)
        if qualifies:
            _,args,_=ops[0]
            assert args[0] is h and args[1] is ids
            assert args[8].shape==(ids.numel(),inner) and torch._C._is_alias_of(out,args[9])
            assert args[10].shape==(1+9*(min(experts,ids.numel())+max(0,ids.numel()-experts)//8),)
            assert args[11]==max(1,min(experts,ids.numel(),(ids.numel()*experts+total-1)//total))
        results.append((m,mode,qualifies))
    return dict(cases=len(results), native=sum(v[2] for v in results),
                empty=sum(v[0]==0 for v in results),fallback=sum(v[0]>0 and not v[2] for v in results))


def startup(path):
    check = []
    class Flag:
        def __init__(self, value):
            self.get = lambda: value
    class Packed:
        def __init__(self, shape, dtype, contig=True):
            self.shape, self.dtype, self.contig = shape, dtype, contig
            self.device, self.is_cuda, self.ndim = "cuda:0", True, len(shape)
        def get_device(self): return 0
        def is_contiguous(self): return self.contig
        def view(self, dtype): return Packed(self.shape, dtype, self.contig)
        def __gt__(self, n): return self
        def __lt__(self, n): return self
        def any(self): return self
        def item(self): return False
        def masked_fill(self, *_): return self
        def to(self, dtype): return self.view(dtype)
        def contiguous(self): return self

    for mode in ("ordinary", "cc8", "cpu", "disabled", "runtime-strict", "deterministic", "65experts",
                 "w13-mma", "w2-mma", "w13-weight-shape", "w2-weight-shape", "w13-scale-stride",
                 "w2-scale-dtype", "w2-fused-scale", "w2-apply-on-C", "cross-geometry", "w2-packed-k",
                 "jit-fallback"):
        e = 65 if mode == "65experts" else 32
        hidden, inter = 4096, 4096
        g=NS(group_size=16)
        layer = NS(num_local_experts=e, global_num_experts=256, quant_config=g)
        for name, n, k in (("gate_up_proj", 2*inter, hidden), ("down_proj", hidden, inter)):
            if mode == "cross-geometry" and name == "down_proj": k += 512
            setattr(layer, name, Packed((e,n,k//2),torch.uint8))
            setattr(layer, name + "_weight_scale", Packed((e,n,k//16),torch.float8_e4m3fn))
            setattr(layer, name + "_weight_scale_2", Packed((e,),torch.bfloat16))
        layer.gate_up_proj.is_cuda = mode != "cpu"
        calls=[]
        def meta(layer, shape_n, shape_k, sublayer_name, **kw):
            key=sublayer_name
            return NS(n=shape_n,k=shape_k,name=key,weight_name=key+"_weight",weight_scale_name=key+"_weight_scale",
                      mma_type="other" if mode==key+"-mma" else "WGMMA",use_packed_k_layout=mode==key+"-packed-k",
                      use_fused_e8m0_scale=mode==key+"-fused-scale",should_apply_bs_on_c=mode==key+"-apply-on-C",to_str=lambda: key)
        def transform(meta, _, **kw):
            n,k,tag=meta.n,meta.k,meta.name
            return dict(weight=Packed((e,k//16,2*n+int(mode==tag+"-weight-shape")),torch.int32),
                        weight_scale=Packed((e,k//16,n), torch.uint8 if mode==tag+"-scale-dtype" else torch.float8_e4m3fn,
                                            mode!=tag+"-scale-stride"))
        def extension():
            calls.append(("module",))
            return None if mode=="jit-fallback" else object()
        st = NS(uint8=torch.uint8,int32=torch.int32,float8_e4m3fn=torch.float8_e4m3fn,bfloat16=torch.bfloat16,
                cuda=NS(get_device_capability=lambda d:(8,0) if mode=="cc8" else (9,0)),nn=NS(Parameter=lambda t,**k:t))
        scope=dict(torch=st,json=json,envs=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=Flag(mode=="deterministic"),
                   SGLANG_ENABLE_MUSE_SPARK_ROUTED_PROJECTION_PAIR=Flag(mode!="disabled")),
                   get_exec=lambda:NS(deterministic=NS(enable_deterministic_inference=mode=="runtime-strict")),
                   _GROUPED_COMPUTE="grouped",
                   HummingKernel=NS(prepare_kernels=lambda *args: None),
                   HummingMethod=NS(prepare_layer_meta=meta,get_default_tuning_configs=lambda *args,**kw:[(0,4096,{})]),
                   transform_humming_tensors=transform,_route_pair_module=extension,
                   dtypes=NS(bfloat16=1,float4e2m1=2,float8e4m3=3),MmaType=NS(WGMMA="WGMMA"),
                   GemmType=NS(INDEXED=1,GROUPED_CONTIGUOUS=2),HummingInputSchema=lambda **kw:kw,HummingWeightSchema=lambda **kw:kw)
        function(path,"prepare_humming_nvfp4",scope)(layer)
        assert layer.humming_route_pair == (mode=="ordinary"), mode
        assert calls == ([("module",)] if mode in ("ordinary","jit-fallback") else []), mode
        assert layer.w13_weight.dtype==layer.w2_weight.dtype==torch.int32
        check.append(mode)
    return {"preparation_cases":len(check), "cases":check}


if __name__ == "__main__":
    path = Path(sys.argv[1]) / "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
    print(json.dumps(dict(dispatch=dispatch(path),startup=startup(path),gpu=False),indent=2))
