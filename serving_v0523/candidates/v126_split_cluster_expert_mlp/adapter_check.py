"""Run the V68 clustered adapter and its preparation with simulated CUDA operators."""

from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS
import json
import math
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v98_graph_grouped_prefill"))
from entry_check import function

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v117_tiled_route_staging"))
from dispatch_check import Record


def tensor(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


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

    for mode in ("ordinary", "cc8", "cpu", "disabled", "grouped-off", "deterministic", "65experts",
                 "w13-mma", "w2-mma", "w13-weight-shape", "w2-weight-shape", "w13-scale-stride",
                 "w2-scale-dtype", "w2-fused-scale", "w2-apply-on-C", "cross-geometry", "w2-packed-k",
                 "jit-fallback", "zero-occupancy"):
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
        def probe(device,h,i,experts):
            calls.append(("probe",device,h,i,experts))
            return mode!="zero-occupancy"
        def extension():
            calls.append(("module",))
            return None if mode=="jit-fallback" else NS(supported=probe)
        st = NS(uint8=torch.uint8,int32=torch.int32,float8_e4m3fn=torch.float8_e4m3fn,bfloat16=torch.bfloat16,
                cuda=NS(get_device_capability=lambda d:(8,0) if mode=="cc8" else (9,0),get_device_properties=lambda d:NS(multi_processor_count=132)),nn=NS(Parameter=lambda t,**k:t))
        scope=dict(torch=st,json=json,envs=NS(SGLANG_ENABLE_MUSE_SPARK_HUMMING_GROUPED_DECODE=Flag(mode!="grouped-off"),
                   SGLANG_ENABLE_DETERMINISTIC_INFERENCE=Flag(mode=="deterministic"),SGLANG_ENABLE_MUSE_SPARK_HUMMING_CLUSTER_MLP=Flag(mode!="disabled")),
                   _GROUPED_COMPUTE="grouped",_MAX_DECODE_SLOTS=2048,_MAX_DECODE_EXPERTS=64,
                   HummingKernel=NS(prepare_kernels=lambda *args: None),
                   HummingMethod=NS(prepare_layer_meta=meta,get_default_tuning_configs=lambda *args,**kw:[(0,4096,{})]),
                   transform_humming_tensors=transform,_cluster_mlp_module=extension,
                   _split_mlp_module=lambda:None if mode=="jit-fallback" else NS(supported=lambda *args: 0 if mode=="zero-occupancy" else 6),
                   dtypes=NS(bfloat16=1,float4e2m1=2,float8e4m3=3),MmaType=NS(WGMMA="WGMMA"),
                   GemmType=NS(INDEXED=1,GROUPED_CONTIGUOUS=2),HummingInputSchema=lambda **kw:kw,HummingWeightSchema=lambda **kw:kw)
        function(path,"prepare_humming_nvfp4",scope)(layer)
        assert layer.humming_cluster_mlp == (mode=="ordinary"), mode
        assert layer.humming_split_mlps == (6 if mode=="ordinary" else 0)
        if mode in ("ordinary","zero-occupancy"): assert calls==[("module",),("probe",0,hidden,inter,e)]
        elif mode=="jit-fallback": assert calls==[("module",)]
        else: assert not calls
        assert layer.w13_weight.dtype==layer.w2_weight.dtype==torch.int32
        check.append(mode)
    return {"preparation_cases":len(check), "cases":check}



def dispatch(tree):
    path=tree/"python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
    final=tree/"python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py"
    counts={}
    # Row counts exercise native and irregular top-k, and density transitions.
    shapes=[(1,1,8),(8,8,256),(16,8,256),(32,8,256),(33,8,256),(128,8,256),(256,8,256),(65,3,256)]
    for (rows,topk,nexperts), enabled, mask, sms, mapped in product(
            shapes,[False,True],[0,2,4,6],[16,132],[False,True]):
        ops=[]
        rec=lambda tag: Record(tag,ops)
        triton=NS(next_power_of_2=lambda n:1<<(n-1).bit_length())
        scope=dict(torch=torch,math=math,triton=triton,_plan_grouped_decode=rec("plan"),_gather_grouped_decode=rec("gather"),
                   _split_mlp_module=lambda:rec("split"),_cluster_mlp_module=lambda:rec("cluster"),
                   _expert_scaled_silu_module=lambda:rec("silu"),HummingMethod=NS(forward_layer=rec("gemm").run),
                   _GROUPED_COMPUTE="grouped")
        function(path,"_sparse_cluster_grid",scope)
        impl=function(path,"humming_grouped_decode_outputs",scope)
        x=tensor(rows,4096)
        ids=tensor(rows,topk,dtype=torch.int32)
        layer=NS(global_num_experts=nexperts,num_local_experts=32,intermediate_size_per_partition=4096,
                 humming_split_mlps=mask,humming_cluster_mlp=enabled,humming_device_sms=sms,
                 _expert_map=tensor(nexperts,dtype=torch.int32) if mapped else None,
                 humming_decode_tunings={n:[(2048,n)] for n in ("w13","w2")},
                 w13_weight=tensor(1,dtype=torch.int32),w13_weight_scale=tensor(1,dtype=torch.uint8),
                 w13_global_scale=tensor(32),w2_weight=tensor(1,dtype=torch.int32),w2_weight_scale=tensor(1,dtype=torch.uint8))
        values,row_map,local_ids=impl(layer,x,ids)
        slots=ids.numel()
        # Use a direct occupancy expectation from probability mass per expert.
        probability=1-(1-min(topk/nexperts,1))**rows
        narrow=slots<=nexperts and 32*probability*8<2*sms
        splits=(4 if mask&4 and (32*probability*16<2*sms or not mask&2) else 2 if mask&2 else 0) if narrow else 0
        cluster=enabled and slots>=4*nexperts
        mode=f"P{splits}" if splits else "full" if cluster else "grouped"
        counts[mode]=counts.get(mode,0)+1
        tags=[a for a,b,c in ops]
        assert tags==["plan","gather"]+(["split"] if splits else ["cluster"] if cluster else ["gemm","silu","gemm"])
        plan,gather=ops[0][1],ops[1][1]
        assert row_map is plan[5] and local_ids is plan[2] and plan[0] is ids
        assert gather[2] is plan[3] and gather[1] is plan[4] and gather[0] is x
        if splits:
            args=ops[2][1]
            assert args[-2] is values and tuple(values.shape)==(splits,slots,4096) and values.dtype==torch.float32
            assert args[0] is gather[3] and args[-3] is plan[3] and args[1] is layer.w13_weight and args[3] is layer.w2_weight
            expected_grid=min(32,max(1,math.ceil(32*probability+math.sqrt(32*probability*(1-probability)))))
            assert args[-1]==expected_grid
            assert not torch._C._is_alias_of(gather[3],values)
        else:
            assert values.shape==(slots,4096) and values.dtype==torch.bfloat16
        output=tensor(rows,4096)
        run_final=function(final,"fused_post_expert_norm_weighted_sum",dict(
            torch=torch,triton=triton,_MIN_TOKENS_FOR_FUSED=32,_post_expert_norm_weighted_sum_kernel=rec("final")))
        assert run_final(values,tensor(rows,topk,dtype=torch.float32),tensor(4096),
                         slot_expert_ids=local_ids,slot_row_indices=row_map,expert_output_scales=tensor(32),out=output) is output
        name,args,kw=ops[-1]
        assert name=="final" and args[0] is values and args[3] is output
        assert kw["SPLITS"]==(splits or 1) and kw["PART_STRIDE"]==(slots*4096 if splits else 0)
        assert kw["HAS_ROW_MAP"] and kw["HAS_EXPERT_SCALE"]
    return dict(dispatch_cases=sum(counts.values()),paths=counts)

if __name__=="__main__":
    tree=Path(sys.argv[1])
    name="python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
    print(json.dumps(dict(**dispatch(tree),**startup(tree/name),physical_cuda=False),indent=2))
