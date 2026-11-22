"""Run the V68 clustered adapter and its preparation with simulated CUDA operators."""

from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v98_graph_grouped_prefill"))
from entry_check import function

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v117_tiled_route_staging"))
from dispatch_check import Record


def tensor(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def dispatch(path):
    used = []
    for (rows, topk, global_experts), enabled, mapped in product(
        [(1, 1, 16), (8, 8, 256), (32, 8, 256), (127, 8, 256), (128, 8, 256),
         (129, 8, 256), (256, 8, 256), (147, 7, 256), (256, 4, 128)], (False, True), (False, True)
    ):
        ops = []
        rec = lambda name: Record(name, ops)
        scope = dict(torch=torch, triton=NS(next_power_of_2=lambda n: 1 << (n-1).bit_length()),
                     _plan_grouped_decode=rec("plan"), _gather_grouped_decode=rec("gather"),
                     _cluster_mlp_module=lambda: rec("cluster"), _expert_scaled_silu_module=lambda: rec("silu"),
                     HummingMethod=NS(forward_layer=rec("gemm").run), _GROUPED_COMPUTE="grouped")
        f = function(path, "humming_grouped_decode_outputs", scope)
        ids = tensor(rows, topk, dtype=torch.int32)
        layer = NS(global_num_experts=global_experts, num_local_experts=32,
                   intermediate_size_per_partition=4096, humming_cluster_mlp=enabled,
                   _expert_map=tensor(global_experts, dtype=torch.int32) if mapped else None,
                   humming_decode_tunings={m: [(2048, m)] for m in ("w13", "w2")},
                   w13_weight=tensor(1, dtype=torch.int32), w13_weight_scale=tensor(1, dtype=torch.uint8),
                   w13_global_scale=tensor(32), w2_weight=tensor(1, dtype=torch.int32),
                   w2_weight_scale=tensor(1, dtype=torch.uint8))
        x = tensor(rows, 4096)
        out, backmap, local_ids = f(layer, x, ids)
        assert out.shape == (ids.numel(),4096) and backmap.shape == local_ids.shape == ids.shape
        tags = [name for name, args, kw in ops]
        plan, gather = ops[0][1], ops[1][1]
        assert plan[0] is ids and plan[1] is (layer._expert_map if mapped else ids)
        assert plan[2] is local_ids and plan[5] is backmap
        assert gather[0] is x and gather[1] is plan[4] and gather[2] is plan[3]
        clustered = enabled and ids.numel() >= 4 * global_experts
        if clustered:
            assert tags == ["plan", "gather", "cluster"]
            interface = ops[2][1]
            assert interface[0] is out is gather[3]
            assert interface[1] is layer.w13_weight and interface[3] is layer.w2_weight
            assert torch._C._is_alias_of(interface[2],layer.w13_weight_scale)
            assert torch._C._is_alias_of(interface[4],layer.w2_weight_scale)
            assert interface[5] is layer.w13_global_scale and interface[6] is plan[3]
            assert out.untyped_storage().nbytes() == 2 * ids.numel() * 4096
        else:
            assert tags == ["plan", "gather", "gemm", "silu", "gemm"]
            assert [ops[i][2]["sublayer_name"] for i in (2,4)] == ["w13","w2"]
            assert ops[2][2]["inputs"] is gather[3]
            assert ops[2][2]["outputs"] is ops[3][1][0]
            assert ops[3][1][1] is ops[4][2]["inputs"] and out is ops[4][2]["outputs"]
            assert ops[2][2]["expert_layout"] is ops[4][2]["expert_layout"] is plan[3]
        used.append(clustered)
    return {"adapter_cases": len(used), "clustered": sum(used), "parent": len(used)-sum(used)}


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
                cuda=NS(get_device_capability=lambda d:(8,0) if mode=="cc8" else (9,0)),nn=NS(Parameter=lambda t,**k:t))
        scope=dict(torch=st,json=json,envs=NS(SGLANG_ENABLE_MUSE_SPARK_HUMMING_GROUPED_DECODE=Flag(mode!="grouped-off"),
                   SGLANG_ENABLE_DETERMINISTIC_INFERENCE=Flag(mode=="deterministic"),SGLANG_ENABLE_MUSE_SPARK_HUMMING_CLUSTER_MLP=Flag(mode!="disabled")),
                   _GROUPED_COMPUTE="grouped",_MAX_DECODE_SLOTS=2048,_MAX_DECODE_EXPERTS=64,
                   HummingKernel=NS(prepare_kernels=lambda *args: None),
                   HummingMethod=NS(prepare_layer_meta=meta,get_default_tuning_configs=lambda *args,**kw:[(0,4096,{})]),
                   transform_humming_tensors=transform,_cluster_mlp_module=extension,
                   dtypes=NS(bfloat16=1,float4e2m1=2,float8e4m3=3),MmaType=NS(WGMMA="WGMMA"),
                   GemmType=NS(INDEXED=1,GROUPED_CONTIGUOUS=2),HummingInputSchema=lambda **kw:kw,HummingWeightSchema=lambda **kw:kw)
        function(path,"prepare_humming_nvfp4",scope)(layer)
        assert layer.humming_cluster_mlp == (mode=="ordinary"), mode
        if mode in ("ordinary","zero-occupancy"): assert calls==[("module",),("probe",0,hidden,inter,e)]
        elif mode=="jit-fallback": assert calls==[("module",)]
        else: assert not calls
        assert layer.w13_weight.dtype==layer.w2_weight.dtype==torch.int32
        check.append(mode)
    return {"preparation_cases":len(check), "cases":check}


if __name__ == "__main__":
    path = Path(sys.argv[1]) / "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
    print(json.dumps(dict(**dispatch(path), **startup(path), physical_cuda=False),indent=2))
