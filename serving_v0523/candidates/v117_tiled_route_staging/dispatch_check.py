"""Exercise the complete expert adapter with full-size PyTorch meta tensors."""
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"v98_graph_grouped_prefill"))
from entry_check import function


class Record:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def __getitem__(self, grid):
        return self.run

    def run(self, *args, **kw):
        self.calls.append((self.name, args, kw))


def main(tree):
    path=tree/"python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
    counts=Counter()
    cases=0
    # Explicit expected schedules for native and generic group sizes.
    shapes=((8,8,256,32,10),(16,8,256,32,16),(32,8,256,32,24),
            (64,8,256,32,0),(127,8,256,32,0),(128,8,256,32,0),
            (256,8,256,32,0),(153,7,256,17,0),(1,1,16,1,0),(7,4,128,64,16))
    def tensor(*shape,dtype=torch.bfloat16):
        return torch.empty(shape,device="meta",dtype=dtype)
    for (tokens,topk,glob,local,expected_groups), norm, skinny, fused, mapped in itertools.product(
            shapes,(True,False),(True,False),(True,False),(True,False)):
        h=4096
        x,ids=tensor(tokens,h),tensor(tokens,topk,dtype=torch.int64)
        layer=SimpleNamespace(intermediate_size_per_partition=h,num_local_experts=local,
            global_num_experts=glob,_expert_map=tensor(glob,dtype=torch.int32) if mapped else None,
            humming_skinny_gate_up=skinny,w13_weight=tensor(1,dtype=torch.int32),
            w13_weight_scale=tensor(1,dtype=torch.uint8),w13_global_scale=tensor(local),
            humming_decode_tunings={name:[(4096,name)] for name in ("w13","w2")})
        calls=[]
        env=dict(torch=torch,math=math,triton=SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length(),
            cdiv=lambda a,b:(a+b-1)//b),envs=SimpleNamespace(SGLANG_ENABLE_MUSE_SPARK_HUMMING_FUSED_STAGE=
            SimpleNamespace(get=lambda:fused)),HummingMethod=SimpleNamespace(forward_layer=Record("humming",calls).run),
            _GROUPED_COMPUTE="grouped",_skinny_gate_up_module=lambda:Record("skinny",calls),
            _expert_scaled_silu_module=lambda:Record("silu",calls))
        for name in ("_plan_route_tiles","_stage_token_routes","_stage_grouped_decode","_plan_grouped_decode","_gather_grouped_decode"):
            env[name]=Record(name,calls)
        function(path,"_active_expert_groups",env)
        adapter=function(path,"humming_grouped_decode_outputs",env)
        values,rows,local_ids=adapter(layer,x,ids,tensor(h) if norm else None,1e-5)
        assert values.shape==(tokens*topk,h) and rows.shape==local_ids.shape==ids.shape
        use_skinny=skinny and tokens*topk<=glob
        groups=expected_groups if norm and use_skinny else 0
        stage_name="_stage_token_routes" if norm else ("_stage_grouped_decode" if fused else "_plan_grouped_decode")
        uses_tiles=norm and not use_skinny and tokens*topk>=1024
        if uses_tiles:
            assert calls[0][0]=="_plan_route_tiles"
        _,args,kw=calls[int(uses_tiles)]
        assert calls[int(uses_tiles)][0]==stage_name
        assert sum(tag=="_plan_route_tiles" for tag,_,_ in calls)==int(uses_tiles)
        if norm:
            assert kw["ROUTE_TILE"] == (128 if uses_tiles else 0)
        if uses_tiles:
            _,planner,keywords=calls[0]
            assert args[1] is planner[4] and args[4] is planner[2] and args[7] is planner[3]
            assert args[1].shape==((tokens*topk+127)//128,local+1)
            assert keywords["TILE"]==128 and keywords["HAS_MAP"]==mapped
            assert planner[1] is (layer._expert_map if mapped else ids)
        elif norm:
            assert args[1] is (layer._expert_map if mapped else ids)
        producer=args[5 if norm else 4 if fused else 3]
        if norm:
            assert kw["WRITE_ACTIVE"]==bool(groups)
        assert producer.numel()==(local+1)*(2 if groups else 1)
        if use_skinny:
            _,args,_=next(c for c in calls if c[0]=="skinny")
            assert args[-3] is producer and args[-1]==groups
            assert args[-2]==max(1,tokens*topk*local//glob)
        assert [kw["sublayer_name"] for name,_,kw in calls if name=="humming"]==(["w2"] if use_skinny else ["w13","w2"])
        for name,_,kw in calls:
            if name=="humming":
                offsets=kw["expert_layout"]
                assert offsets.shape==(local+1,) and offsets.is_contiguous()
                assert offsets.storage_offset()==producer.storage_offset()==0
                assert torch._C._is_alias_of(offsets,producer)
        assert ("_gather_grouped_decode" in [c[0] for c in calls])==(not norm and not fused)
        counts[f"{stage_name} / {'tiled' if uses_tiles else 'worklist' if groups else 'ordinary layout'}"]+=1
        cases+=1
    print(json.dumps(dict(cases=cases,dispatch=counts,storage="PyTorch meta",real_gpu=False),indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]))
