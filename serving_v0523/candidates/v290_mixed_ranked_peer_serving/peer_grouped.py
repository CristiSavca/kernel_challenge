"""Exercise V290's peer eligibility vote and stable sorted grouped entry across real Gloo ranks."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/"v286_alternating_peer_prompt_inputs"))
import transport

PEER=transport.SOURCE
HUM=PEER.replace("peer_prompt_pack.py","muse_spark_humming_moe.py")
MODEL="python/sglang/srt/models/muse_spark.py"

def definitions(stage, path, names, scope):
    tree=ast.parse((stage/path).read_bytes())
    body=[n for n in tree.body if getattr(n,"name",None) in names]
    assert len(body)==len(names)
    exec("from __future__ import annotations\n"+ast.unparse(ast.Module(body,type_ignores=[])),scope)

def module(name,**kwargs):
    m=ModuleType(name);vars(m).update(kwargs);return m

def grouped(stage,binding,workspace,mode,rank,world,state,group,data,rows,width):
    class Map:
        def __getitem__(self,grid):
            def invert(shard,slots,inputs,mapped,first,live,n,*args,**kwargs):
                assert live==0 and n==mapped.numel()
                mapped.fill_(-1)
                mapped.view(-1)[slots[first:]]=torch.arange(n-first,dtype=mapped.dtype)
            return invert
    def gather(value,n):
        all_rows=[torch.empty_like(value) for _ in range(world)]
        dist.all_gather(all_rows,value)
        return torch.cat(all_rows)[:n]
    gates=dict(torch=torch,_GROUPED_COMPUTE={},_pack_prompt_routes=Map(),
               triton=NS(cdiv=lambda a,b:(a+b-1)//b,next_power_of_2=lambda n:1<<(n-1).bit_length()))
    definitions(stage,HUM,{"prompt_route_order","local_humming_expert_ids","humming_grouped_expert_outputs"},gates)
    obs=dict(grouped=0,fallback=0,empty=0,words=0,stubs=0)
    class Experts:
        num_local_experts=2
        intermediate_size_per_partition=width
        humming_prompt_gate_up=True
        humming_prompt_down=None
        humming_grouped_tunings={"w13":{},"w2":{}}
        _expert_map=object()
        post_expert_norm=object()
        def _grouped_humming_and_combine_impl(self,shard,local_ids,scores,out,peer_plan):
            actual,maps=gates["humming_grouped_expert_outputs"](self,shard,local_ids,defer_output_scale=True,peer_plan=peer_plan)
            golden=state.gold
            positions=torch.where((local_ids>=0).flatten())[0]
            positions=positions[torch.argsort(local_ids.flatten()[positions],stable=True)]
            ref=golden[positions//local_ids.shape[1]]
            assert torch.equal(actual.view(torch.int16),ref)
            if actual.numel():
                target=torch.full_like(local_ids,-1).flatten()
                target[positions]=torch.arange(positions.numel(),dtype=target.dtype)
                assert torch.equal(maps.flatten(),target)
            obs["grouped"]+=1;obs["empty"]+=int(not actual.numel());obs["words"]+=actual.numel()
            out.zero_()
        def __call__(self,inputs,scores,ids,norm):
            assert norm is self.post_expert_norm and torch.equal(inputs.view(torch.int16),state.gold)
            assert scores is state.scores and ids is state.ids
            obs["fallback"]+=1;obs["words"]+=inputs.numel()
            return torch.zeros_like(inputs)
    method=NS(forward_layer=lambda layer,*,inputs,outputs,sublayer_name,**kw:outputs.copy_(inputs) if sublayer_name=="w2" else None)
    modules={m.__name__:m for m in (
        module("sglang.srt.layers.muse_spark.muse_spark_humming_moe",**{k:gates[k] for k in ("prompt_route_order","local_humming_expert_ids")}),
        module("sglang.srt.layers.moe.token_dispatcher.standard",_map_topk_ids_to_local_experts=lambda ids,unused:torch.where(ids//2==rank,ids%2,-1)),
        module("humming.layer",HummingMethod=method),
        module("sglang.kernels.ops.activation.activation",silu_and_mul=None),
        module("sglang.srt.layers.muse_spark.decode_gate_up",run_gate_up=lambda layer,inp,*a,**kw:inp.clone()),
    )}
    group.all_gather_into_tensor=lambda dest,src:dist.all_gather_into_tensor(dest,src)
    ctx=dict(torch=torch,layernorm_sp=NS(sp_exit_gather=gather),
             _local_grouped_eligible=lambda experts,shard:rank!=state.reject)
    definitions(stage,PEER,{"_peer_experts_into","_capture_stub"},ctx)
    experts=Experts()
    banks=len(workspace.banks);pub=0;before=state.barriers
    with patch.dict(sys.modules,modules):
        for epoch in range(15):
            state.epoch=epoch; n=(rows,1,7,5)[epoch%4]
            shard=data(epoch,rank,n).view(torch.bfloat16)
            state.gold=torch.cat([data(epoch,r,n) for r in range(world)])
            ids=(torch.arange(n*world*2,dtype=torch.int32).view(n*world,2)*3+epoch)%(world*2)
            if epoch in (1,8):ids.fill_(0)
            if epoch==10:ids.fill_(-1)
            state.ids=ids;state.scores=torch.linspace(.01,1,ids.numel()).view_as(ids)
            state.reject=(epoch+1)%world if epoch in (2,5,11) else -1
            out=torch.full_like(state.gold.view(torch.bfloat16),1)
            if epoch in (3,9):
                ctx["_capture_stub"](workspace,experts,shard,state.scores,ids,group,out);obs["stubs"]+=1
            else:
                ctx["_peer_experts_into"](workspace,experts,shard,state.scores,ids,group,out)
                pub+=int(state.reject<0)
            assert not out.view(torch.int16).any()
            assert workspace.next_bank==pub%banks
    assert state.barriers-before==pub*(1 if banks==2 else 2)
    return dict(mode=mode,banks=banks,publications=pub,barriers=state.barriers-before,**obs)

def worker(rank,world,stage,folder):
    original=transport.exercise
    transport.exercise=lambda *args:original(*args,grouped_check=grouped)
    transport.run_world(rank,world,stage,folder)

def main(stage):
    totals={}
    with tempfile.TemporaryDirectory(prefix="v290-peer-grouped-") as root:
        for world in (4,8):
            mp.spawn(worker,args=(world,str(stage),root),nprocs=world)
            for rank in range(world):
                for row in json.loads((Path(root)/f"{world}_{rank}.json").read_bytes()):
                    target=totals.setdefault(row["mode"],{})
                    for key in ("grouped","fallback","empty","stubs","publications","barriers","words","declined"):
                        target[key]=target.get(key,0)+int(row.get(key,0))
    for key in ("ordinary","large_allocation","large_registration","large_probe"):
        assert totals[key]["grouped"]==120 and totals[key]["fallback"]==36 and totals[key]["empty"]>0
    print(json.dumps(dict(tree="a024fdaa7a134625083a6557fd5a6e05a0894e44",torch=torch.__version__,worlds=[4,8],
        sources={p:hashlib.sha256((stage/p).read_bytes()).hexdigest() for p in (PEER,HUM,MODEL)},
        counts=totals,words_compared=sum(x["words"] for x in totals.values()),mismatches=0,
        scope="Actual V290 peer startup, _peer_experts_into, capture stub, prompt_route_order and Humming grouped input dispatch. Gloo collectives and rank-local hook-equivalent veto. Host shims for GPU eligibility, Triton inversion and peer loads, Humming projections and stream dispatch. BF16 words checked on host using the receiver's independent sorted-route and full-gather references."),indent=2))
if __name__=="__main__":main(Path(sys.argv[1]).resolve())
