"""Drive the pinned eager-on-graph closure with changing owner shards and vetoes."""
import ast
from contextvars import ContextVar
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType,SimpleNamespace as NS
from unittest.mock import patch

import torch

HERE=Path(__file__).resolve().parent
candidate = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
manifest=json.loads((candidate/"validation.json").read_text())
stage=Path(sys.argv[1])
member="python/sglang/srt/layers/muse_spark/peer_prompt_pack.py"
graph_path="python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py"
def define(path,names,env):
    data=subprocess.check_output(["git","-C",str(stage),"show",manifest["tree"]+":"+path])
    nodes=[n for n in ast.parse(data).body if getattr(n,"name",None) in names]
    assert len(nodes)==len(names)
    exec("from __future__ import annotations\n"+ast.unparse(ast.Module(nodes,[])),env)
ctx=ContextVar("capture",default=None)
rt=dict(torch=torch,functools=functools,_current_capture_var=ctx)
define(graph_path,["_weak_ref_if_tensor","_copy_output","_EagerOnGraphCallable","eager_on_graph"],rt)
pkg=ModuleType("Humming")
pkg.local_humming_expert_ids=lambda exp,x:x.int().contiguous()
def order(ids,n):
    vals,slots=ids.flatten().sort(stable=True)
    return vals,slots,torch.searchsorted(vals,torch.arange(n+1,dtype=vals.dtype),out_int32=True)
pkg.prompt_route_order=order
env=dict(torch=torch,_local_grouped_eligible=lambda experts,shard:state["eligible"])
define(member,["_gather_prompt_routes","_peer_experts_into","_capture_stub"],env)
torch.set_num_threads(1)
stats={"live_replays":0,"veto_replays":0,"fp32_score_bits":0,"int32_ids":0,"collectives_in_capture":0}
rng=torch.Generator().manual_seed(242)
with patch.dict(sys.modules,{"sglang.srt.layers.muse_spark.muse_spark_humming_moe":pkg}):
    for size in (2,4,8):
        rows,k,h,rank=31,8,16,size-1
        tokens=rows*size-1
        shard=torch.zeros(rows,h,dtype=torch.bfloat16)
        scores=torch.zeros(rows,k);ids=torch.zeros(rows,k,dtype=torch.int32)
        dest=torch.empty(tokens,h,dtype=torch.bfloat16)
        state=dict(eligible=True,calls=0)
        def collective(out,packet):
            state["calls"]+=1
            per_rank=out.view(size,-1)
            for i in range(size):
                a=state["all_scores"][i].view(torch.uint8).flatten()
                b=state["all_ids"][i].view(torch.uint8).flatten()
                if i==rank:
                    assert torch.equal(a,packet[:len(a)]) and torch.equal(b,packet[len(a):len(a)+len(b)])
                    assert packet[-1].item()==state["eligible"]
                per_rank[i].copy_(torch.cat((a,b,b.new_full((16,),int(state["vote"][i])))))
        group=NS(world_size=size,all_gather_into_tensor=collective)
        workspace=NS(supports=lambda *args:True)
        env["layernorm_sp"]=NS(sp_exit_gather=lambda shard,tokens:torch.zeros(tokens,h,dtype=torch.bfloat16))
        class Experts:
            num_local_experts=7
            post_expert_norm="ordinary"
            def check(self,s,idx,fast):
                assert fast==all(state["vote"])
                expected_s=state["all_scores"].view(-1,k)[:tokens]
                expected_i=state["all_ids"].view(-1,k)[:tokens]
                assert torch.equal(s.view(torch.int32),expected_s.view(torch.int32))
                assert torch.equal(idx,expected_i)
                stats["fp32_score_bits"]+=s.numel();stats["int32_ids"]+=idx.numel()
                return (s.sum(1)+idx.float().sum(1)).bfloat16()[:,None].expand(-1,h)
            def _grouped_humming_and_combine_impl(self,x,loc,s,out,peer_plan):
                assert torch.equal(peer_plan[3],order(loc,7)[2]) and peer_plan[-1]==(loc<0).sum()
                out.copy_(self.check(s,loc,True))
            def __call__(self,x,s,idx,norm):
                assert s.untyped_storage().data_ptr()!=idx.untyped_storage().data_ptr()
                return self.check(s,idx,False)
        graph=NS(_break_fns=[])
        cap=NS(_barrier_fn=None,_end_current_segment=lambda:None,_begin_new_segment=lambda:None,cuda_graph=graph)
        wrapper=rt["eager_on_graph"](True,capture_stub=env["_capture_stub"])(env["_peer_experts_into"])
        token=ctx.set(cap);wrapper(workspace,Experts(),shard,scores,ids,group,dest);ctx.reset(token)
        assert state["calls"]==0 and not dest.count_nonzero() and len(graph._break_fns)==1
        addresses=tuple(x.data_ptr() for x in (shard,scores,ids,dest))
        for step in range(24):
            state["all_scores"]=torch.randn(size,rows,k,generator=rng)
            state["all_ids"]=torch.randint(-1,7,(size,rows,k),dtype=ids.dtype,generator=rng)
            if step%6==0:state["all_ids"].fill_(-1)
            if step%6==1:state["all_ids"].fill_(0)
            scores.copy_(state["all_scores"][rank]);ids.copy_(state["all_ids"][rank])
            state["vote"]=[True]*size
            if step%3==1:state["vote"][0]=False
            if step%3==2:state["vote"][-1]=False
            state["eligible"]=state["vote"][-1]
            graph._break_fns[0]()
            expected=(state["all_scores"].sum(-1)+state["all_ids"].float().sum(-1)).bfloat16().flatten()[:tokens]
            assert torch.equal(dest,expected[:,None].expand_as(dest)) and state["calls"]==step+1
            assert tuple(x.data_ptr() for x in (shard,scores,ids,dest))==addresses
            stats["live_replays" if all(state["vote"]) else "veto_replays"]+=1
print(json.dumps(dict(physical_gpu=False,source_tree=manifest["tree"],coverage="Pinned breakable capture and replay with CPU tensors and a synthetic group; real Gloo transport covered separately.",**stats),indent=2))
