"""Execute the V64 and V149 SP/experts entry points with real CPU Torch/Gloo."""
from __future__ import annotations
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

V64 = "b5a37c2923f8d1ab24c1c5cf7920a21e2c086774"
SRC = Path("python/sglang/srt")
HUM = SRC / "layers/muse_spark/muse_spark_humming_moe.py"
MOE = SRC / "layers/muse_spark/muse_spark_fp4_moe.py"
MODEL = SRC / "models/muse_spark.py"
MAP = SRC / "layers/moe/token_dispatcher/standard.py"


def source(stage, path, new):
    if new and (stage / path).exists():
        return (stage / path).read_text()
    return subprocess.check_output(["git", "-C", stage, "show", f"{V64}:{path}"], text=True)


def install(text, names, env):
    class Drop(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None
    for name in names:
        f=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name)
        f.decorator_list=[]; f=Drop().visit(f)
        exec(compile(ast.Module([f],type_ignores=[]),name,"exec",dont_inherit=False),env)


class Stream:
    def __init__(self,label,events):
        self.label=label;self.events=events
    def wait_stream(self,other):
        self.events.append((self.label,"wait",other.label))


class Cuda:
    def __init__(self,events,mode):
        self.main=Stream("main",events); self.side=Stream("sort",events); self.mode=mode;self.active=self.main
    def current_stream(self,device=None):return self.active
    def is_current_stream_capturing(self):return self.mode in ("breakable","full")
    @contextlib.contextmanager
    def stream(self,value):
        old=self.active;self.active=value
        try:yield
        finally:self.active=old


class Expert(torch.nn.Module):
    def forward(self,*args,**kwargs):
        return self.env["forward"](self,*args,**kwargs)


class Humming:
    @staticmethod
    def forward_layer(layer,*,inputs,outputs,expert_layout,sublayer_name,**kw):
        bounds=expert_layout.tolist()
        layer.gemms.append([sublayer_name,bounds])
        for i,(a,b) in enumerate(zip(bounds,bounds[1:])):
            outputs[a:b]=inputs[a:b].float() @ layer.weights[sublayer_name][i]


def finalize(outputs,weights,*,norm_weight,eps,weight_first,slot_expert_ids,slot_row_indices,out):
    assert slot_expert_ids is None
    combined=torch.zeros_like(out,dtype=torch.float32)
    for k in range(weights.shape[1]):
        a=slot_row_indices[:,k];live=a>=0
        x=outputs[a[live].long()].float()
        if weight_first:x=x*norm_weight
        x=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)
        if not weight_first:x=x*norm_weight
        combined[live]+=x*weights[live,k,None]
    out.copy_(combined)


def runpath(stage,rank,size,rows,ids_local,score_local,x_local,mode,new,wt):
    events=[]; patterns=[];cuda=Cuda(events,mode)
    # Native ATen/CUDA and Humming math are replaced only at hardware boundaries.
    class Torch:
        def __getattr__(self,k):return getattr(torch,k)
    proxy=Torch();proxy.cuda=cuda
    env=dict(torch=proxy,nn=torch.nn,MuseSpark1xFp4Experts=Expert,
             HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256,_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256,
             _GROUPED_COMPUTE="grouped",HummingMethod=Humming,
             is_in_breakable_cuda_graph=lambda:mode=="breakable",
             fused_post_expert_norm_weighted_sum=finalize,
             silu_and_mul=lambda g,o:o.copy_(torch.nn.functional.silu(g.float().chunk(2,1)[0])*g.float().chunk(2,1)[1]))
    install(source(stage,MAP,False),["_map_topk_ids_to_local_experts"],env)
    hf=["local_humming_expert_ids","humming_grouped_expert_outputs"]+(["humming_grouped_route_order"] if new else [])
    install(source(stage,HUM,new),hf,env)
    if new:
        native=env["humming_grouped_route_order"]
        def order(*args):
            events.append(("order",cuda.active.label))
            result=native(*args)
            patterns.append([a.clone() for a in result])
            return result
        env["humming_grouped_route_order"]=order
    install(source(stage,MOE,new),["forward","_forward_w4a16","_grouped_humming_and_combine_impl"],env)
    install(source(stage,MODEL,new),["_forward_sp_replicated_experts"]+(["_prepare_sp_grouped_routes"] if new else []),env)
    e=Expert();e.env=env;e.num_local_experts=4;e.global_num_experts=4*size
    e.use_humming=e.use_w4a16=True;e.post_expert_norm=object();e._post_expert_norm=e.post_expert_norm
    e._post_expert_norm_weight=torch.arange(1,x_local.shape[1]+1).bfloat16()*0.25
    e._post_expert_norm_eps=1e-6;e._post_expert_norm_weight_first=wt
    e.intermediate_size_per_partition=x_local.shape[1];e._expert_map=torch.full((4*size,),-1,dtype=torch.int32)
    e._expert_map[rank*4:(rank+1)*4]=torch.arange(4,dtype=torch.int32)
    e.humming_grouped_tunings={"w13":"w13","w2":"w2"};e.gemms=[]
    n=x_local.shape[1]
    tr=torch.Generator().manual_seed(2167+rank)
    e.weights={k:torch.randint(-3,4,(4,n,n*(2 if k=="w13" else 1)),generator=tr).float()/8 for k in ("w13","w2")}
    e.w13_global_scale=torch.tensor([0.75,1.125,0.625,1.25]).bfloat16()
    e.w2_global_scale=torch.tensor([1.25,0.75,1.5,1.125]).bfloat16()
    e._forward_w4a16=lambda *a,**kw:env["_forward_w4a16"](e,*a,**kw)
    e._grouped_humming_and_combine=lambda *a:env["_grouped_humming_and_combine_impl"](e,*a)
    ref=list()
    if mode=="hook" and rank==0:
        def pre(mod,args):
            ref.append(1);x,s,ids,norm=args
            return x,s,torch.where(ids>=0,(ids+1)%(4*size),ids),norm
        e.register_forward_pre_hook(pre)
    elif mode=="override" and rank==0:
        e.forward=lambda *a,**kw:env["forward"](e,*a,**kw)
    obj=NS(experts=e,top_k=ids_local.shape[1],num_experts=e.global_num_experts,
           sp_order_humming_routes=True,sp_route_stream=cuda.side)
    if new:
        orig=env["_prepare_sp_grouped_routes"]
        def prepare(*a):
            # The call only inspects the input's device type/metadata.
            with patch.object(torch.Tensor,"is_cuda",property(lambda _:True)):
                return orig(obj,*a)
        obj._prepare_sp_grouped_routes=prepare
    by_id={id(ids_local):"ids",id(score_local):"scores",id(x_local):"pooled"}
    def gather(x,nrows):
        events.append(("gather",by_id[id(x)],cuda.active.label))
        out=[torch.empty_like(x) for _ in range(size)];dist.all_gather(out,x)
        return torch.cat(out)[:nrows]
    def reduce(x,_):
        events.append(("return",cuda.active.label))
        y=x.float().contiguous();dist.all_reduce(y)
        z=torch.nn.functional.pad(y.bfloat16(),(0,0,0,(-len(y))%size))
        return z.chunk(size)[rank]
    def dense_fallback(layer,x,indices):
        # Cover native full-capture/small shape dispatcher as an indexed reference.
        out=torch.zeros((len(x),indices.shape[1],n),dtype=x.dtype)
        for i in range(4):
            a,b=torch.where(indices==i)
            gu=(x[a].float()@e.weights["w13"][i]).bfloat16()*e.w13_global_scale[i]
            g,u=gu.float().chunk(2,1);act=(torch.nn.functional.silu(g)*u).bfloat16()
            out[a,b]=(act.float()@e.weights["w2"][i]).bfloat16()*e.w2_global_scale[i]
        return out,indices
    def dispatch_finalize(v,w,*,norm_weight,eps,weight_first,slot_expert_ids=None,slot_row_indices=None,out=None):
        if slot_row_indices is not None:
            return finalize(v,w,norm_weight=norm_weight,eps=eps,weight_first=weight_first,slot_expert_ids=None,slot_row_indices=slot_row_indices,out=out)
        ix=torch.arange(w.numel()).reshape(w.shape);ix=torch.where(slot_expert_ids>=0,ix,-1)
        out=torch.empty((len(w),n),dtype=v.dtype)
        finalize(v.flatten(0,1),w,norm_weight=norm_weight,eps=eps,weight_first=weight_first,slot_expert_ids=None,slot_row_indices=ix,out=out)
        return out
    env.update(layernorm_sp=NS(sp_num_tokens=lambda:rows,sp_exit_gather=gather,reduce_scatter_token_rows=reduce),
               get_parallel=lambda:NS(tp_group=None),humming_expert_outputs=dense_fallback,
               fused_post_expert_norm_weighted_sum=dispatch_finalize)
    with torch.no_grad(), patch.object(torch.Tensor,"record_stream",lambda self,stream:events.append(("record",stream.label))):
        out=env["_forward_sp_replicated_experts"](obj,x_local,score_local,ids_local)
    assert len(ref)==int(mode=="hook" and rank==0)
    if new:
        large=rows*obj.top_k>=256*4*size
        assert [z[1] for z in events if z[0]=="gather"]==(["scores","ids","pooled"] if large else ["pooled","scores","ids"])
        assert all(z[-1]=="main" for z in events if z[0] in ("return","gather"))
        side=[z for z in events if z==("order","sort")]
        expect=large and mode!="full" and not (mode in ("hook","override") and rank==0)
        assert len(side)==int(expect),(rows,rank,mode,events)
        if expect:
            for mark in (("sort","wait","main"),("gather","pooled","main"),("main","wait","sort")):assert mark in events
            assert events.index(("gather","ids","main"))<events.index(("order","sort"))<events.index(("gather","pooled","main"))<events.index(("main","wait","sort"))<events.index(("return","main"))
            assert ("order","main") not in events
            assert (("record","sort") in events)==(mode!="breakable")
    return out, e.gemms, events


def worker(rank,size,stage,folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(folder/"rdzv"),rank=rank,world_size=size)
    checks=elems=overlap=0
    try:
        for rows in (7,128*size,128*size+3,8192):
            for i,(mode,pattern) in enumerate((("eager","all_remote"),("breakable","mix"),("hook","rank0"),
                                               ("override","duplicate"),("full","mix"),("eager","drop"))):
                local=(rows+size-1)//size
                g=torch.Generator().manual_seed(rows*7+rank*97+i)
                ids=torch.randint(0,4*size,(local,8),generator=g,dtype=torch.int32)
                if pattern=="all_remote":ids[:]=4*(size-1)
                elif pattern=="rank0":ids[:]=0
                elif pattern=="duplicate":ids[:,1::2]=ids[:,::2]
                elif pattern=="drop":ids[:,::3]=-1
                w=torch.randn((local,8),generator=g).softmax(-1)
                x=(torch.randint(-7,8,(local,8),generator=g).float()/8).bfloat16()
                a,old,_=runpath(stage,rank,size,rows,ids,w,x,mode,False,bool(i%2))
                b,new,events=runpath(stage,rank,size,rows,ids,w,x,mode,True,bool(i%2))
                assert torch.equal(a.view(torch.int16),b.view(torch.int16)) and old==new
                checks+=1;elems+=b.numel();overlap+=int(("order","sort") in events)
        (folder/f"{rank}.json").write_text(json.dumps([checks,elems,overlap]))
    finally:dist.destroy_process_group()


def main():
    stage=Path(sys.argv[1]).resolve()
    results={}
    for size in (2,4,8):
        with tempfile.TemporaryDirectory() as d:
            tmp=Path(d)
            mp.spawn(worker,args=(size,stage,tmp),nprocs=size,join=True)
            results[str(size)]=[sum(x) for x in zip(*(json.loads((tmp/f"{r}.json").read_text()) for r in range(size)))]
    print(json.dumps(dict(torch=torch.__version__,rank_cases_values_early_plans=results,
          production_methods=["_forward_sp_replicated_experts","_prepare_sp_grouped_routes","forward","_forward_w4a16",
                              "_grouped_humming_and_combine_impl","humming_grouped_expert_outputs","humming_grouped_route_order"],
          real_gloo_collectives=True,physical_cuda=False,
          includes=["changing rows and routes","ragged","mixed rank hooks/overrides","full-capture fallback","breakable instruction ordering","all remote and drop sentinels"]),indent=2))


if __name__=="__main__":main()
