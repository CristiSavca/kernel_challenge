"""Replay the submitted SP body, output allocator, BCG copies, FC and draft KV on CPU/Gloo."""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


MODEL = "models/muse_spark.py"
RUNNER = "model_executor/runner/prefill_cuda_graph_runner.py"
BACKEND = "model_executor/runner_backend/breakable_cuda_graph_backend.py"


def get_source(stage,path,parent=False):
    path="python/sglang/srt/"+path
    return (subprocess.check_output(["git","-C",stage,"show",":"+path]).decode()
            if parent else (Path(stage)/path).read_text())


def load(tree, name, cls, env):
    if cls is not None:
        tree=next(n for n in tree.body if getattr(n,"name",None)==cls)
    node=next(n for n in tree.body if getattr(n,"name",None)==name)
    node.decorator_list=[]
    fn=ast.parse("from __future__ import annotations")
    fn.body.append(node)
    exec(compile(ast.fix_missing_locations(fn),"<submitted source>","exec"),env)
    return env[name]


def gather(local):
    with torch.inference_mode(False):
        packets=[torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(packets,local.contiguous())
    return torch.cat(packets)


class Block(torch.nn.Module):
    def __init__(self,i):
        super().__init__()
        self.delta=(i+1)/8
        self.self_attn_input_norm=lambda x:x
    def forward(self,positions,state,attn,batch):
        return state+self.delta,attn


def classes(stage,parent,rank,size):
    model=ast.parse(get_source(stage,MODEL,parent))
    runner=ast.parse(get_source(stage,RUNNER,parent))
    bcg=ast.parse(get_source(stage,BACKEND,parent))
    parallel=NS(tp_group=NS(world_size=size,rank_in_group=rank),attn_dp_enabled=False,attn_cp_size=1)
    env=dict(torch=torch,get_parallel=lambda:parallel,
             ceil_align=lambda n,k:((n+k-1)//k)*k,
             set_sp_num_tokens=lambda n:None,
             get_forward=lambda:NS(set=lambda *a:None),
             _ceil_div=lambda n,k:(n+k-1)//k, PPProxyTensors=type("Proxies",(),{}))
    sp=types.ModuleType("layernorm_sp")
    sp.runs_sp=lambda mode:mode.is_extend()
    sp.sp_entry_scatter=load(ast.parse(get_source(stage,"layers/layernorm_sp.py",parent)),
                             "sp_entry_scatter",None,env)
    env["layernorm_sp"]=sp
    load(ast.parse(get_source(stage,"layers/aux_hidden_states.py",parent)),"pack_aux_hidden_states",None,env)
    class Body(torch.nn.Module):
        forward=load(model,"forward","MuseSpark1xModel",env)
    class Wrapper(torch.nn.Module):
        pass
    class Target(torch.nn.Module):
        pass
    class BCG:
        pass
    class Runner:
        pass
    for name in ("_alloc_full_buffer","_copy_output_to_buffer","_output_rows","_slice_output"):
        setattr(BCG,name,load(bcg,name,"BreakableCudaGraphBackend",env))
    if not parent:
        setattr(BCG,"bind_shared_output",load(bcg,"bind_shared_output","BreakableCudaGraphBackend",env))
        env["BreakableCudaGraphBackend"]=BCG
        setattr(Runner,"bind",load(runner,"_bind_muse_prompt_taps","PrefillCudaGraphRunner",env))
        load(model,"_pack_muse_prompt_taps",None,env)
        package=types.ModuleType("sglang.srt.layers")
        package.layernorm_sp=sp
        sys.modules["sglang.srt.layers"]=package
        mod=types.ModuleType("muse_spark")
        mod.MuseSpark1xModel,mod.MuseSparkModel,mod.MuseSparkTextForCausalLM=Body,Wrapper,Target
        sys.modules["sglang.srt.models.muse_spark"]=mod
    body=Body();body.layers=torch.nn.ModuleList(Block(i) for i in range(6))
    body.config=NS(hidden_size=16);body.operator_dtype=torch.bfloat16;body.residual_dtype=torch.float32
    body.norm=lambda x:x;body.layers_to_capture=[0,1,2,4,5]
    wrapper=Wrapper();wrapper.language_model=body
    target=Target();target.model=wrapper;target.capture_aux_hidden_states=True
    rt=Runner(); rt.model_runner=NS(model=target,is_draft_worker=False,lora_manager=None,
                                     spec_algorithm=NS(is_dflash=lambda:True))
    rt.backend=BCG();rt.layer_model=wrapper;rt.pp_size=1;rt.device="cpu"
    rt.backend._shared_output_buffer=None;rt.backend._graphs={};rt.backend._debug_eager=False
    rt.backend._cuda_graph_runner=NS(cuda_graph_output_rows=lambda y:None)
    rt.capture_forward_mode=NS(is_target_verify=lambda:False,is_extend=lambda:True)
    rt.capture_hidden_mode=NS(is_full=lambda:True)
    rt.max_num_tokens=264;rt._muse_prompt_taps=None
    return rt,body,env


def run(rank,size,stage,directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(Path(directory)/f"sync{size}"),rank=rank,world_size=size)
    parent,pbody,penv=classes(stage,True,rank,size)
    child,cbody,cenv=classes(stage,False,rank,size)
    torch.manual_seed(6)
    fc=torch.randint(-4,5,(16,80)).to(torch.bfloat16)/4
    norm=torch.nn.RMSNorm(16,eps=1e-5,dtype=torch.bfloat16)
    kv=torch.randint(-3,4,(16,10)).to(torch.bfloat16).add_(rank+1)
    measured=dict(captures=0,raw_bf16=0,fc_bf16=0,kv_bf16=0,parent_copy_bytes=0,child_copy_bytes=0)
    with torch.inference_mode():
        for mismatch in ("lora", "duplicate_taps","hook","oom"):
            child._muse_prompt_taps=None
            if mismatch=="lora": child.model_runner.lora_manager=object()
            if mismatch=="duplicate_taps": cbody.layers_to_capture=[0,0,1,2,3]
            if mismatch=="hook": h=child.layer_model.register_forward_hook(lambda m,a,o:o)
            if mismatch=="oom":
                with patch.object(torch,"empty",side_effect=torch.OutOfMemoryError()):
                    child.bind()
            else:
                child.bind()
            assert child._muse_prompt_taps is None and child.backend._shared_output_buffer is None
            child.model_runner.lora_manager=None;cbody.layers_to_capture=[0,1,2,4,5]
            if mismatch=="hook":h.remove()
        child.bind();raw=child._muse_prompt_taps
        assert raw.shape==((264+size-1)//size,80)
        cb=child.backend._shared_output_buffer
        assert len(cb[1])==5 and all(x.untyped_storage().data_ptr()==raw.untyped_storage().data_ptr() for x in cb[1])
        parent_copy,child_copy=[],[]
        original=torch.Tensor.copy_
        for n in (264,256,255,129,97,32,17,8,3,1,97,264):
            x=torch.randint(-8,9,(n,16),dtype=torch.int16).float()/4
            local=(n+size-1)//size
            batch=NS(forward_mode=child.capture_forward_mode)
            native=pbody(None,None,batch,input_embeds=x)
            if parent.backend._shared_output_buffer is None:
                parent.backend._shared_output_buffer=parent.backend._alloc_full_buffer(native,n)
            pitch=parent.backend._output_rows(native,n)
            assert pitch==local
            def trace_parent(dst,src,*args,**kwargs):
                parent_copy.append(src.numel()*src.element_size());return original(dst,src,*args,**kwargs)
            with patch.object(torch.Tensor,"copy_",trace_parent):
                parent.backend._copy_output_to_buffer(native,parent.backend._shared_output_buffer,pitch)
            expected=parent.backend._slice_output(parent.backend._shared_output_buffer,pitch)
            batch.dflash_prefill_raw_buffer=raw[:local]
            observed=cbody(None,None,batch,input_embeds=x)
            assert isinstance(observed[1],list) and len(observed[1])==len(expected[1])
            def trace_child(dst,src,*args,**kwargs):
                child_copy.append(src.numel()*src.element_size());return original(dst,src,*args,**kwargs)
            with patch.object(torch.Tensor,"copy_",trace_child):
                child.backend._copy_output_to_buffer(observed,cb,pitch)
            shared=child.backend._slice_output(cb,pitch)
            assert expected[0].equal(shared[0])
            got=cenv["_pack_muse_prompt_taps"](shared[1],raw)
            want=penv["pack_aux_hidden_states"](expected[1])
            assert got.is_set_to(raw[:local]) and got.view(torch.int16).equal(want.view(torch.int16))
            projected=norm(torch.nn.functional.linear(got,fc))
            reference=norm(torch.nn.functional.linear(want,fc))
            assert projected.view(torch.int16).equal(reference.view(torch.int16))
            out=gather(projected)[:n]@kv
            ref=gather(reference)[:n]@kv
            assert out.view(torch.int16).equal(ref.view(torch.int16))
            measured["captures"]+=1;measured["raw_bf16"]+=got.numel()
            measured["fc_bf16"]+=projected.numel();measured["kv_bf16"]+=out.numel()
            assert not torch.isnan(raw[:local]).any()
        measured["parent_copy_bytes"]=sum(parent_copy);measured["child_copy_bytes"]=sum(child_copy)
        assert measured["parent_copy_bytes"]==6*measured["child_copy_bytes"]
        # Direct source views of a different logical dtype, lazy negative bits,
        # clone and replacement must all use ordinary packing.
        shape=3;leaves=list(raw[:shape].split(16,-1));pack=cenv["_pack_muse_prompt_taps"]
        for mode in ("clone","reinterpret","neg","replace","one-row"):
            modified=list(leaves)
            if mode=="clone": modified[2]=modified[2].clone()
            if mode=="reinterpret": modified=[x.view(torch.float16) for x in modified]
            if mode=="neg": modified=[torch._neg_view(x) for x in modified]
            if mode=="replace":modified[0]=modified[0]+1
            if mode=="one-row": modified=[x[:1].contiguous().view(torch.float16) for x in modified]
            got=pack(modified,raw);ref=torch.cat(modified,dim=-1)
            assert got.dtype==ref.dtype and got.view(torch.int16).equal(ref.view(torch.int16))
            assert got.untyped_storage().data_ptr()!=raw.untyped_storage().data_ptr()
        # An already-used backend never relinquishes its existing output tree.
        assert not child.backend.bind_shared_output((None,None))
        # One rank's allocation failure is local: the outer model consumes the
        # native leaf list there and a packed view elsewhere before the same Gloo call.
        mixed,mbody,menv=classes(stage,False,rank,size)
        if rank==1:
            with patch.object(torch,"empty",side_effect=torch.OutOfMemoryError()):
                mixed.bind()
        else:
            mixed.bind()
        assert (mixed._muse_prompt_taps is None) == (rank==1)
        n=97;rows=(n+size-1)//size
        x=torch.arange(n*16,dtype=torch.float32).reshape(n,16)/32
        b=NS(forward_mode=mixed.capture_forward_mode,dflash_prefill_raw_buffer=(
            mixed._muse_prompt_taps[:rows] if rank!=1 else None))
        res=mbody(None,None,b,input_embeds=x)
        if rank==1:
            mixed.backend._shared_output_buffer=mixed.backend._alloc_full_buffer(res,n)
        mixed.backend._copy_output_to_buffer(res,mixed.backend._shared_output_buffer,rows)
        taps=mixed.backend._slice_output(mixed.backend._shared_output_buffer,rows)[1]
        fresh=menv["_pack_muse_prompt_taps"](taps,mixed._muse_prompt_taps)
        native=pbody(None,None,NS(forward_mode=parent.capture_forward_mode),input_embeds=x)
        ref=penv['pack_aux_hidden_states'](native[1])
        assert gather(fresh).equal(gather(ref))
        measured["asymmetric_oom_groups"]=1
    dist.destroy_process_group()
    (Path(directory)/f"{size}-{rank}.json").write_text(json.dumps(dict(size=size,rank=rank,**measured)))


if __name__=="__main__":
    details=[]
    with tempfile.TemporaryDirectory() as directory:
        for size in (4,8):
            mp.spawn(run,args=(size,sys.argv[1],directory),nprocs=size)
            details.extend(json.loads((Path(directory)/f"{size}-{r}.json").read_text()) for r in range(size))
    print(json.dumps(dict(hardware="CPU Gloo",torch_version=str(torch.__version__),rank_runs=details,
        totals={k:sum(v[k] for v in details) for k in ("captures","raw_bf16","fc_bf16","kv_bf16",
                  "parent_copy_bytes","child_copy_bytes")}),indent=2))
