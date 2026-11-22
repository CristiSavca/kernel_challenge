"""Replay the complete changed Muse body, wrapper hook, and BCG eager-tail entry on CPU."""
import ast
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch
from torch import nn

from independent_capacity import HERE, PARENT, PPProxyTensors, util


FILES = (
    "models/muse_spark.py", "models/muse_spark_omni.py", "layers/layernorm_sp.py",
    "layers/aux_hidden_states.py", "model_executor/runner/prefill_cuda_graph_runner.py",
    "model_executor/runner_backend/breakable_cuda_graph_backend.py",
)


def sources(path=None):
    if path is None:
        raw = {name:util.read_source(name,False) for name in FILES}
    else:
        raw = {name:(path/"python/sglang/srt"/name).read_text() for name in FILES}
    return ({name:ast.parse(s) for name,s in raw.items()},
            {name:hashlib.sha256(s.encode()).hexdigest() for name,s in raw.items()})


def forward_class(tree, name, base, env):
    cls = next(n for n in tree.body if getattr(n,"name",None)==name)
    code = copy.deepcopy(cls)
    code.bases = [ast.Name(id="Base")]
    code.decorator_list = []
    code.body = [n for n in code.body if getattr(n,"name",None)=="forward"]
    code.body[0].decorator_list=[]
    ns = dict(env, Base=base)
    exec("from __future__ import annotations\n" + ast.unparse(code), ns)
    return ns[name]


class Layer(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.i=i
        self.self_attn_input_norm=nn.Identity()
    def forward(self,positions,stream,attn,batch):
        value=stream*(1+self.i/32)+self.i/7
        return value,value.to(torch.bfloat16)


class Processor(nn.Module):
    def forward(self,input_ids,last,lm_head,batch,aux,**kwargs):
        return NS(hidden_states=aux, final=last)


def make_pipeline(source, rows, width, parent):
    size,rank=4,1
    group=NS(world_size=size,rank_in_group=rank)
    observations=[]
    def project(raw,*args):
        observations.append(raw)
        return None
    srt="model_executor/runner/prefill_cuda_graph_runner.py"
    sp_src=source["layers/layernorm_sp.py"]
    scatter=util.bind(sp_src,None,"sp_entry_scatter",dict(
        torch=torch,get_parallel=lambda:NS(tp_group=group),set_sp_num_tokens=lambda _:None,
        ceil_align=lambda a,b:(a+b-1)//b*b))
    sp=NS(runs_sp=lambda _:True,sp_entry_scatter=scatter,sp_exit_gather=lambda v,**kw:v)
    env=dict(torch=torch,nn=nn,layernorm_sp=sp,get_parallel=lambda:NS(tp_group=group),
             get_forward=lambda:NS(set=lambda *a:None),project_dflash_prefill_shard=project,
             copy=copy,_slice_output_rows=None)
    env["pack_aux_hidden_states"]=util.bind(source["layers/aux_hidden_states.py"],None,"pack_aux_hidden_states",env)
    model=source["models/muse_spark.py"]
    if not parent:
        env["_pack_muse_prompt_taps"]=util.bind(model,None,"_pack_muse_prompt_taps",env)
    Body=forward_class(model,"MuseSpark1xModel",nn.Module,env)
    Wrapper=forward_class(model,"MuseSparkModel",nn.Module,env)
    Text=forward_class(model,"MuseSparkTextForCausalLM",nn.Module,env)
    Omni=forward_class(source["models/muse_spark_omni.py"],"MuseSparkForConditionalGeneration",Text,env)
    body=Body()
    body.layers=nn.ModuleList(Layer(i) for i in range(1,6))
    body.layers_to_capture=[0,1,2,3,4]
    body.residual_dtype=torch.float32
    body.operator_dtype=torch.bfloat16
    body.embed_norm=None
    body.norm=nn.Identity()
    target=Omni()
    target.model=Wrapper()
    target.model.language_model=body
    target.capture_aux_hidden_states=True
    target.logits_processor=Processor()
    target.lm_head=nn.Identity()
    target.get_input_embeddings=lambda:lambda ids:ids[:,None].float().expand(-1,width)
    execute=util.bind(source[srt],"PrefillCudaGraphRunner","_execute_body_capture",env)
    be_src=source["model_executor/runner_backend/breakable_cuda_graph_backend.py"]
    Backend=type("Backend",(),{n:util.bind(be_src,"BreakableCudaGraphBackend",n,dict(torch=torch,PPProxyTensors=PPProxyTensors))
                                 for n in ("_copy_output_to_buffer","_slice_output")})
    backend=Backend()
    runner=NS(_is_full_backend=False,is_dsv41_model=False,model_runner=NS(model=target,pp_group=NS(is_first_rank=True)),
              buffer_registry=NS(has_slot=lambda *_:False),layer_model=target.model,backend=backend,
              _restore_qwen_bcg_hc_output=lambda x:x,_prefill_forward_context=lambda *a,**k:nullcontext(),
              _use_draft_input_embeds=False)
    max_local=(rows+size-1)//size
    raw=None if parent else torch.empty((max_local,5*width),dtype=torch.bfloat16)
    local=NS()
    local.buf=(torch.empty((max_local,width),dtype=torch.bfloat16),[torch.empty((max_local,width),dtype=torch.bfloat16)
               for _ in range(5)] if parent else list(raw.split(width,dim=-1)))
    runner._muse_prompt_taps=raw
    backend.replay=lambda *a,**kw:backend._slice_output(local.buf,local.rows)
    def run(tokens,hook_kind,global_hook):
        local.rows=(tokens+size-1)//size
        mode=NS(is_target_verify=lambda:False,is_extend=lambda:True,is_decode=lambda:False)
        batch=NS(input_ids=torch.arange(tokens),positions=torch.arange(tokens),
                 forward_mode=mode,capture_hidden_mode=NS(is_full=lambda:True),
                 multi_item_delimiter_indices=None,token_indices_to_pool=None,
                 mm_input_embeds=None,contains_mm_inputs=lambda:False,dflash_verify_raw_buffer=None)
        if not parent:
            batch.dflash_prefill_raw_buffer=raw[:local.rows]
        cap=target.model.forward(batch.input_ids,batch.positions,batch,input_embeds=target.get_input_embeddings()(batch.input_ids))
        backend._copy_output_to_buffer(cap,local.buf,local.rows)
        observed=[]
        def hook(module,args,out):
            if module is not target.model: return
            last,taps=out
            observed.append(dict(container=type(taps).__name__,shapes=[list(t.shape) for t in taps],
                                 strides=[list(t.stride()) for t in taps],offsets=[t.storage_offset() for t in taps]))
            if hook_kind=="replace":
                repl=list(taps)
                repl[2]=torch.flip(taps[2],[-1])+1.25
                return last,repl
            if hook_kind=="inplace":
                taps[1].neg_()
                return last,taps
            if hook_kind=="reinterpret":
                return last,[t.view(torch.float16) for t in taps]
            if hook_kind=="reshape":
                return last,[t.view(local.rows,2,width//2) for t in taps]
        handle=None
        if hook_kind:
            handle=(torch.nn.modules.module.register_module_forward_hook(hook) if global_hook
                    else target.model.register_forward_hook(hook))
        try:
            result=execute(runner,batch,batch,tokens,tokens,tokens)
        finally:
            if handle is not None:handle.remove()
        return (result.hidden_states.clone(),result.final.clone(),observed,
                False if parent else observations[-1].is_set_to(raw[:local.rows]))
    return run


def main():
    torch.set_num_threads(1)
    parent,p_sha=sources()
    patch=HERE/"patch.diff"
    if len(sys.argv)>1:
        child,c_sha=sources(Path(sys.argv[1]))
    else:
        util.PARENT=patch.read_text()
        child,c_sha=sources()
    p=make_pipeline(parent,32,8,True)
    c=make_pipeline(child,32,8,False)
    check=vals=hooks=0
    for tokens in (3,4,7,12,19,32):
        for hook,global_hook in [(None,False)]+[(h,g) for h in ("observe","replace","inplace","reinterpret","reshape") for g in (False,True)]:
            lhs,lf,lobs,_=p(tokens,hook,global_hook)
            rhs,rf,robs,alias=c(tokens,hook,global_hook)
            assert lhs.dtype==rhs.dtype and lhs.shape==rhs.shape, (tokens,hook,lhs.dtype,rhs.dtype)
            assert torch.equal(lhs.view(torch.int16),rhs.view(torch.int16)) and torch.equal(lf,rf),(tokens,hook)
            assert lobs==robs,(tokens,hook,lobs,robs)
            if hook is None:assert alias
            if hook in ("replace","reinterpret","reshape"):assert not alias
            check+=1; vals+=lhs.numel();hooks+=len(lobs)
    digest=hashlib.sha256(patch.read_bytes()).hexdigest() if len(sys.argv)==1 else None
    print(json.dumps(dict(patch_sha256=digest, paired_replays=check,hook_observations=hooks,compared_tap_words=vals,
                          capture= "Real MuseSpark1xModel, MuseSparkModel, registered Omni, native Breakable copy/slice and _execute_body_capture",
                          witness= "TP4 CPU rank 1; synthetic transformer blocks and final logits receiver; identity gather; no H200",
                          candidate_source_sha256=c_sha,parent_source_sha256=p_sha),indent=2))


if __name__=="__main__":
    main()
