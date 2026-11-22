"""Replay V68/V254 model, provider gate, native JIT binding and Humming SP handoff with CPU tensors."""
import ast
from collections import Counter
import contextlib
import enum
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch, PropertyMock

import torch
from torch import nn

stage=Path(sys.argv[1])
here=Path(__file__).resolve().parent
parent=json.loads((here/'validation.json').read_bytes())['parent_tree']
models='python/sglang/srt/models/muse_spark.py'
routing='python/sglang/kernels/ops/moe/muse_spark_routing.py'
jit='python/sglang/kernels/ops/moe/muse_spark_routing_jit.py'
get=lambda p:(stage/p).read_text()
old=lambda p:subprocess.check_output(['git','-C',stage,'show',parent+':'+p]).decode()
def functions(src, names, env, cls=None):
    nodes=ast.parse(src).body
    if cls: nodes=next(node.body for node in nodes if getattr(node,'name',None)==cls)
    chosen=[node for node in nodes if getattr(node,'name',None) in names]
    assert len(chosen)==len(names)
    for node in chosen: node.decorator_list=[]
    exec('from __future__ import annotations\n'+ast.unparse(ast.Module(chosen,[])),env)
    return {name:env[name] for name in names}
class TorchProxy:
    def __getattr__(self,n):return getattr(torch,n)
    def mm(self,x,w,out_dtype):
        assert x.dtype==w.dtype==torch.bfloat16 and out_dtype==torch.float32
        mm.append((tuple(x.shape),x.stride(),tuple(w.shape),w.stride()))
        return torch.mm(x.float(),w.float())
t=TorchProxy()
mm=[]; launches=[]; captures=[]; hooks=[]; returns=[]; responses=[]
class KB(enum.Enum): JIT=1; TRITON=2
settings=NS(forced=None,provider='jit',prepare=True,norm='sigmoid',capture=False)
pkg=ModuleType('sglang.kernels')
pkg.fused_op=NS(get_fused_op_backend=lambda:settings.forced,_trace_enabled=False)
pkg.kernel_api_logging=NS(_KERNEL_API_LOG_LEVEL=0)
modjit=ModuleType('sglang.kernels.ops.moe.muse_spark_routing_jit')
def kernel(high,bias,weights,ids,packed,mapping,valid,mult,scale,eps,*res):
    assert len(res) in (0,1)
    if res: assert res[0].shape==high.shape and res[0].stride()==high.stride()
    x=high+res[0] if res else high
    launches.append(('split' if res else 'stock',x.clone()))
    activated=torch.sigmoid(x*mult) if settings.norm=='sigmoid' else torch.softmax(x*mult,dim=1)
    selected=torch.argsort(activated+bias,descending=True,dim=1,stable=True)[:,:ids.shape[1]]
    scores=activated.gather(1,selected)
    weight=scores*(scale/(scores.sum(1,keepdim=True)+eps))
    i=selected.int() if mapping is None else mapping[selected]
    if valid is not None: i=i.masked_fill(torch.arange(len(i)).view(-1,1)>=valid,-1)
    weights.copy_(weight)
    ids.copy_(i)
    if packed is not None:
        wb=weights.bfloat16().view(torch.int16).int()&65535
        packed.copy_((i<<16)|torch.where(i<0,0,wb))
    responses.append([weights.clone(),ids.clone(),packed.clone()])
wrapped=NS(fused_muse_spark_routing=kernel)
je=vars(modjit);je.update(torch=torch,_NORMALIZATION_MODES={'sigmoid':0,'softmax':1},_jit_module=lambda *args:wrapped)
functions(get(jit),['_experts_per_lane','_warps_per_block','fused_muse_spark_routing_jit'],je)
je['_LARGE_BATCH_TOKENS']=2048
je['_split_module']=lambda *args:wrapped if settings.prepare else None
class Base(nn.Module):
    def __init__(self):super().__init__();self._forward_method=None;self.is_torch_compile=False
    def backend_eligible(self,*args,**kwargs): return True
    def _forward_backend_dynamic(self,*args,**kwargs):return self.forward_jit(*args,**kwargs)
    def _resolve_forward_method(self):
        self._dynamic_backend_candidates=(KB.JIT if settings.provider=='jit' else KB.TRITON,)
        return self._forward_backend_dynamic
    def forward(self,*args,**kwargs):return self.forward_jit(*args,**kwargs)
re={'BaseFusedOp':Base,'torch':t,'KernelBackend':KB}
# Keep the production JIT eligibility implementation including zero-argument super().
rr=ast.parse(get(routing));cls=next(n for n in rr.body if getattr(n,'name',None)=='FusedMuseSparkRouting')
cls.body=[n for n in cls.body if getattr(n,'name',None) in ['__init__','split_logits_module','forward_jit','backend_eligible']]
exec('from __future__ import annotations\n'+ast.unparse(cls),re)
Route=re['FusedMuseSparkRouting']
oldgate=type('OldGate',(nn.Module,),functions(old(models),['_build_split2','forward_split2'],{'torch':t},'MuseSpark1xRouter'))
gateclass=type('MuseSpark1xRouter',(nn.Module,),functions(get(models),['_build_split2','_split2_parts','forward_split2'],{'torch':t},'MuseSpark1xRouter'))
class Mode:
    sp_active=True
state=Mode()
comm=NS(sp_num_tokens=lambda:16384,sp_exit_gather=lambda x,**kw:x,reduce_scatter_token_rows=lambda x,group:x)
comm.sp_exit_gather=lambda x,*args,**kw:x
class Dispatch:
    def dispatch(self,input,top): assert isinstance(top[2],torch.Tensor);returns.append(top[2].clone());return top
    def combine(self,out):return out
class Experts:
    use_w4a16=True;post_expert_norm=None
    def __call__(self,input,scores,ids,norm,**kw):return (scores*(ids.float()+1)).sum(dim=1)
    def forward_dispatched(self,top,norm):return (top[0]*(top[1].float()+1)).sum(dim=1)
env=dict(torch=t,_is_cpu=False,get_forward=lambda:state,layernorm_sp=comm,contextlib=contextlib,
    _use_split_bf16_router=lambda rows:rows>8,StandardTopKOutput=lambda *a:a,
    get_global_experts_capturer=lambda:NS(capture=lambda **kw:captures.append(kw['topk_indices'].clone())) if settings.capture else None,
    get_parallel=lambda:NS(tp_group=NS(ca_comm=None)),FusedMuseSparkRouting=Route,MuseSpark1xRouter=gateclass)
class Owner:
    router_in_fp32=True;pre_expert_proj=pre_expert_norm=None;tp_size=8;nvfp4_checkpoint=True;top_k=8
    sp_replicate_expert_input=True;use_megamoe=False;sp_dispatcher=Dispatch();experts=Experts();layer_idx=3
    def _routing_op(self):return self.route
methods=['_forward_experts','_forward_sp_experts','_forward_sp_replicated_experts']
Old=type('Old',(Owner,),functions(old(models),methods,env.copy(),'MuseSpark1xMoE'))
New=type('New',(Owner,),functions(get(models),methods,env.copy(),'MuseSpark1xMoE'))
def pair(rows,k,n,kind):
    settings.norm='softmax' if kind=='softmax' else 'sigmoid'
    settings.provider='other' if kind=='alternative' else 'jit'
    settings.forced='native' if kind=='forced' else None
    settings.capture=kind=='capturer';settings.prepare=kind!='failed_build'
    pkg.fused_op._trace_enabled=kind=='trace'; pkg.kernel_api_logging._KERNEL_API_LOG_LEVEL=int(kind=='debug')
    state.sp_active=kind!='non_sp'
    weights=torch.randn(n,k,generator=rng)*.1
    x=torch.randn(rows,k,generator=rng).bfloat16()
    if kind=='hostile':
        x.fill_(1);weights.zero_();weights[:7,0]=4
        weights[7,[72,114,6]]=torch.tensor([2**24,1,-2**24]).float();weights[8,0]=.5
    if kind=='tie': weights.zero_()
    outputs=[]
    for model,gate in ((Old,oldgate),(New,gateclass)):
        own=model();own.sp_replicate_expert_input=kind!='a2a';g=gate();g.weight=weights;g._split2_key=None
        r=Route(torch.zeros(n),1.25,.73,1e-7,settings.norm)
        own.route=r;own.gate=g;r.is_torch_compile=kind=='compiled'
        if kind=='mapped':r.local_expert_mapping=torch.arange(n,dtype=torch.int32)%32
        if kind=='padded':r.num_valid_tokens=torch.tensor([rows-2],dtype=torch.int32);r.sol_topk_out=torch.full((rows+1,8),-102,dtype=torch.int32)
        if kind=='hook':r.register_forward_pre_hook(lambda module,args:hooks.append(args[1].clone()))
        if kind=='gate_hook':g.register_forward_pre_hook(lambda *a:None)
        if kind=='custom_gate':g.forward_split2=g.forward_split2
        if kind=='custom_op':r.forward=r.forward
        if kind=='pinned_method':r._forward_method=r.forward_jit
        if kind=='compiled_call':r._compiled_call_impl=r._call_impl
        if kind=='backward_hook':r.register_full_backward_hook(lambda *a:None)
        if kind=='global_hook':
            hook=nn.modules.module.register_module_forward_pre_hook(lambda module,args:hooks.append(args[1].clone()) if isinstance(module,Route) else None)
        mm.clear();launches.clear();returns.clear();hooks.clear();captures.clear();responses.clear()
        with contextlib.ExitStack() as stack:
            if kind=='mode':
                class Pass(torch.overrides.TorchFunctionMode):
                    def __torch_function__(self,func,types,args=(),kwargs=None):
                        return func(*args,**(kwargs or {}))
                stack.enter_context(Pass())
            if kind=='dispatch_mode':
                from torch.utils._python_dispatch import TorchDispatchMode
                class Pass(TorchDispatchMode):
                    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                        return func(*args,**(kwargs or {}))
                stack.enter_context(Pass())
            # Check the normal inference-mode forward with persistent weights created at model setup.
            with torch.inference_mode():
                y=own._forward_experts(x)
        if kind=='global_hook':hook.remove()
        assert len(mm)==2 and len(launches)==1
        outputs.append((mm.copy(),launches[0],y,responses[-1],hooks.copy(),returns.copy(),captures.copy()))
    a,b=outputs
    on=kind in {'normal','non_sp','softmax','tie','hostile','mapped','padded','capturer'}
    assert a[1][0]=='stock' and b[1][0]==('split' if on else 'stock'),(kind,a[1][0],b[1][0])
    assert a[0]==b[0] and torch.equal(a[1][1],b[1][1]) and torch.equal(a[2],b[2])
    for i in (3,4,5,6):
        assert len(a[i])==len(b[i]) and all(torch.equal(p,q) for p,q in zip(a[i],b[i]))
    if kind in ('hook','global_hook','a2a'):
        section=5 if kind=='a2a' else 4
        assert len(b[section])==1 and torch.equal(b[section][0],a[1][1])
    return dict(kind=kind,shape=[rows,k,n],split=on,logits=rows*n,ids=rows*8,native_mm=2)
if __name__=='__main__':
    rng=torch.Generator().manual_seed(254);torch.set_num_threads(4)
    with torch.no_grad(),patch.dict(sys.modules,{'sglang.kernels':pkg,modjit.__name__:modjit}),patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True):
        cases=[pair(2048,8192,256,'normal')]
        for case in ('non_sp','softmax','tie','hostile','mapped','padded','capturer','a2a','alternative','forced','failed_build','trace','debug','hook','global_hook','gate_hook','custom_gate','custom_op','pinned_method','compiled','mode','dispatch_mode','backward_hook','compiled_call'):
            cases.append(pair(35,128,256,case))
    # The published loader raises RuntimeError for native build failures. The
    # optional module caches that denial per geometry before any target launch.
    import functools
    util='python/sglang/kernels/jit/utils/common.py'
    cache=functions(old(util),['cache_once'],{'functools':functools})['cache_once']
    build_calls=[]
    def rejected(*args,**kw):
        build_calls.append(args)
        raise RuntimeError("synthetic compile failure")
    import logging
    split=cache(functions(get(jit),['_split_module'],dict(load_jit=rejected,
          logging=logging,make_cpp_args=lambda *a:tuple(map(str,a)),is_arch_support_pdl=lambda:True))['_split_module'])
    for w in (1,1,4,4): assert split(8,8,0,w) is None
    assert len(build_calls)==2
    print(json.dumps(dict(cases=cases,compile_denials=4,distinct_failed_builds=len(build_calls),oracle='native CPU FP32 GEMMs over exact BF16 inputs; equal high and low reduction order; CUDA route launch replaced by common CPU semantics',host_source_sha=hashlib.sha256(get(models).encode()).hexdigest()),indent=2))
