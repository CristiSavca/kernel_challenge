"""Compare the V68 and paired router through Muse's owner-row MoE entry and native routing."""
import ast
import contextlib
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch, PropertyMock
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
stage=Path(sys.argv[1])
model_path='python/sglang/srt/models/muse_spark.py'
paired_path='python/sglang/kernels/ops/moe/muse_spark_paired_router.py'
route_path='python/sglang/kernels/ops/moe/muse_spark_routing.py'
base=json.loads((HERE.parent/'v68_grouped_decode_verify/validation.json').read_text())['trees']['v68']
read=lambda tree,p:subprocess.check_output(['git','-C',stage,'show',tree+':'+p]).decode()
m_before=read(base,model_path);m_after=(stage/model_path).read_text()
source=(stage/paired_path).read_text();route=read(base,route_path)

def defs(s,names,env,cls=None):
    nodes=ast.parse(s).body
    if cls:nodes=next(n.body for n in nodes if getattr(n,'name',None)==cls)
    m=[n for n in nodes if getattr(n,'name',None) in names];assert len(m)==len(names)
    for n in m:n.decorator_list=[]
    exec('from __future__ import annotations\n'+ast.unparse(ast.Module(m,[])),env)
    return {n:env[n] for n in names}
stats=Counter()
class T:
    def __getattr__(self,name):return getattr(torch,name)
    def mm(self,x,w,out_dtype):
        stats['legacy_mm']+=1;assert out_dtype==torch.float32
        return x.float()@w.float()
t=T()
before=type('Before',(),defs(m_before,['_build_split2','forward_split2'],{'torch':t},'MuseSpark1xRouter'))
after=type('After',(),defs(m_after,['_build_split2','forward_split2'],{'torch':t},'MuseSpark1xRouter'))

# Checked NumPy implementation of individual source program bodies. Inputs are
# exact BF16 values widened to store them on this CPU. It models FP32 accumulation
# and makes no assertion about NVIDIA tensor-core internal rounding.
a=next(HERE.parent.glob('v62_*/cpu_check.py'));ns={}
body=ast.parse(a.read_text()).body
i=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and n.targets[0].id=='parser')
exec(compile(ast.Module(body[:i],[]),str(a),'exec'),ns)
tl,Vec,Ptr=ns['tl'],ns['Vector'],ns['Pointer']
def dot(x,w,acc=0):
    return (np.einsum('ik,kj->ij',x,w,optimize=False,dtype='f4')+acc).view(Vec)
tl.dot=dot
def store(ptr,v,mask):
    offsets,live,chosen=ns['bounds'](ptr,mask)
    assert np.isnan(ptr.data[chosen]).all()
    ptr.data[chosen]=np.broadcast_to(v,offsets.shape)[live]
tl.store=store
dev=defs(source,['_paired_router'],{'tl':tl})['_paired_router']
def runbody(x,h,l,subgrid=None):
    m,k=x.shape;n=h.shape[1];y=np.full((m,n),np.nan,np.float32)
    args=[Ptr(z.float().numpy()) for z in (x,h,l)]
    grid=subgrid or [(a,b) for a in range((m+63)//64) for b in range(n//32)]
    for a,b in grid:
        ns['coords']=(a,b,0)
        dev(*args,Ptr(y),m,k,n,64,32,64)
        stats['checked_device_ctas']+=1
    return torch.from_numpy(y)

control=NS(host_grid=[],use_exact_body=False)
class Launch:
    def __getitem__(self,grid):
        def run(x,h,l,out,m,k,n,bm,bn,bk,**kw):
            assert grid==((m+63)//64,n//32) and (bm,bn,bk)==(64,32,64)
            assert x.shape==(m,k) and h.shape==l.shape==(k,n) and out.shape==(m,n)
            control.host_grid.append(grid);stats['new_launch']+=1
            if control.use_exact_body:out.copy_(runbody(x,h,l))
            else:out.copy_(x.float()@h.float()+x.float()@l.float())
        return run
tr=NS(cdiv=lambda a,b:(a+b-1)//b)
env={'torch':torch,'triton':tr,'_paired_router':Launch(),'get_exec':lambda:NS(deterministic=NS(enable_deterministic_inference=False)),
     'envs':NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:False))}
new_op=defs(source,['paired_router'],env)['paired_router']
module=ModuleType('sglang.kernels.ops.moe.muse_spark_paired_router');module.paired_router=new_op
renv={'torch':torch,'fast_topk':torch.topk,'envs':NS(SGLANG_USE_CUSTOM_ROUTER_SUM=NS(get=lambda:False))}
defs(route,['pack_topk_ids_native'],renv)
Router=type('Router',(),defs(route,['_finalize','forward_native'],renv,'FusedMuseSparkRouting'))
ro=Router();ro.router_output_multiplier=1.;ro.route_scale=1.;ro.router_score_division_eps=1e-8
ro.num_valid_tokens=ro.local_expert_mapping=None
state=NS(sp_active=True);trace=[]
menv=dict(torch=torch,_is_cpu=False,get_forward=lambda:state,layernorm_sp=NS(sp_num_tokens=lambda:16384),
          get_parallel=lambda:NS(tp_group=NS(ca_comm=None)),_use_split_bf16_router=lambda m:m>256,
          bf16_gemm_dispatch=lambda x,*_:x,contextlib=contextlib,get_global_experts_capturer=lambda:None)
class M:
    _forward_experts=defs(m_after,['_forward_experts'],menv,'MuseSpark1xMoE')['_forward_experts']
    sp_replicate_expert_input=router_in_fp32=nvfp4_checkpoint=True
    experts=NS(use_w4a16=True)
    tp_size=8;top_k=8;pre_expert_proj=NS(weight=None)
    def _routing_op(self):return ro.forward_native
    def _forward_sp_experts(self,x,scores,ids,logits):
        trace.append((scores.clone(),ids.clone(),logits.clone()))
        # Deterministic synthetic expert return isolates the selected route payload.
        return (scores*(ids.float()+1.)).sum(1)
op=M()
def makegate(cls,w):
    g=cls();g.weight=w;g._split2_key=None;return g

torch.set_num_threads(4)
rgen=torch.Generator().manual_seed(252)
rows=[1024,1025,2048]
cases=[]
with patch.dict(sys.modules,{module.__name__:module}),patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True),patch.object(torch.cuda,'get_device_properties',return_value=NS(major=9,multi_processor_count=132)):
    for idx,(m,k,n) in enumerate([(1024,128,256),(1025,128,256),(2048,8192,256),(1025,130,256),(1031,128,256)]):
        w=(torch.randint(-4,5,(n,k),generator=rgen).float()+torch.randint(-2,3,(n,k),generator=rgen).float()/1024)
        x=(torch.randint(-3,4,(m,k),generator=rgen).bfloat16()/8).contiguous()
        if idx==4:
            w.fill_(-2.);w[:7]=4.;w[7].fill_(1.0009765625);w[8].fill_(0.9990234375)
            x.zero_();x[:,0]=1.
        ro.expert_bias=torch.linspace(-.001,.001,n)
        if idx==4:ro.expert_bias.zero_()
        g0=makegate(before,w);g1=makegate(after,w)
        op.gate=g0;p=op._forward_experts(x);prev=trace[-1]
        ctl=stats.copy();control.use_exact_body=k==128
        op.gate=g1;q=op._forward_experts(x);cur=trace[-1]
        launches=stats['new_launch']-ctl['new_launch'];fallback=stats['legacy_mm']-ctl['legacy_mm']
        eligible=(m+63)//64*(n//32)>=132 and k%64==0
        assert (launches,fallback)==((1,0) if eligible else (0,2))
        for a,b in zip((p,*prev),(q,*cur)):assert torch.equal(a,b),(idx,(a!=b).sum().item())
        packed=renv['pack_topk_ids_native'](cur[1],cur[0])
        assert torch.equal(packed>>16,cur[1])
        if idx==4:assert (cur[1]==7).any(1).all() and not (cur[1]==8).any()
        if k==8192:
            # The normal physical-shape host launch above exercises all rows.
            # Check four additional complete production CTAs, including endpoint columns.
            sample=[(0,0),(0,7),(31,0),(31,7)];y=runbody(x,g1._split2_hi,g1._split2_lo,sample)
            for a,b in sample:
                sl=(slice(64*a,64*a+64),slice(32*b,32*b+32))
                assert torch.equal(cur[2][sl],y[sl])
        cases.append(dict(rows=m,features=k,experts=n,paired=bool(launches),logits=m*n,
                          selected_experts=m*8,checked_return=m))
    # Opposite orders of finite *exact* BF16 products can straddle top-k:
    # row7 2^24 + 1 - 2^24 vs row8 .5. Distinct legal FP32 summations
    # give 0 and 1. This is an order sensitivity witness, not an H200 mismatch.
    w=torch.zeros(256,128);w[:7,0]=4.;w[7,[0,1,64]]=torch.tensor([2.**24,1.,-2.**24]);w[8,0]=.5
    exact_x=torch.ones(64,128,dtype=torch.bfloat16);g=makegate(after,w);g._build_split2()
    generated=runbody(exact_x,g._split2_hi,g._split2_lo)
    assert generated[0,7].item()==0.
    regroup=torch.cat((generated[:,:7],torch.ones(64,1),generated[:,8:]),1)
    ro.expert_bias=torch.zeros(256)
    _,cut0,_=ro.forward_native(None,generated,8,True,True)
    _,cut1,_=ro.forward_native(None,regroup,8,True,True)
    assert (cut0==8).any(1).all() and (cut1==7).any(1).all()
    # Search for a genuine host arithmetic difference, then position its logit
    # on the eighth-expert boundary and run the unmodified routing reference.
    probes = torch.zeros(256,128)
    tested=[]
    for j in range(256):
        ix = torch.randperm(128,generator=rgen)[:3].tolist()
        probes[j,ix]=torch.tensor([2.**24,1.,-2.**24])
        tested.append(ix)
    probe=makegate(after,probes);probe._build_split2()
    cb=exact_x.float()@probe._split2_hi.float()+exact_x.float()@probe._split2_lo.float()
    em=runbody(exact_x,probe._split2_hi,probe._split2_lo)
    mismatch=(em[0]!=cb[0]).nonzero().flatten().tolist()
    host_sensitivity={'tested_finite_logits':len(tested),'different_from_CPU_BLAS':len(mismatch)}
    if mismatch:
        j=mismatch[0];tied=torch.zeros_like(probes);tied[:7,0]=4.;tied[7]=probes[j]
        tied[8,0]=(em[0,j]+cb[0,j])/2
        actual=makegate(after,tied);actual._build_split2()
        a=exact_x.float()@actual._split2_hi.float()+exact_x.float()@actual._split2_lo.float()
        b=runbody(exact_x,actual._split2_hi,actual._split2_lo)
        _,idsa,_=ro.forward_native(None,a,8,True,True);_,idsb,_=ro.forward_native(None,b,8,True,True)
        assert not torch.equal(idsa,idsb)
        host_sensitivity.update(positions=tested[j],native_cpu_score=a[0,7].item(),paired_emulator_score=b[0,7].item(),
                                intervening_score=a[0,8].item(),example_top8=[idsa[0].tolist(),idsb[0].tolist()])
manifest=json.loads((HERE/'validation.json').read_bytes())
print(json.dumps(dict(candidate_sha256=manifest['patch_sha256'],tree=manifest['tree'],host_order_sensitivity=host_sensitivity,sources={model_path:sha256(m_after.encode()).hexdigest(),paired_path:sha256(source.encode()).hexdigest(),route_path:sha256(route.encode()).hexdigest()},
    normal_SP_calls=cases,stats=stats,shape_assumption='SM90 with 132 SMs, TP8, BF16 input; public V68 reports TP8 and static Breakable prefill.',physical_H200=False,
    limitation='The full-size paired host launch uses CPU GEMM for its result plus four independently executed physical-shape CTAs. Synthetic routing uses the pinned native Torch reference. These results do not establish that the original H200 cuBLAS and Triton choose identical FP32 summation order.',
    order_witness={'BF16_products':[16777216,1,-16777216],'finite_alternate_FP32_results':[0,1],'competitor_logit':.5,'top8_eighth_ids':[8,7]}),indent=2))
