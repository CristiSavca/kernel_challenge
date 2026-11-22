"""Run complete pinned/V95/V137 prefill methods and check native bottom-right causal inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
PARENT = "985b7b1e74040fed4cd673ba89324e4dc7f35c7e"
BACKEND = "python/sglang/srt/layers/attention/flashattention_backend.py"
KERNEL = "python/sglang/kernels/ops/attention/dense_prefill_kv.py"
PAGED = "python/sglang/srt/mem_cache/layout/paged_view.py"

def src(stage,path,tree=None):
    if tree: return subprocess.check_output(["git","-C",str(stage),"show",tree+":"+path],text=True)
    return (stage/path).read_text()

def method(raw,name):
    return next(node for node in ast.walk(ast.parse(raw)) if isinstance(node,ast.FunctionDef) and node.name==name)

class Mode:
    def is_target_verify(self): return False
EXTEND=Mode();Mode.EXTEND=EXTEND

class Launch:
    def __init__(self,ragged): self.ragged=ragged
    def __getitem__(self,grid):
        def go(*args,**kw):
            if self.ragged:
                k,v,table,cu,offsets,dk,dv,ks,vs,ts,batch=args
                offset=0
                for i in range(batch):
                    n=int(cu[i+1]-cu[i]);offsets[i]=offset
                    idx=table[i,:n].long();dk[offset:offset+n]=k[idx,0];dv[offset:offset+n]=v[idx,0]
                    dk[offset+n:offset+n+256]=0;dv[offset+n:offset+n+256]=0;offset+=n+256
                offsets[-1]=offset
            else:
                k,v,table,lens,dk,dv,*_=args
                for i,n in enumerate(lens.tolist()):
                    idx=table[i,:n].long();dk[i,:n]=k[idx,0];dv[i,:n]=v[idx,0]
        return go

def build(stage,tree):
    backend=src(stage,BACKEND,tree);kernel=src(stage,KERNEL,tree);calls=[]
    def fa(tag,**kw):
        calls.append((tag,kw))
        q=kw["q"]
        return (q,torch.zeros(q.shape[1],q.shape[0])) if kw.get("return_softmax_lse") else q
    env=dict(torch=torch,ForwardMode=Mode,AttentionType=NS(ENCODER_ONLY="enc",DECODER_BIDIRECTIONAL="bi"),
             is_cp_active=lambda fb:False,flash_attn_with_kvcache=lambda **kw:fa("paged",**kw),
             triton=NS(cdiv=lambda x,y:-(-x//y)),_pack_prefill_kv=Launch(False),_pack_ragged_prefill_kv=Launch(True))
    def add(raw,names):
        for n in names: exec("from __future__ import annotations\n"+ast.unparse(method(raw,n)),env)
    add(src(stage,PAGED,BASE),["paged_view","paged_kv_view"])
    add(kernel,["dense_prefill_kv"]+([] if tree else ["pack_ragged_prefill_kv"]))
    methods=["forward_extend","get_paged_mha_kv_cache","prepare_paged_mha_query","_mxfp8_sf_kwargs"]
    add(backend,methods)
    obj=type("Backend",(),{n:env[n] for n in methods})()
    obj.__dict__.update(_dense_fp8_prefill=True,topk=1,page_size=1,fa_impl_ver=3,num_splits=0,use_mla=False,
        fa_skip_kv_cache=False,kv_cache_is_mxfp8=False,kv_cache_dtype=torch.float8_e4m3fn,kv_cache_dtype_str="fp8_e4m3",
        local_attn_builder=None,use_sliding_window_kv_pool=False,kv_index_translator=NS(is_translating=False),
        _disable_scheduler_metadata_precompute=False,attn_cp_size=1,is_encoder_decoder=False,_kv_shard_pool=None,
        full_cg_prefill_metadata=None,is_draft_runner=False,num_kv_heads=2,head_dim=64)
    obj.flash_attn_varlen_func=lambda **kw: fa("varlen",**kw)
    if tree: return obj,calls,None
    marker=next(n for n in ast.walk(method(backend,"init_forward_metadata")) if isinstance(n,ast.If)
                and isinstance(n.body[0],ast.Assign) and any(isinstance(x,ast.Attribute) and isinstance(x.ctx,ast.Store) and x.attr=="ragged_prefill_k_rows"
                        for x in n.body[0].targets))
    return obj,calls,compile(ast.Module([marker],[]),"<metadata predicate>","exec")

def spot(kw,table,lens,cuq,physical_k,physical_v):
    k,v,cu=kw["k"],kw["v"],kw["cu_seqlens_k"]
    maximum=kw["max_seqlen_k"]; assert maximum==max(lens.tolist())
    cu_ref=torch.cat((torch.tensor([0],dtype=torch.int32),torch.cumsum(lens,0,dtype=torch.int32)))
    assert torch.equal(cu,cu_ref+256*torch.arange(len(cu),dtype=torch.int32))
    checked=0
    for i,used in enumerate(lens.tolist()):
        lo=int(cu[i]);qlo=int(cuq[i]);qcount=int(cuq[i+1])-qlo
        assert int(cu[i+1])-lo==used+256
        idx=table[i,:used].long()
        for packed,phys in ((k,physical_k),(v,physical_v)):
            assert torch.equal(packed[lo:lo+used].view(torch.uint8),phys.view(torch.uint8)[idx])
            assert torch.count_nonzero(packed[lo+used:lo+used+256].view(torch.uint8))==0
        for qi in sorted({0,qcount//2,qcount-1}) if qcount else ():
            # Full causal bottom-right: each query attends all K rows through prefix+qi.
            n=used-qcount+qi+1
            for h in (0,15):
                kh=h//8
                q=kw["q"][qlo+qi,h].float()
                def out(a,b):
                    score=torch.mv(a[:,kh].float(),q)*kw["softmax_scale"]*float(kw["k_descale"][i,kh])
                    return torch.mv(b[:,kh].float().T,score.softmax(0))*float(kw["v_descale"][i,kh])
                refk=physical_k.view(torch.uint8)[idx[:n]].view(physical_k.dtype)
                refv=physical_v.view(torch.uint8)[idx[:n]].view(physical_v.dtype)
                assert torch.equal(out(refk,refv),out(k[lo:lo+n],v[lo:lo+n]))
                checked+=64
    return checked

def check(stage):
    torch.manual_seed(371)
    fixture=[build(stage,PARENT),build(stage,None)]
    kv=[torch.randint(-3,4,(97,2,64)).to(torch.float8_e4m3fn) for _ in range(2)]
    cases=0;scalars=0;routes={}
    for name in ("normal","long_left","long_right","both_large","rectangle","zero","padded","no_mirror",
                 "capture","compiler","full_graph","dp","cp","sharded","draft","sharded_kv",
                 "short_q","swa","sinks","lse","raw_bf16"):
        ns=[74160,8192];qs=[770,513]
        if name=="long_left": ns=[74160,600,604,648];qs=[760,515,600,501]
        if name=="long_right": ns=ns[::-1];qs=qs[::-1]
        if name=="zero":ns+=[0];qs+=[0]
        if name=="both_large":ns=[79999,80100]
        if name=="rectangle":ns=[8000,5130]
        if name=="short_q":qs=[200,10]
        batch=len(ns);nq=sum(qs)
        q=torch.randint(-2,3,(nq+(name=="padded"),16,64)).to(torch.float8_e4m3fn)
        lens=torch.tensor(ns,dtype=torch.int32);cuq=torch.tensor([0,*torch.tensor(qs).cumsum(0).tolist()],dtype=torch.int32)
        cuk=torch.tensor([0,*lens.cumsum(0).tolist()],dtype=torch.int32)
        table=torch.randint(1,97,(batch,max(ns)),dtype=torch.int32)
        fb=NS(forward_mode=EXTEND,dcp_kv_mask=None,batch_size=batch,extend_num_tokens=nq)
        kscale=torch.tensor([.25,.5]);vscale=torch.tensor([.125,.75])
        layer=NS(tp_q_head_num=16,tp_k_head_num=2,tp_v_head_num=2,head_dim=64,v_head_dim=64,
                 is_cross_attention=False,attn_type="decoder",sliding_window_size=2047 if name=="swa" else -1,
                 layer_id=1,k_scale=kscale,v_scale=vscale,logit_cap=0.,scaling=.125)
        for obj,calls,marker in fixture:
            calls.clear()
            md=NS(cache_seqlens_int32=lens,max_seq_len_k=max(ns),max_seq_len_q=max(qs),cu_seqlens_q=cuq,
                  cu_seqlens_k=cuk,page_table=table,image_token_ranges=None,swa_spec_metadata=None,
                  swa_out_cache_loc=None,ragged_prefill_k_rows=None)
            obj._disable_scheduler_metadata_precompute=name=="dp";obj.attn_cp_size=2 if name=="cp" else 1
            obj._kv_shard_pool=object() if name=="sharded" else None;obj.is_draft_runner=name=="draft"
            obj.kv_index_translator=NS(is_translating=name=="sharded_kv");obj.full_cg_prefill_metadata=md if name=="full_graph" else None
            obj.forward_metadata=md;obj.token_to_kv_pool=NS(get_kv_buffer=lambda _:kv)
            obj.kv_cache_dtype_str="auto" if name=="raw_bf16" else "fp8_e4m3"
            if marker: exec(marker,dict(self=obj,forward_batch=fb,ForwardMode=Mode,seq_lens_cpu=None if name=="no_mirror" else lens,
                                       batch_size=batch,eager_max_k=max(ns),metadata=md))
            with patch.object(torch.compiler,"is_compiling",lambda:name=="compiler"), \
                    patch.object(torch.cuda,"is_current_stream_capturing",lambda:name=="capture"):
                obj.forward_extend(q.bfloat16() if name=="raw_bf16" else q,None,None,layer,fb,
                                   sinks=q if name=="sinks" else None,return_lse=name=="lse")
        p,c=fixture[0][1][0],fixture[1][1][0]
        assert p[0]=="paged"
        wanted=name in ("normal","long_left","long_right","zero")
        assert (c[0]=="varlen")==wanted,(name,c[0])
        if wanted:
            a,b=p[1],c[1]
            for field in ("softmax_scale","causal","window_size","softcap","num_splits","out","max_seqlen_q"):
                assert a[field]==b[field],field
            assert a["q"].data_ptr()==b["q"].data_ptr() and b["cu_seqlens_q"] is cuq and b["seqused_k"] is lens
            for field in ("v_descale","k_descale"):assert a[field].data_ptr()==b[field].data_ptr()
            scalars+=spot(b,table,lens,cuq,*kv)
        else:
            assert c[1].keys()==p[1].keys()
            assert (c[1]["page_table"] is None)==(p[1]["page_table"] is None)
            for field in ("page_table","cache_seqlens","cu_seqlens_q","cu_seqlens_k_new"):
                assert c[1][field] is p[1][field]
        routes[name]=c[0] if c[1].get("page_table") is not None or wanted else "dense_uniform"
        cases+=1
    return dict(cases=cases,routes=routes,reference_output_scalars=scalars,torch=torch.__version__,
                source_sha256={p:hashlib.sha256(src(stage,p).encode()).hexdigest() for p in (BACKEND,KERNEL)},
                actual_native_gpu_fa3=False)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",type=Path)
    print(json.dumps(check(parser.parse_args().stage),indent=2))
