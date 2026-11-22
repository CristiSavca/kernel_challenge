"""Paired execution of the V64 and V178 attention/writer code with real CPU PyTorch tensors."""
import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as S
from unittest.mock import patch, PropertyMock

import torch

V64 = "b5a37c2923f8d1ab24c1c5cf7920a21e2c086774"
SR = "python/sglang/srt/"
BACK = SR+"layers/attention/flashattention_backend.py"
HELP = SR+"layers/attention/direct_prompt_kv.py"
POOL = SR+"mem_cache/memory_pool.py"
SWA = SR+"mem_cache/swa_memory_pool.py"
PAGED = SR+"mem_cache/layout/paged_view.py"
stage = Path(sys.argv[1])
torch.set_num_threads(1)


@lru_cache(None)
def src(path, old=True):
    if old:
        return subprocess.check_output(["git","-C",str(stage),"show",V64+":"+path],text=True)
    return (stage/path).read_text()


def add(ns, path, names, cls=None, old=True):
    tree = ast.parse(src(path,old))
    body = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body if cls else tree.body
    for name in names:
        node = next(n for n in body if n.name==name) if all(hasattr(n,'name') for n in body) else next(
            n for n in body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec("from __future__ import annotations\n"+ast.unparse(node),ns)


class Mode:
    def is_target_verify(self): return False
Mode.EXTEND=Mode()
ATTN=S(ENCODER_ONLY="enc",DECODER_BIDIRECTIONAL="bi")


def mk(old, full, sliding, q, slots, swas, width, k_scale, v_scale):
    calls, writes=[],[]
    def fa(**kw):
        calls.append(kw)
        out=kw.get("out")
        result=kw["q"].bfloat16() if out is None else out
        return (result, torch.zeros((result.shape[1],result.shape[0]))) if kw.get("return_softmax_lse") else result
    ns=dict(torch=torch,ForwardMode=Mode,AttentionType=ATTN,is_cp_active=lambda fb:False,
            flash_attn_with_kvcache=fa,dataclass=dataclass,maybe_detect_oob=lambda *a:None)
    add(ns,PAGED,["paged_view","paged_kv_view"])
    add(ns,POOL,["unwrap_write_loc"])
    exec("from __future__ import annotations\n"+ast.unparse(next(n for n in ast.parse(src(POOL)).body if getattr(n,"name",None)=="KVWriteLoc")),ns)
    add(ns,POOL,["set_kv_buffer"],"MHATokenToKVPool")
    impl=ns["set_kv_buffer"]
    class Owner:
        set_kv_buffer=impl
        def _check_physical_write_loc(self,*a): pass
        def _store_kv_layer(self,layer_idx,loc,k,v):
            self.data[0].view(torch.uint8)[loc.long()]=k.view(torch.uint8)
            self.data[1].view(torch.uint8)[loc.long()]=v.view(torch.uint8)
            writes.append((loc.clone(),k.clone(),v.clone()))
        def get_kv_buffer(self,layer_id): return self.data
    add(ns,SWA,["set_kv_buffer","get_kv_buffer"],"SWAKVPool")
    class Hybrid:
        swa_req_ring_size=None
        set_kv_buffer=ns["set_kv_buffer"]
        get_kv_buffer=ns["get_kv_buffer"]
        def _wait_for_layer(self,layer_id): pass
    ns.update(MHATokenToKVPool=Owner,SWAKVPool=Hybrid,MLATokenToKVPool=type("MLA",(),{}))
    if not old:
        add(ns,HELP,["supports_direct_prompt_pool","prepare_direct_prompt_kv","complete_direct_prompt_kv"],old=False)
    methods=["forward_extend","get_paged_mha_kv_cache","prepare_paged_mha_query","_mxfp8_sf_kwargs"]
    add(ns,BACK,methods,"FlashAttentionBackend",old=old)
    backend=type("Backend",(),{n:ns[n] for n in methods})()
    owners=[]
    for pair in (full,sliding):
        o=Owner()
        o.__dict__.update(data=tuple(x.clone() for x in pair),size=pair[0].shape[0]-1,page_size=1,
                          is_quantized_kv_cache=False,kv_cache_layout="nhd",requires_physical_write_loc=False,
                          dtype=torch.float8_e4m3fn,store_dtype=torch.uint8,start_layer=0,use_hnd=False)
        owners.append(o)
    pool=Hybrid()
    pool.__dict__.update(full_kv_pool=owners[0],swa_kv_pool=owners[1],layers_mapping={0:(0,width>=0)})
    backend.__dict__.update(_direct_fp8_prompt=True,token_to_kv_pool=pool,topk=1,page_size=1,fa_impl_ver=3,
        num_splits=0,use_mla=False,fa_skip_kv_cache=False,kv_cache_is_mxfp8=False,kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_dtype_str="fp8_e4m3",local_attn_builder=None,use_sliding_window_kv_pool=True,
        kv_index_translator=S(is_translating=False),full_cg_prefill_metadata=None)
    assert old or ns["supports_direct_prompt_pool"](pool,backend.kv_cache_dtype)
    return backend,pool,calls,writes


def run(name,n,p,w,scaled,expect,device_cuda=True):
    h,d,qh=2,16,4
    torch.manual_seed(5)
    total=p+n
    cap=total+11
    full,slide=[tuple(torch.randn(cap,h,d).to(torch.float8_e4m3fn) for _ in range(2)) for _ in range(2)]
    t1,t2=[(torch.randperm(cap-1)[:total]+1).int()[None] for _ in range(2)]
    k0=torch.randn(n,h,d);v0=torch.randn(n,h,d)
    q=torch.randn(n,qh*d).bfloat16();ks,vs=(torch.tensor([.75]),torch.tensor([.625])) if scaled else (None,None)
    layer=S(layer_id=0,tp_v_head_num=h,tp_k_head_num=h,tp_q_head_num=qh,v_head_dim=d,head_dim=d,
            is_cross_attention=False,attn_type="decoder",sliding_window_size=w,k_scale=ks,v_scale=vs,logit_cap=2.,scaling=.25)
    cuq=torch.tensor([0,n],dtype=torch.int32);lens=torch.tensor([total],dtype=torch.int32)
    tab=t2 if w>=0 else t1
    fb=S(seq_lens_cpu=lens,forward_mode=Mode.EXTEND,batch_size=1,dcp_kv_mask=None,extend_seq_lens_cpu=[n],extend_prefix_lens_cpu=[p],
         out_cache_loc=t1[0,p:].long(),out_cache_loc_is_physical=False)
    if name=="pad": fb.extend_seq_lens_cpu=[n-1]
    if name=="missing": fb.extend_prefix_lens_cpu=None
    if name=="batch": fb.batch_size=2
    if name=="no_mirror": fb.seq_lens_cpu=None
    multi=[]
    with torch.set_grad_enabled(name=="autograd"), patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=device_cuda), patch.object(
        torch.cuda,"is_current_stream_capturing",return_value=name=="capture"),patch.object(
        torch.compiler,"is_compiling",return_value=name=="compile"),patch.object(
        torch,"are_deterministic_algorithms_enabled",return_value=name=="deterministic"):
        for old in (True,False):
            b,pool,calls,writes=mk(old,full,slide,q,t1,t2,w,ks,vs)
            md=S(cache_seqlens_int32=lens,max_seq_len_k=total+int(name=="static_bound"),max_seq_len_q=n,cu_seqlens_q=cuq,
                 cu_seqlens_k=lens.new_tensor([0,total]),page_table=t1,swa_page_table=t2,
                 image_token_ranges=None,swa_spec_metadata=None,swa_out_cache_loc=t2[0,p:].long())
            b.forward_metadata=md
            if name=="full_graph": b.full_cg_prefill_metadata=md
            kv=torch.stack((k0,v0),dim=1).bfloat16()
            k,v=kv.unbind(1)  # outer-replayed QKV split can have pitched token rows
            b.forward_extend(q,k,v,layer,fb,sinks=q if name=="sinks" else None,return_lse=name=="lse")
            assert len(calls)==len(writes)==1
            multi.append((calls[0],writes[0],[*pool.full_kv_pool.data,*pool.swa_kv_pool.data],k,v))
    (a,aw,ad,ak,av),(b,bw,bd,bk,bv)=multi
    assert a["page_table"] is tab
    assert (b["page_table"] is None)==expect,(name,expect,b["page_table"] is None)
    assert torch.equal(ak,bk) and torch.equal(av,bv)
    for x,y in zip(ad,bd,strict=True): assert torch.equal(x.view(torch.uint8),y.view(torch.uint8))
    for x,y in zip(aw,bw,strict=True): assert torch.equal(x.view(torch.uint8),y.view(torch.uint8))
    assert torch.equal(a["q"].view(torch.uint8),b["q"].view(torch.uint8))
    assert a["cu_seqlens_q"] is b["cu_seqlens_q"] and a["out"] is b["out"]
    for key in ("softmax_scale","softcap","causal","window_size","max_seqlen_q","num_splits","return_softmax_lse","ver"):
        assert a[key]==b[key]
    if scaled:
        assert torch.equal(a["k_descale"],b["k_descale"]) and torch.equal(a["v_descale"],b["v_descale"])
    checked=0
    if expect:
        cut=max(0,p-w) if w>=0 else 0
        assert int(b["cache_seqlens"][0])==total-cut
        pk,pv=b["k_cache"][0],b["v_cache"][0]
        rk,rv=[x[tab[0,cut:].long()] for x in (a["k_cache"][:,0].view(torch.uint8),a["v_cache"][:,0].view(torch.uint8))]
        assert torch.equal(pk.view(torch.uint8),rk) and torch.equal(pv.view(torch.uint8),rv)
        for qi in (0,1,n//2,n-1):
            hi=p+qi+1
            lo=max(0,hi-1-w) if w>=0 else 0
            for head in (0,qh-1):
                def output(K,V):
                    Q=b["q"][qi,head].float()
                    sc=K[:,head//2].float().matmul(Q)*layer.scaling*(float(ks) if scaled else 1.)
                    sc=torch.tanh(sc/layer.logit_cap)*layer.logit_cap
                    return sc.softmax(0).matmul(V[:,head//2].float()*(float(vs) if scaled else 1.))
                nidx=tab[0,lo:hi].long()
                assert torch.equal(output(pk[lo-cut:hi-cut],pv[lo-cut:hi-cut]),output(
                    a["k_cache"][:,0].view(torch.uint8)[nidx].view(torch.float8_e4m3fn),
                    a["v_cache"][:,0].view(torch.uint8)[nidx].view(torch.float8_e4m3fn)))
                checked+=d
    return dict(case=name,scale=scaled,query=n,prefix=p,window=w,dense=expect,
                checked_cache_bytes=sum(x.numel() for x in ad),output_scalars=checked)


def startup():
    from copy import deepcopy
    ns=dict(torch=torch,ForwardMode=Mode,AttentionType=ATTN)
    class Owner:
        pass
    class Hybrid:
        swa_req_ring_size=None
    ns.update(MHATokenToKVPool=Owner,SWAKVPool=Hybrid)
    add(ns,HELP,["supports_direct_prompt_pool"],old=False)
    tree=ast.parse(src(BACK,False))
    assign=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(getattr(t,"attr",None)=="_direct_fp8_prompt" for t in n.targets))
    expression=compile(ast.Expression(assign.value),"<production initialization>","eval")
    platform=S(is_sm90=True);parallel=S(enable_prefill_cp=False,attn_dcp_size=1,attn_dp_enabled=False)
    ns.update(get_platform=lambda:platform,get_parallel=lambda:parallel,
              envs=S(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=S(get=lambda:False)),deterministic=False)
    pool=Hybrid();owners=[Owner(),Owner()]
    for obj in owners:
        obj.__dict__.update(dtype=torch.float8_e4m3fn,kv_cache_layout="nhd",is_quantized_kv_cache=False,requires_physical_write_loc=False)
    pool.full_kv_pool,pool.swa_kv_pool=owners
    b=S(fa_impl_ver=3,page_size=1,kv_cache_dtype=torch.float8_e4m3fn,kv_cache_dtype_str="fp8_e4m3",
        use_mla=False,_kv_shard_pool=None,is_draft_runner=False,kv_index_translator=S(is_translating=False),
        is_prefill_aware_swa=False,local_attn_builder=None,is_encoder_decoder=False,token_to_kv_pool=pool)
    ns.update(self=b)
    assert eval(expression,ns)
    rejected=0
    for obj,name,val in [(platform,"is_sm90",False),(b,"fa_impl_ver",4),(b,"page_size",160),(b,"is_draft_runner",True),
                         (b,"use_mla",True),(b,"kv_cache_dtype",torch.bfloat16),(b,"kv_cache_dtype_str","auto"),
                         (b,"_kv_shard_pool",object()),(b.kv_index_translator,"is_translating",True),
                         (b,"is_prefill_aware_swa",True),(b,"local_attn_builder",object()),(b,"is_encoder_decoder",True),
                         (parallel,"attn_dcp_size",2),(parallel,"attn_dp_enabled",True),(parallel,"enable_prefill_cp",True),
                         (pool,"swa_req_ring_size",1024),(pool,"full_kv_pool",object()),(owners[0],"is_quantized_kv_cache",True),
                         (owners[1],"requires_physical_write_loc",True),(owners[0],"kv_cache_layout","hnd")]:
        with patch.object(obj,name,val):
            assert not eval(expression,ns),(name,val)
        rejected+=1
    ns["deterministic"]=True
    assert not eval(expression,ns)
    return dict(baseline="Hopper FP8 hybrid full/SWA pool",passing=1,resolver_and_pool_rejections=rejected+1)


if __name__=="__main__":
    cases=[]
    for n,p,w in ((4096,0,-1),(4113,0,2047),(4096,37,-1),(4101,5051,2047),(4096,23,2047),(4096,5001,0),(4096,37,16)):
        for scaled in (False,True):
            cases.append(run("valid",n,p,w,scaled,True))
    for key in ("pad","missing","batch","compile","capture","deterministic","full_graph","sinks","lse","cpu","short","autograd","static_bound","no_mirror"):
        cases.append(run(key,378 if key=="short" else 4096,31,2047,True,False,device_cuda=key!="cpu"))
    print(json.dumps(dict(torch=torch.__version__,startup=startup(),paired_cases=cases,reference_cache_bytes=sum(x["checked_cache_bytes"] for x in cases),
          reference_output_scalars=sum(x["output_scalars"] for x in cases),actual_fa3_gpu=False,
          source_hashes={p:hashlib.sha256(src(p,False).encode()).hexdigest() for p in (BACK,HELP)}),indent=2))
