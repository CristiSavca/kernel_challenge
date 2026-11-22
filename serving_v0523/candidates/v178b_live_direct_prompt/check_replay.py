"""Replay pinned prefill scope, Radix eager callback, MHA/SWA writes and FA3 dispatch on CPU."""
import ast
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as S
from unittest.mock import patch,PropertyMock

import torch

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/"v178_direct_fp8_prompt_kv"
spec=importlib.util.spec_from_file_location("case178",PARENT/"check.py")
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
torch.set_num_threads(1)
V178="8f5a57c5ac575321b442c78e6c19e7f4a83c8eae"
G=shared.SR+"layers/attention/graph_utils.py"
R=shared.SR+"layers/radix_attention.py"
P=shared.SR+"model_executor/runner_utils/prefill_graph.py"
mode={}
original_add=shared.add
ctx={}
exec(shared.src(P),ctx)
graph_helpers={k:ctx[k] for k in ("get_prefill_raw_num_tokens","is_in_full_prefill_graph","is_in_prefill_graph_scope")}


def inject(ns,*args,**kw):
    ns.update(graph_helpers)
    original_add(ns,*args,**kw)


def method(path,fn,cls=None):
    tree=ast.parse(shared.src(path))
    body=tree.body if not cls else next(n.body for n in tree.body if getattr(n,"name",None)==cls)
    n=next(n for n in body if getattr(n,"name",None)==fn)
    exec("from __future__ import annotations\n"+ast.unparse(n),mode)


mode.update(torch=torch,contextmanager=contextmanager,eager_on_graph=lambda f:f,
            attention_query_tokens=lambda fb,n:n,attention_output_tokens=lambda fb,n:n,**graph_helpers)
for name in ("attention_input_scope","_zero_padded_tokens"):method(G,name)
method(R,"_eager_attention","RadixAttention")


def test(label, n, p, left, pad, zeros, want):
    torch.manual_seed(n+p)
    cap=n+p+29; h=2;d=16;qh=4
    full,sliding=[tuple(torch.randn(cap,h,d).to(torch.float8_e4m3fn) for _ in range(2)) for _ in range(2)]
    ft,st=[(torch.randperm(cap-1)[:p+n]+1).int().unsqueeze(0) for _ in range(2)]
    if zeros:ft.fill_(0);st.fill_(0)
    stale=123
    ft_pad=torch.cat((ft[0,p:].long(),torch.zeros(pad,dtype=torch.long)))
    st_pad=torch.cat((st[0,p:].long(),torch.zeros(pad,dtype=torch.long)))
    Q=torch.randn(n+pad,qh*d).bfloat16()
    kv=torch.randn(n+pad,2,h,d).bfloat16()
    results=[]
    ks=torch.tensor([.25]);vs=torch.tensor([.5])
    lens=torch.tensor([p+n],dtype=torch.int32);cuq=lens.new_tensor([0,n]);cuk=lens.new_tensor([0,p+n])
    for old in (True,False):
        actual=S(layer_id=0,tp_v_head_num=h,tp_k_head_num=h,tp_q_head_num=qh,v_head_dim=d,head_dim=d,
                 is_cross_attention=False,attn_type="decoder",sliding_window_size=left,k_scale=ks,v_scale=vs,logit_cap=0.,scaling=.25)
        b,pool,calls,writes=shared.mk(old,full,sliding,Q,ft,st,left,ks,vs)
        b.forward=b.forward_extend
        md=S(cache_seqlens_int32=lens,cu_seqlens_k=cuk,cu_seqlens_q=cuq,page_table=ft,swa_page_table=st,
             max_seq_len_q=n,max_seq_len_k=n+p,image_token_ranges=None,swa_spec_metadata=None,swa_out_cache_loc=st_pad)
        b.forward_metadata=md
        fb=S(seq_lens_cpu=lens,forward_mode=shared.Mode.EXTEND,batch_size=1,dcp_kv_mask=None,extend_prefix_lens_cpu=[p],
             extend_seq_lens_cpu=[n],out_cache_loc=ft_pad,out_cache_loc_is_physical=False,positions=torch.arange(n+pad),
             global_num_token_non_padded_cpu=n,mha_return_lse=False,_attn_output=None,req_pool_indices_cpu=None)
        if label=="eager":fb.req_pool_indices_cpu=lens.new_tensor([3])
        output=torch.full_like(Q,stale)
        mode.update(get_forward_batch=lambda:fb,get_attn_backend=lambda:b)
        raw=None if label=="warmup" else n-1 if label=="wrong_raw" else n
        context=(nullcontext() if label.startswith("eager") else ctx["prefill_graph_scope"](full_graph=label=="full",raw_num_tokens=raw))
        with context,torch.inference_mode(),patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True),patch.object(torch.cuda,"is_current_stream_capturing",return_value=False):
            k,v=kv.clone().unbind(1)
            mode["_eager_attention"](actual,Q,k,v,output,True)
        assert fb.out_cache_loc is ft_pad and fb._attn_output is None and fb.positions.shape[0]==n+pad
        assert len(writes)==len(calls)==1 and writes[0][0].numel()==n
        fa=calls[0];assert fa["q"].shape==(n,qh,d)
        assert fa["out"].shape==(n,qh,d) and fa["out"].data_ptr()==output.data_ptr()
        if label!="warmup" and not label.startswith("eager"):
            assert not torch.count_nonzero(output[raw:])
        results.append((fa,writes,tuple(pool.full_kv_pool.data)+tuple(pool.swa_kv_pool.data)))
    (a,aw,ac),(b,bw,bc)=results
    assert (b["page_table"] is None)==want,label
    for x,y in zip(ac,bc,strict=True):
        assert torch.equal(x.view(torch.uint8),y.view(torch.uint8))
    assert torch.equal(aw[0][0],bw[0][0])
    tab=st if left>=0 else ft
    assert a["page_table"] is tab
    if want:
        drop=max(0,p-left) if left>=0 else 0
        assert int(b["cache_seqlens"][0])==p+n-drop
        for data in ("k_cache","v_cache"):
            assert torch.equal(b[data][0].view(torch.uint8),a[data][:,0].view(torch.uint8)[tab[0,drop:].long()])
    else:
        assert b["page_table"] is tab
    return dict(context=label,prefix=p,live_rows=n,padding=pad,window=left,physical_zero_slots=zeros,
                dense=want,all_physical_pool_bytes=sum(t.numel() for t in ac))


if __name__=="__main__":
    cases=[]
    with patch.object(shared,"add",side_effect=inject):
        for w in (-1,2047):
            for p in (0,5017):
                for label,zeros,want in (("warmup",True,False),("full",False,False),("eager_dummy",True,False),
                                          ("eager",False,True),("live",False,True),("wrong_raw",False,False)):
                    cases.append(test(label,4127,p,w,97,zeros,want))
    print(json.dumps(dict(real_code=[G,R,P,shared.BACK,shared.POOL,shared.SWA,shared.HELP],cases=cases,
              padded_callbacks=len(cases)*2,physical_pool_bytes=sum(x["all_physical_pool_bytes"] for x in cases),
              compiled_gpu=False,candidate_source_sha256=hashlib.sha256(shared.src(shared.HELP,False).encode()).hexdigest()),indent=2))
