"""Run the V64 and V143 FA3 wrapper on real FP8 CPU storage, with a recorded layout certificate."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import itertools
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
BACKEND = "python/sglang/srt/layers/attention/flashattention_backend.py"
SPAN = "python/sglang/srt/layers/attention/prefill_kv_span.py"
PAGED = "python/sglang/srt/mem_cache/layout/paged_view.py"


def pinned(repo, path):
    return subprocess.check_output(["git","-C",repo,"show",f"{PIN}:{path}"],text=True)


def node(src,name,kind=ast.FunctionDef):
    return next(n for n in ast.walk(ast.parse(src)) if isinstance(n,kind) and n.name==name)


class Mode:
    def is_target_verify(self):
        return False


EXTEND=Mode(); Mode.EXTEND=EXTEND


class Gate:
    def __init__(self, ready):
        self.ready=ready
        self.queries=0
    def query(self):
        self.queries+=1
        return self.ready


def build(repo, candidate):
    backend = (repo/BACKEND).read_text() if candidate else pinned(repo,BACKEND)
    raw = (repo/SPAN).read_text()
    calls=[]
    def flash(**kwargs):
        calls.append(kwargs)
        q=kwargs["q"]
        return (q,torch.zeros(q.shape[1],q.shape[0])) if kwargs["return_softmax_lse"] else q
    env=dict(torch=torch, dataclass=dataclass, _PAGE=160, ForwardMode=Mode, is_cp_active=lambda fb:False,
             AttentionType=NS(ENCODER_ONLY="enc",DECODER_BIDIRECTIONAL="bi"),flash_attn_with_kvcache=flash)
    def install(n):
        exec("from __future__ import annotations\n"+ast.unparse(n),env)
    for name in ("paged_view","paged_kv_view"):
        install(node(pinned(repo,PAGED),name))
    names=("forward_extend","get_paged_mha_kv_cache","prepare_paged_mha_query","_mxfp8_sf_kwargs")
    for name in names:
        install(node(backend,name))
    obj=type("Backend",(),{name:env[name] for name in names})()
    obj.__dict__.update(topk=1,page_size=1,fa_impl_ver=3,num_splits=0,use_mla=False,fa_skip_kv_cache=False,
        kv_cache_is_mxfp8=False,kv_cache_dtype=torch.float8_e4m3fn,kv_cache_dtype_str="fp8_e4m3",local_attn_builder=None,
        use_sliding_window_kv_pool=False,kv_index_translator=NS(is_translating=False),_disable_scheduler_metadata_precompute=False,
        attn_cp_size=1,is_encoder_decoder=False,_kv_shard_pool=None,full_cg_prefill_metadata=None,is_draft_runner=False)
    install(node(raw,"PrefillKVSpan",ast.ClassDef))
    return obj,calls,env["PrefillKVSpan"]


def all_rows(tensor,table,b,length):
    if table is None:
        return tensor[b,:length].view(torch.uint8)
    width=tensor.shape[1];idx=torch.arange(length)
    return tensor.view(torch.uint8)[table[b,idx//width].long(),idx%width]


def test(repo):
    attrs = ("fa3","sm90","page_one","fp8","target","deterministic_off","env_off","mha","global_kv",
             "physical_ids","dcp1","no_prefill_cp")
    source = (repo/BACKEND).read_text()
    assignment = next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Attribute) and t.attr=="_inspect_prefill_spans" for t in n.targets))
    predicate = compile(ast.Expression(assignment.value),"<V143 startup>","eval")
    for bits in itertools.product((False,True),repeat=len(attrs)):
        v=dict(zip(attrs,bits))
        obj=NS(page_size=1 if v["page_one"] else 16,kv_cache_dtype=torch.float8_e4m3fn if v["fp8"] else torch.bfloat16,
               is_draft_runner=not v["target"],use_mla=not v["mha"],_kv_shard_pool=None if v["global_kv"] else 'shard',
               kv_index_translator=NS(is_translating=not v["physical_ids"]))
        e=dict(self=obj,torch=torch,fa_impl_ver=3 if v["fa3"] else 4,get_platform=lambda:NS(is_sm90=v["sm90"]),
               deterministic=not v["deterministic_off"],envs=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:not v["env_off"])),
               get_parallel=lambda:NS(attn_dcp_size=1 if v["dcp1"] else 2,enable_prefill_cp=not v["no_prefill_cp"]))
        assert eval(predicate,e) == all(bits)
    torch.manual_seed(143);torch.set_num_threads(1)
    parent,p_calls,_=build(repo,False)
    candidate,c_calls,Span=build(repo,True)
    stats=dict(startup_predicates=2**len(attrs),pair_cases=0,compared_fp8_bytes=0,attention_output_scalars=0,routes={})
    kinds=("dense","dense_gap","coarse","tail_fragment","shared_key","not_ready","padded","graph","compile",
           "swa","lse","sinks","no_certificate","stale_table","stale_lengths","mut_table","mut_lengths",
           "deterministic","out_bounds","odd_pitch","offset")
    for kind in kinds:
        n=640 if kind in ("coarse","shared_key") else 649
        batch=1 if kind=="dense" else 2
        qrows=[517]*batch
        live=[n]*batch
        offset=1 if kind in ("coarse","offset") else 159
        starts=[offset+b*1600 for b in range(batch)]
        table=torch.tensor([[s+j for j in range(n)] for s in starts],dtype=torch.int32)
        if kind in ("coarse","tail_fragment","shared_key"):
            table[:,160:320]+=6400
            if kind=="shared_key":
                table[1,320:480]=table[0,320:480]
        lens=torch.tensor(live,dtype=torch.int32)
        maxid=int(table.max())
        size=maxid+1 if kind!="out_bounds" else maxid
        pitch=145 if kind=="odd_pitch" else 144
        kv=[torch.randint(-4,5,(size,pitch),dtype=torch.int16).to(torch.float8_e4m3fn)[:,:128].view(size,2,64) for _ in (0,1)]
        q=torch.randint(-2,3,(sum(qrows)+int(kind=="padded"),16,64),dtype=torch.int16).to(torch.float8_e4m3fn)
        cuq=torch.tensor([0,*torch.tensor(qrows).cumsum(0).tolist()],dtype=torch.int32)
        cuk=torch.tensor([0,*torch.tensor(live).cumsum(0).tolist()],dtype=torch.int32)
        fb=NS(forward_mode=EXTEND,dcp_kv_mask=None,batch_size=batch,extend_num_tokens=sum(qrows))
        scales=[torch.tensor([.25,.5]),torch.tensor([.125,.75])]
        layer=NS(tp_q_head_num=16,tp_k_head_num=2,tp_v_head_num=2,head_dim=64,v_head_dim=64,
                 is_cross_attention=False,attn_type="decoder",sliding_window_size=255 if kind=="swa" else -1,
                 layer_id=4,k_scale=scales[0],v_scale=scales[1],logit_cap=0.,scaling=.125)
        gate=Gate(kind!="not_ready")
        dense=kind not in ("coarse","tail_fragment","shared_key")
        coarse=kind in ("coarse","shared_key")
        status=torch.tensor([dense,coarse,offset,1600 if batch>1 else n,n,maxid],dtype=torch.int64)
        s=Span(table.clone() if kind=="stale_table" else table,lens.clone() if kind=="stale_lengths" else lens,
               sum(qrows),(table[:,::160] - offset)//160,status,gate)
        if kind=="mut_table":
            table.add_(0)
        if kind=="mut_lengths":
            lens.add_(0)
        for splits in (0,1,4):
            for obj in (parent,candidate):
                obj.num_splits=splits
                md=NS(cache_seqlens_int32=lens,max_seq_len_k=n,max_seq_len_q=max(qrows),cu_seqlens_q=cuq,
                      cu_seqlens_k=cuk,page_table=table,image_token_ranges=None,swa_spec_metadata=None,swa_out_cache_loc=None)
                if obj is candidate and kind!="no_certificate":
                    md.prefill_kv_span=s
                obj.forward_metadata=md
                obj.token_to_kv_pool=NS(get_kv_buffer=lambda _:kv)
                with patch.object(torch,"are_deterministic_algorithms_enabled",lambda:kind=="deterministic"), \
                        patch.object(torch.compiler,"is_compiling",lambda:kind=="compile"), \
                        patch.object(torch.cuda,"is_current_stream_capturing",lambda:kind=="graph"):
                    obj.forward_extend(q,None,None,layer,fb,sinks=q if kind=="sinks" else None,return_lse=kind=="lse")
            before=p_calls.pop();after=c_calls.pop()
            for field in set(before)-{"k_cache","v_cache","page_table"}:
                a,b=before[field],after[field]
                assert (torch.equal(a.view(torch.uint8),b.view(torch.uint8)) if isinstance(a,torch.Tensor) else a==b),(kind,field)
            route="dense" if after["page_table"] is None else "coarse" if after["k_cache"].shape[1]==160 else "stock"
            expected="dense" if kind in ("dense","dense_gap","offset") else "coarse" if kind in ("coarse","shared_key") else "stock"
            assert route==expected,(kind,route)
            stats["routes"][kind]=route;stats["pair_cases"]+=1
            for field,physical in zip(("k_cache","v_cache"),kv):
                view=after[field]
                assert view.untyped_storage().data_ptr()==physical.untyped_storage().data_ptr()
                if kind=="out_bounds":continue
                for b in range(batch):
                    reference=all_rows(before[field],table,b,n)
                    actual=all_rows(view,after["page_table"],b,n)
                    assert torch.equal(reference,actual)
                    stats["compared_fp8_bytes"]+=actual.numel()
            if route!="stock":
                for b in range(batch):
                    outputs=[]
                    for kw in (before,after):
                        keys=all_rows(kw["k_cache"],kw["page_table"],b,n).view(torch.float8_e4m3fn).float()
                        vals=all_rows(kw["v_cache"],kw["page_table"],b,n).view(torch.float8_e4m3fn).float()
                        z=[]
                        for qi in (0,228,qrows[b]-1):
                            kmax=n-qrows[b]+qi+1
                            for h in (0,15):
                                head=h//8
                                raw=keys[:kmax,head] @ q[b*qrows[b]+qi,h].float()
                                probs=(raw*kw["softmax_scale"]*kw["k_descale"][b,head]).softmax(0)
                                z.append((probs@vals[:kmax,head])*kw["v_descale"][b,head])
                        outputs.append(torch.stack(z))
                    assert torch.equal(*outputs)
                    stats["attention_output_scalars"]+=outputs[0].numel()
        if kind=="not_ready":
            assert s.outcome is None and gate.queries==3
            gate.ready=True
            _,_,proof=s.view(kv[0][:,None],kv[1][:,None],table,lens,sum(qrows))
            assert proof is None and s.outcome is not None
    # Native ordinary EXTEND advanced-indexes the req→slot table, then installs
    # a fresh metadata object. Recycle the same request slot and let the old
    # transfer finish after the new one. Its proof still names only the old table.
    req=torch.tensor([[3,4,5,6,7],[70,71,72,73,74]],dtype=torch.int32)
    sel=torch.tensor([1]);old=req[sel,:];old_lens=torch.tensor([5],dtype=torch.int32)
    late=Span(old,old_lens,5,None,torch.tensor([1,0,70,5,5,74]),Gate(False))
    req[sel]=torch.tensor([[80,93,55,54,53]],dtype=torch.int32)
    new=req[sel,:];fresh_lens=old_lens.clone();late.ready.ready=True
    raw=torch.zeros(99,1,2,64).to(torch.float8_e4m3fn)
    assert old.data_ptr()!=new.data_ptr() and old.tolist()==[[70,71,72,73,74]]
    assert late.view(raw,raw,new,fresh_lens,5)[2] is new
    assert late.view(raw,raw,new,old_lens,5)[2] is new
    with torch.inference_mode():
        owned=req[sel,:]
        owned_lens=old_lens.clone()
        receipt=Span(owned,owned_lens,5,None,torch.tensor([0,0,80,5,5,93]),Gate(True))
        assert receipt.table_version is None and receipt.length_version is None
        assert receipt.view(raw,raw,owned,owned_lens,5)[2] is owned
    stats["late_request_slot_reuse_checked"]=True
    stats["torch"]=torch.__version__
    stats["native_h200_fa3"]=False
    return stats


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("repo",type=Path)
    print(json.dumps(test(parser.parse_args().repo),indent=2))
