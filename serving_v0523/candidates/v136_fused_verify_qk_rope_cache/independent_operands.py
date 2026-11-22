"""Exercise native RadixAttention → FA3 target verify with no K and an inaccessible raw V."""
from pathlib import Path
import argparse
import ast
import json
from types import SimpleNamespace as NS
import subprocess

PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
B="python/sglang/srt/layers/attention/flashattention_backend.py"
R="python/sglang/srt/layers/radix_attention.py"

def extract(text,cls,name):
    module=ast.parse(text)
    node=next(x for x in module.body if isinstance(x,ast.ClassDef) and x.name==cls)
    f=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name==name)
    f.decorator_list=[]
    return "from __future__ import annotations\n"+ast.unparse(f)

class RawV:
    def __getattribute__(self,name): raise AssertionError("unexpected read of raw V: "+name)

class Q:
    dtype="e4m3"
    def __init__(self,n): self.shape=(n,16,64)
    def contiguous(self): return self
    def view(self,*shape): return self

def check(stage):
    mode=NS(is_target_verify=lambda:True,is_extend=lambda:False)
    calls=[];backend=NS()
    def record(**kwargs): calls.append(kwargs);return kwargs["q"]
    native=NS(ENCODER_ONLY=1,DECODER_BIDIRECTIONAL=2)
    env=dict(AttentionType=native,is_cp_active=lambda _:False,flash_attn_with_kvcache=record,
             is_in_breakable_cuda_graph=lambda:False,is_in_full_prefill_graph=lambda:False,
             is_in_prefill_graph_scope=lambda:False,padded_extend_real_tokens=lambda *a:None,
             get_attn_backend=lambda:backend)
    source=(stage/B).read_text()
    radix=subprocess.check_output(["git","-C",str(stage),"show",PIN+":"+R],text=True)
    exec(extract(source,"FlashAttentionBackend","forward_extend"),env)
    backend.forward=lambda q,k,v,layer,fb,save,**kw: env["forward_extend"](backend,q,k,v,layer,fb,save,**kw)
    exec(extract(radix,"RadixAttention","forward"),env)
    def tensor(rows): return NS(shape=(rows,))
    class Table: pass
    md=NS(max_seq_len_q=8,cache_seqlens_int32=tensor(1),cu_seqlens_q=tensor(2),
          cu_seqlens_k=tensor(2),page_table=Table(),swa_page_table=Table(),swa_spec_metadata=None,image_token_ranges=None)
    fb=NS(forward_mode=mode,batch_size=1,dcp_kv_mask=None)
    pools={};backend.__dict__.update(forward_metadata=md,use_mla=False,topk=1,local_attn_builder=None,
         is_encoder_decoder=False,fa_impl_ver=3,fa_skip_kv_cache=False,use_sliding_window_kv_pool=True,num_splits=0)
    backend.prepare_paged_mha_query=lambda q,*a,**kw:(q,None,None,None,None)
    backend._mxfp8_sf_kwargs=lambda *a:{}
    backend.get_paged_mha_kv_cache=lambda layer:pools[layer.layer_id]
    for i in range(62):
        q=Q(32);pools[i]=(object(),object())
        swa=(i%4!=0)
        layer=NS(sliding_window_size=2047 if swa else -1,is_cross_attention=False,attn_type=0,layer_id=i,
                 tp_q_head_num=16,tp_k_head_num=2,v_head_dim=64,head_dim=64,scaling=.125,logit_cap=0.,
                 use_prefill_attention_wrapper=True)
        assert env["forward"](layer,q,None,RawV(),fb,False) is q
        args=calls[-1];assert args["q"] is q and (args["k_cache"],args["v_cache"])==pools[i]
        assert args["page_table"] is (md.swa_page_table if swa else md.page_table)
        assert args["cache_seqlens"] is md.cache_seqlens_int32 and args["cu_seqlens_q"] is md.cu_seqlens_q
        assert args["max_seqlen_q"]==8 and args["window_size"]==((2047,0) if swa else (-1,-1))
    return dict(complete_native_radix_and_forward_extend=True,passed_layers=len(calls),raw_v_reads=0,
                selected_full_layers=16,selected_sliding_layers=46,hardware=False)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",type=Path);print(json.dumps(check(p.parse_args().stage),indent=2))
