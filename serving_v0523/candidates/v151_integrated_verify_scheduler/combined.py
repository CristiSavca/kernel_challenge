"""Run the registered Muse attention, FP8 writer selector, FA3 consumer and model scheduler scope together."""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import runpy
import sys
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import torch

DIR=Path(__file__).resolve().parent
S=runpy.run_path(str(DIR.parent/"v132_verify_scheduler_metadata/verify_dispatch.py"))
D=runpy.run_path(str(DIR.parent/"v136_fused_verify_qk_rope_cache/dispatch_check.py"))


def ingress(stage):
    path=stage/S["BACKEND"]
    ctor=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.ClassDef) and n.name=="FlashAttentionBackend")
    init=next(n for n in ctor.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    ix=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="self._verify_schedule_sms")
    block=compile(ast.Module(init.body[ix:ix+2],type_ignores=[]),str(path),"exec")
    native=dict(enabled=True,dflash=True,draft=False,fa=3,sm=True,page=1,model=torch.bfloat16,kv=torch.float8_e4m3fn,
                kvname="fp8_e4m3",mx=False,determ=False,envdet=False,ragged="static",dcp=1,disabled=False,
                encoder=False,mla=False,topk=1,local=None,prefill_aware=False,cp=1,shard=None,translate=False,splits=0)
    changed=dict(enabled=False,dflash=False,draft=True,fa=4,sm=False,page=160,model=torch.float16,kv=torch.float16,
                 kvname="auto",mx=True,determ=True,envdet=True,ragged="dynamic",dcp=2,disabled=True,
                 encoder=True,mla=True,topk=2,local=object(),prefill_aware=True,cp=2,shard=object(),translate=True,splits=1)
    for field in (None,*native):
        a=NS(**(native if field is None else native|{field:changed[field]}))
        me=NS(page_size=a.page,fa_impl_ver=a.fa,is_draft_runner=a.draft,kv_cache_dtype=a.kv,
              kv_cache_dtype_str=a.kvname,kv_cache_is_mxfp8=a.mx,_disable_scheduler_metadata_precompute=a.disabled,
              is_encoder_decoder=a.encoder,use_mla=a.mla,topk=a.topk,local_attn_builder=a.local,is_prefill_aware_swa=a.prefill_aware,
              attn_cp_size=a.cp,_kv_shard_pool=a.shard,kv_index_translator=NS(is_translating=a.translate),num_splits=a.splits,device="cuda:0")
        scope=dict(self=me,model_runner=NS(dtype=a.model,spec_algorithm=NS(is_dflash=lambda:a.dflash)),
                   get_platform=lambda:NS(is_sm90=a.sm),get_parallel=lambda:NS(attn_dcp_size=a.dcp),
                   get_spec=lambda:NS(speculative_dflash_share_verify_schedule=a.enabled),deterministic=a.determ,
                   envs=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:a.envdet),SGLANG_RAGGED_VERIFY_MODE=NS(get=lambda:a.ragged)),torch=torch)
        with patch.object(torch.cuda,"get_device_properties",return_value=NS(multi_processor_count=132)):exec(block,scope)
        assert me._verify_schedule_sms==(132 if field is None else 0),field
    return len(native)+1


def check(stage):
    counts=Counter();torch.set_num_threads(1);torch.manual_seed(151)
    backend,owner,forward,trace=S["build"](stage)
    layer_type=D["RadixAttention"];norm_type=D["FusedQKNorm"];Pool=D["MHATokenToKVPool"];Hybrid=D["SWAKVPool"]
    H=S["Hopper"];dev=lambda x:x.as_subclass(H)
    mode=S["Mode"];native_backend=backend.forward_extend.__globals__
    write=NS(for_layer=lambda fb,l,swa_loc:NS(loc=fb.out_cache_loc,swa_loc=swa_loc))
    native_backend.update(KVWriteLoc=write)
    env=dict(D);env.update(FlashAttentionBackend=type(backend),KVWriteLoc=write,envs=NS(SGLANG_ENABLE_ASYNC_ASSERT=NS(get=lambda:False)),
        get_fused_op_backend=lambda:None,is_cp_active=lambda _:False,is_in_prefill_graph_scope=lambda:False,
        KernelBackend=NS(TRITON="triton"),can_use_store_cache=lambda *_:True)
    choose=D["load_function"](stage/S["BACKEND"],"muse_verify_fp8_store",env)
    backend._muse_verify_encodings=dev(torch.zeros(65536,dtype=torch.uint8));backend.page_size=1
    raw_extract=runpy.run_path(str(DIR.parent/"v136_fused_verify_qk_rope_cache/independent_operands.py"))["extract"]
    scope=dict(torch=torch,get_attn_backend=lambda:backend)
    exec(raw_extract((stage/S["MUSE"]).read_text(),"MuseSpark1xAttention","forward"),scope)
    def pool():
        p=object.__new__(Pool);p.size=639;p.page_size=1;p.requires_physical_write_loc=p.is_quantized_kv_cache=False
        p.kv_cache_layout="nhd";p.dtype=torch.float8_e4m3fn;p.store_dtype=torch.uint8;p.row_dim=p.v_row_dim=128
        p._muse_verify_capacity=dev(torch.tensor([640],dtype=torch.int32));return p
    hybrid=Hybrid(pool(),pool());hybrid.layers_mapping={i:(i,i%4!=0) for i in range(62)}
    buffers={i:tuple(dev(torch.zeros(640,2,64,dtype=torch.float8_e4m3fn)) for _ in range(2)) for i in range(62)}
    Hybrid.get_kv_buffer=lambda self,i:buffers[i]
    def store(ks,vs,slots,k,v):
        for target,rhs in ((ks,k),(vs,v)):
            raw=rhs.contiguous().reshape(-1,128).to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
            target.view(-1,128).view(torch.uint8).index_copy_(0,slots.long(),raw)
    def fallback(self,l,loc,k,v,*unused):
        counts["ordinary_writes"]+=1
        store(*buffers[l.layer_id],loc.swa_loc if l.layer_id%4 else loc.loc,k,v)
    Hybrid.set_kv_buffer=fallback
    def fused(norm,raw,kv_store):
        q,k,v=raw.split((1024,128,128),-1)
        if kv_store is None:return q,k
        counts["fused_writes"]+=1
        assert kv_store.capacity is (hybrid.swa_kv_pool if state["i"]%4 else hybrid.full_kv_pool)._muse_verify_capacity
        store(kv_store.key,kv_store.value,kv_store.slots,k,v)
        return q.to(torch.float8_e4m3fn),None
    norm_type.forward=fused
    backend.token_to_kv_pool=hybrid
    backend.muse_verify_fp8_store=MethodType(choose,backend)
    layer_type.forward=lambda l,q,k,v,b,save_kv_cache:backend.forward_extend(q,k,v,l,b,save_kv_cache)
    state={}
    with patch.object(torch.cuda,"current_device",return_value=0),patch.object(torch.cuda,"current_stream",return_value=NS(cuda_stream=7)):
        for b in (1,4,16,32):
            for variant in ("mixed","flag_off","short","ragged","compiled","deterministic","fullgraph_replay"):
                md,fb,_,_,_,_=S["tensors"](b,8,1,128 if variant=="short" else 131072,16,2)
                fb.spec_info.ragged_verify_layout=object() if variant=="ragged" else None
                fb.out_cache_loc=dev(torch.arange(b*8,dtype=torch.int32)+13)
                md.swa_out_cache_loc=dev(torch.arange(b*8,dtype=torch.int32).flip(0)+104)
                backend.forward_metadata=md;owner._share_verify_schedule=variant!="flag_off";trace.fa.clear();trace.nodes.clear();trace.prepare.clear()
                state["calls"]=0
                def body(ids,pos,batch,input_embeds=None):
                    for i in range(62):
                        state["i"]=i
                        l=layer_type(i);l.sliding_window_size=2047 if i%4 else -1;l.tp_q_head_num=16;l.attn_type="decoder";l.logit_cap=0.;l.scaling=.125
                        norm=norm_type();x=dev(torch.randint(-9,10,(b*8,1280))).bfloat16()
                        attention=NS(qkv_proj=lambda h:(h,None),q_size=1024,kv_size=128,use_o_gate=False,_fused_qk_norm=norm,
                                     rotary_emb=None,_fused_kv_store=lambda f:None,attn=l,_fused_o_norm_gate=lambda o,g:o,o_proj=lambda o:(o,None))
                        handle=norm.register_forward_hook(lambda *_:None) if i%3==1 else None
                        old=counts["ordinary_writes"]+counts["fused_writes"]
                        out=scope["forward"](attention,None,x,batch)
                        if handle:handle.remove()
                        assert counts["ordinary_writes"]+counts["fused_writes"]==old+1
                        args=trace.fa[-1];slots=md.swa_out_cache_loc if i%4 else fb.out_cache_loc
                        assert args["q"].data_ptr()==out.data_ptr() and args["cache_seqlens"] is md.cache_seqlens_int32 and args["page_table"] is (md.swa_page_table if i%4 else md.page_table)
                        assert args["cu_seqlens_q"] is md.cu_seqlens_q
                        for key,arg,rhs in zip(("k_cache","v_cache"),buffers[i],x[:,1024:].split(128,1)):
                            assert args[key].data_ptr()==arg.data_ptr()
                            seen=arg.view(torch.uint8)[slots.long()].reshape(b*8,128)
                            assert torch.equal(seen,rhs.contiguous().to(torch.float8_e4m3fn).contiguous().view(torch.uint8))
                            counts["checked_fp8_bytes"]+=seen.numel()
                        state["calls"]+=1
                    return out
                owner.model=body
                with patch.object(torch.compiler,"is_compiling",return_value=variant=="compiled"),patch.object(torch,"are_deterministic_algorithms_enabled",return_value=variant=="deterministic"):
                    forward(owner,fb.out_cache_loc,None,fb)
                assert fb._fa3_verify_schedules is None and state["calls"]==62
                eligible=variant in ("mixed","fullgraph_replay")
                assert len(trace.prepare)==(2 if eligible else 0) and sum("scheduler_metadata" in c for c in trace.fa)==(62 if eligible else 0)
                counts["complete_verification_calls"]+=1;counts["FA3_consumers"]+=62;counts["native_metadata_prepares"]+=len(trace.prepare)
                if eligible:
                    for step in (3,9,19):
                        md.cache_seqlens_int32.add_(step)
                        for node in trace.nodes:node()
                        counts["length_refresh_replays"]+=1
    return dict(counts,physical_GPU=False,constructor_patterns=ingress(stage),
                method_scope="Real pure-Python FP8 ownership selection, Muse attention/outer call, paged preparation, FA3 forward and scheduler helper. GPU kernel math, streams, FA3 calls and semaphore are host stand-ins.")


if __name__=="__main__":print(json.dumps(check(Path(sys.argv[1]).resolve()),indent=2))
