"""Run both frozen patch sources through the Muse, Radix and FA3 prefill entry points on CPU."""
import argparse
import ast
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import PropertyMock, patch

import torch

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE.parent
SR = "python/sglang/srt/"
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
FP8 = torch.float8_e4m3fn
H, D, QH = 2, 64, 16
PACK = "sglang.kernels.ops.attention.batched_prompt_kv"
MODEL = SR + "models/muse_spark.py"
NORM = "python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py"
CALLBACK = SR + "layers/radix_attention.py"
GRAPH = SR + "layers/attention/graph_utils.py"
BASE = SR + "layers/attention/base_attn_backend.py"
CTX = SR + "model_executor/runner_utils/prefill_graph.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Use the existing strict unified-diff reader on the official checkout and the
# previously validated pool harness. Set the sources to exactly V219 and V223.
def fixtures(candidate):
    old_argv = sys.argv.copy()
    sys.argv = [str(__file__), str(HERE)]
    try:
        rep = module("v223_source_reader", CANDIDATES / "v217_captured_verify_context" / "independent_model_entry.py")
        harness = module("v223_pool_harness", CANDIDATES / "v178_direct_fp8_prompt_kv" / "check.py")
    finally:
        sys.argv = old_argv
    rep.SRT = ""
    rep.STAGE = None
    rep.PARENT = (CANDIDATES / "v219_verify_graph_sliding_tiles/patch.diff").read_text()
    rep.CHILD = (candidate / "patch.diff").read_text()
    return lru_cache(None)(rep.read_source), harness, (sha256(rep.PARENT.encode()).hexdigest(), sha256(rep.CHILD.encode()).hexdigest())


def define(path, names, env, cls=None):
    tree = ast.parse(source(path, child))
    body = tree.body if not cls else next(x.body for x in tree.body if getattr(x, "name", "") == cls)
    for name in names:
        node = next(x for x in body if getattr(x, "name", "") == name)
        if name == "_eager_attention":
            node.decorator_list = []
        exec("from __future__ import annotations\n" + ast.unparse(node), env)


class KernelBoundary:
    """Storage semantics for the launch boundary; interpreter.py tests the actual Triton."""
    def __getitem__(self, grid):
        assert grid[0] > 0 and grid[1] >= 2
        def kernel(ck, cv, nk, nv, table, lengths, q, starts, used, ok, ov, kp, vp, tp, *, ROW, BLOCK, GUARD):
            assert (kp, vp, tp) == (ck.stride(0), cv.stride(0), table.stride(0))
            assert all(x.is_contiguous() for x in (nk, nv, ok, ov))
            assert ROW == H * D and BLOCK == min(32, 4096 // ROW) and GUARD == 256
            for seq in range(grid[1]):
                length, live = int(lengths[seq]), int(used[seq])
                q0, q1 = q[seq:seq+2].tolist()
                halo = live - (q1 - q0)
                i, j = starts[seq:seq+2].tolist()
                assert j-i == live+GUARD
                for old, new, out in ((ck, nk, ok), (cv, nv, ov)):
                    prefix_slots = table[seq, length-live:length-live+halo].long()
                    out[i:i+halo].copy_(old[:, 0][prefix_slots])
                    out[i+halo:i+live].copy_(new[q0:q1])
                    out[i+live:j].zero_()
        return kernel


def data(rows, prefixes, pad):
    g = torch.Generator().manual_seed(22341)
    live = sum(rows)
    qkv = torch.randn((live+pad, (QH+2*H)*D), generator=g).bfloat16()
    if pad:
        qkv[live:].fill_(float("nan"))
    totals = [r+p for r,p in zip(rows,prefixes)]
    # Prefix slots in the memory-budget rejection can reuse already cached
    # physical rows. All positive execution cases instead have unique slots.
    prefix_slots = min(sum(prefixes), 14000)
    cap = prefix_slots + live + 47
    tables = []
    for _ in range(2):
        slot = torch.randperm(cap-1, generator=g).add_(1).int()
        table = torch.zeros((len(rows), max(totals)+9), dtype=torch.int32)[:, :max(totals)]
        pi, ni = 0, prefix_slots
        for b, (r, p) in enumerate(zip(rows, prefixes)):
            if p:
                table[b,:p] = slot[torch.arange(pi,pi+p) % prefix_slots]
            table[b,p:p+r] = slot[ni:ni+r]
            pi, ni = pi+p, ni+r
        tables.append(table)
    banks = [tuple(torch.randn((cap,H,D), generator=g).to(FP8) for _ in range(2)) for _ in range(2)]
    # Poison unused slot zero; downstream reads must never borrow padded keys.
    for pair in banks:
        for bank in pair:
            bank[0].fill_(float("nan"))
    return qkv, totals, tables, banks


def initialize(b, pool, fb, tables, writes, count, lengths, rows, guard):
    """Build normal FA3 metadata on the live batch, then bind its plan to replay."""
    kvplan = SR + "mem_cache/kv_loc_plan.py"
    translator = SR + "mem_cache/kv_index_translator.py"
    kinds = NS(FULL="full", SLIDING_WINDOW="swa")
    env = dict(torch=torch, dataclass=dataclass, ForwardMode=harness.Mode, IdSpaceKind=kinds)
    define(kvplan,["pad_with_sink"],env)
    define(harness.SWA,["translate_loc_from_full_to_swa"],env,"SWAKVPool")
    pool.translate_loc_from_full_to_swa=env["translate_loc_from_full_to_swa"].__get__(pool)
    mapping = torch.zeros(pool.full_kv_pool.data[0].shape[0],dtype=torch.int32)
    for a,bidx in zip(*tables):
        mapping[a.long()]=bidx
    pool.full_to_swa_index_mapping=mapping
    methods=["write_ids","space"]
    define(translator,methods,env,"KVIndexTranslator")
    trans=type("Translator",(),{n:env[n] for n in methods})()
    trans.is_translating=False
    trans._spaces={"full":NS(key="full",write=None),"swa":NS(key="swa",write=pool.translate_loc_from_full_to_swa)}
    methods=["__init__","write_ids","_window_in","_cols","bind","bind_replay","is_translated_for"]
    define(kvplan,methods,env,"KVLocPlan")
    plan_type=type("Plan",(),{n:env[n] for n in methods})
    fb.seq_lens=lengths
    fb.extend_seq_lens=torch.tensor(rows,dtype=torch.int32)
    fb.encoder_lens=None
    fb.req_pool_indices=torch.arange(count)
    plan=plan_type(source=trans,req_pool_indices=fb.req_pool_indices,seq_lens=lengths,seq_lens_cpu=fb.seq_lens_cpu,
                   write_virtual=writes[0][:sum(rows)])
    live_batch=copy(fb)
    plan.bind(live_batch,trans)
    template=next(n for n in ast.parse(source(harness.BACK,child)).body if getattr(n,"name",None)=="FlashAttentionMetadata")
    exec("from __future__ import annotations\n"+ast.unparse(template),env)
    methods=["init_forward_metadata","_get_swa_write_locations","_maybe_init_local_attn_metadata"]
    define(harness.BACK,methods,env,"FlashAttentionBackend")
    for n in methods:
        setattr(b,n,env[n].__get__(b))
    b.kv_index_translator=trans
    b.req_to_token_pool=NS(req_to_token=tables[0])
    b.is_prefill_aware_swa=False
    b._kv_shard_pool=None
    b.init_forward_metadata(live_batch)
    md=b.forward_metadata
    assert torch.equal(md.swa_out_cache_loc,writes[1][:sum(rows)])
    assert torch.equal(md.page_table,tables[0]) and torch.equal(md.swa_page_table,tables[1])
    assert md.cache_seqlens_int32 is lengths
    plan.bind_replay(fb,trans,slots=writes[0])
    if guard=="static":
        b.full_cg_prefill_metadata=md
    if guard=="mirror":
        fb.seq_lens_cpu=lengths+1
    return md


def exercise(label, rows, prefixes, left, expected_dense, scale=True, guard=None):
    global child
    pad = 0 if label == "eager" else 17
    raw_qkv, totals, tables, banks = data(rows, prefixes, pad)
    live, count = sum(rows), len(rows)
    cu = torch.tensor([0, *torch.tensor(rows).cumsum(0).tolist()], dtype=torch.int32)
    lengths = torch.tensor(totals, dtype=torch.int32)
    writes = [torch.cat([tab[b,p:p+r] for b,(p,r) in enumerate(zip(prefixes,rows))]+[torch.zeros(pad,dtype=torch.int32)]).long() for tab in tables]
    ks, vs = (torch.tensor([.75]),torch.tensor([1.625])) if scale else (None,None)
    pairs = []
    for child in (False, True):
        graph = {}
        exec(source(CTX, child), graph)
        runtime = dict(torch=torch,contextmanager=contextmanager,**{n:graph[n] for n in ("get_prefill_raw_num_tokens", "is_in_full_prefill_graph", "is_in_prefill_graph_scope")})
        runtime.update(is_in_breakable_cuda_graph=lambda:label!="full",attention_query_tokens=lambda fb,x:x,attention_output_tokens=lambda fb,x:x,padded_extend_real_tokens=lambda q,fb:None)
        define(GRAPH,["attention_input_scope","_zero_padded_tokens","allocate_attention_outputs"],runtime)
        define(CALLBACK,["forward","_eager_attention"],runtime,"RadixAttention")
        radix = type("Radix",(),{"__call__":runtime["forward"],"_eager_attention":runtime["_eager_attention"]})()
        opts = dict(layer_id=0,tp_v_head_num=H,tp_k_head_num=H,tp_q_head_num=QH,v_head_dim=D,head_dim=D,qk_head_dim=D,
                    is_cross_attention=False,attn_type="decoder",sliding_window_size=left,k_scale=ks,v_scale=vs,
                    logit_cap=17. if scale else 0.,scaling=D**-.5,use_prefill_attention_wrapper=True)
        radix.__dict__.update(opts)
        def injected(ns, path, names, cls=None, old=True):
            ns.update({key:runtime[key] for key in ("get_prefill_raw_num_tokens","is_in_prefill_graph_scope","is_in_full_prefill_graph")})
            if child and path==harness.HELP and "prepare_direct_prompt_kv" in names:
                names=[*names,"prepare_batched_prompt_kv"]
            define(path,names,ns,cls)
        with patch.object(harness, "src", side_effect=lambda path,*args:source(path,child)), patch.object(harness,"add",side_effect=injected):
            b,pool,paged,stored=harness.mk(False,*banks,raw_qkv,*tables,left,ks,vs)
        b._hopper_verify_sms=132
        base = dict(debug_kernel_api=lambda f:f)
        define(BASE,["forward"],base,"AttentionBackend")
        b.forward=base["forward"].__get__(b)
        runtime.update(get_attn_backend=lambda:b, get_token_to_kv_pool=lambda:pool, get_memory=lambda:NS(enable_unified_memory=False),
                       TRTLLMHAAttnBackend=type("TRT",(),{}))
        define(NORM,["_slices","_rope_native","forward_native"],runtime,"FusedQKNorm")
        norm=type("Norm",(),{"_slices":runtime["_slices"],"_rope_native":runtime["_rope_native"],"__call__":runtime["forward_native"]})()
        norm.__dict__.update(_num_q_heads=QH,_num_kv_heads=H,_head_dim=D,_eps=1e-5)
        define(MODEL,["_fused_kv_store","forward"],runtime,"MuseSpark1xAttention")
        model=type("Attention",(),{"forward":runtime["forward"],"_fused_kv_store":runtime["_fused_kv_store"]})()
        freq = torch.arange((live+pad)*(D//2)).view(live+pad,D//2) / 10000.
        rope = NS(cos_sin_cache=torch.cat((freq.cos(),freq.sin()),dim=-1)) if left>=0 else None
        model.__dict__.update(qkv_proj=lambda x:(x,None),q_size=QH*D,kv_size=H*D,use_o_gate=False,
                              _fused_qk_norm=norm,rotary_emb=rope,attn=radix,_fused_o_norm_gate=lambda x,g:x,o_proj=lambda x:(x,None))
        dense=[]
        def varlen(**kw):
            dense.append(kw)
            return kw["q"].bfloat16() if kw["out"] is None else kw["out"]
        b.flash_attn_varlen_func=varlen
        fb=NS(seq_lens_cpu=lengths,forward_mode=harness.Mode.EXTEND,batch_size=count,dcp_kv_mask=None,extend_prefix_lens_cpu=list(prefixes),
              extend_seq_lens_cpu=list(rows),out_cache_loc=writes[0],out_cache_loc_is_physical=False,positions=torch.arange(live+pad),
              global_num_token_non_padded_cpu=live,mha_return_lse=False,_attn_output=None,req_pool_indices_cpu=torch.arange(count) if label=="eager" else None)
        if guard=="mixed": fb.mixed_verify_logits_indices=cu
        if guard=="no_mirror": fb.req_pool_indices_cpu=None
        initialize(b,pool,fb,tables,writes,count,lengths,rows,guard)
        if guard=="writer_hook":
            pool.set_kv_buffer=pool.set_kv_buffer
        runtime["get_forward_batch"]=lambda:fb
        pack=ModuleType(PACK)
        if child:
            pipe={"torch":torch,"triton":NS(cdiv=lambda a,b:(a+b-1)//b),"_pack_prompt_rows":KernelBoundary()}
            define("python/sglang/kernels/ops/attention/batched_prompt_kv.py",["pack_prompt_rows"],pipe)
            pack.pack_prompt_rows=pipe["pack_prompt_rows"]
        context=nullcontext() if label=="eager" else graph["prefill_graph_scope"](full_graph=label=="full",
                                    raw_num_tokens=None if label=="warmup" else live-1 if label=="wrong_raw" else live)
        qkv=raw_qkv.clone()
        with context, torch.inference_mode(), patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True), \
             patch.object(torch.cuda,"is_current_stream_capturing",return_value=guard=="capturing"), \
             patch.object(torch,"are_deterministic_algorithms_enabled",return_value=guard=="deterministic"), patch.dict(sys.modules,{PACK:pack}):
            result=model.forward(fb.positions,qkv,fb)
        assert fb.out_cache_loc is writes[0] and fb._attn_output is None
        assert len(paged)+len(dense)==len(stored)==1
        assert bool(dense)==bool(child and expected_dense),(label,guard,child,bool(dense))
        call=(dense or paged)[0]
        assert call["q"].shape==(live,QH,D) and call["q"].dtype==FP8
        if label!="eager":
            assert call["out"].shape==(live,QH,D) and call["out"].data_ptr()==result.data_ptr()
            if label not in ("warmup","wrong_raw"):
                assert not torch.count_nonzero(result[live:])
        else: assert call["out"] is None
        assert torch.equal(stored[0][0],writes[int(left>=0)][:live])
        pairs.append((call,stored[0],(*pool.full_kv_pool.data,*pool.swa_kv_pool.data),qkv))
    (a, aw, ac, aq),(b, bw, bc, bq)=pairs
    for x,y in zip((*aw,*ac),(*bw,*bc)):
        assert torch.equal(x.view(torch.uint8),y.view(torch.uint8))
    assert torch.equal(aq.view(torch.int16),bq.view(torch.int16))
    assert torch.equal(a["q"].view(torch.uint8),b["q"].view(torch.uint8))
    assert torch.equal(a["cu_seqlens_q"],b["cu_seqlens_q"])
    for key in ("softmax_scale","softcap","causal","window_size","max_seqlen_q","num_splits"):
        assert a[key]==b[key],key
    if scale:
        for key,scalar in (("k_descale",ks),("v_descale",vs)):
            assert torch.equal(a[key],b[key]) and torch.equal(b[key],scalar.expand(count,H))
    else:
        assert "k_descale" not in b and "v_descale" not in b
    checked_scalars=0
    if expected_dense:
        for seq,(r,p) in enumerate(zip(rows,prefixes)):
            cut=0 if left<0 else max(0,p-left)
            live_k=r+p-cut
            i,j=b["cu_seqlens_k"][seq:seq+2].tolist()
            assert j-i==live_k+256 and int(b["seqused_k"][seq])==live_k
            assert b["max_seqlen_k"]==max(b["seqused_k"]).item()
            ref=[]
            for fld,bank in (("k","k_cache"),("v","v_cache")):
                packed=b[fld].view(torch.uint8)
                physical=a[bank][:,0].view(torch.uint8)[a["page_table"][seq,:p+r].long()]
                assert torch.equal(packed[i:i+live_k],physical[cut:]) and not torch.count_nonzero(packed[i+live_k:j])
                ref.append(physical.view(FP8))
            for qi in (0,r//2,r-1):
                hi=p+qi+1
                lo=0 if left<0 else max(0,hi-1-left)
                for head in (0,QH-1):
                    query=b["q"][int(cu[seq])+qi,head].double()
                    def attend(key,value):
                        z=(key[:,head//(QH//H)].double()@query)*(D**-.5)*(ks.item() if scale else 1.)
                        if radix.logit_cap:
                            z=z.div(radix.logit_cap).tanh().mul(radix.logit_cap)
                        return z.softmax(0)@value[:,head//(QH//H)].double()*(vs.item() if scale else 1.)
                    x=attend(b["k"][i+lo-cut:i+hi-cut],b["v"][i+lo-cut:i+hi-cut])
                    y=attend(ref[0][lo:hi],ref[1][lo:hi])
                    assert torch.equal(x,y) and torch.isfinite(x).all()
                    checked_scalars+=D
    else:
        assert torch.equal(a["page_table"],b["page_table"])
    return dict(mode=label,guard=guard,rows=rows,prefixes=prefixes,left=left,scaled=scale,dense_candidate=expected_dense,
                parent_and_candidate_writes=2,physical_cache_bytes=sum(x.numel() for x in ac),attention_output_scalars=checked_scalars)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, nargs="?", default=HERE, help="directory containing the candidate full patch.diff")
    candidate = parser.parse_args().candidate.resolve()
    source,harness,hashes=fixtures(candidate)
    for name in ("is_decode","is_idle","is_mixed","is_decode_or_idle","is_draft_extend_v2"):
        setattr(harness.Mode,name,lambda self:False)
    harness.Mode.is_extend=lambda self:True
    harness.Mode.is_extend_or_draft_extend_or_mixed=lambda self,**kwargs:True
    rows=[37,1735,2339]
    prefixes=[0,2713,19]
    cases=[]
    for left in (-1,2047):
        cases.extend([exercise("breakable",rows,prefixes,left,True),exercise("eager",rows,prefixes,left,True,False)])
    for mode,guard in (("wrong_raw",None),("warmup",None),("full",None),("breakable","mixed"),("breakable","mirror"),("breakable","static"),("breakable","capturing"),("breakable","writer_hook"),("breakable","deterministic"),("eager","no_mirror")):
        cases.append(exercise(mode,rows,prefixes,2047,False,guard=guard))
    cases.append(exercise("breakable",[1066,3033],[0,0],-1,True))
    cases.append(exercise("breakable",[2000,2095],[11,17],-1,False))
    cases.append(exercise("breakable",[1777,2319],[122368,0],-1,True))
    cases.append(exercise("breakable",[1777,2319],[122369,0],-1,False))
    cases.append(exercise("breakable",[1777,2319],[122369,0],2047,True))
    print(json.dumps(dict(candidate=candidate.name,parent_patch_sha256=hashes[0],candidate_patch_sha256=hashes[1],pinned_base=PIN,
        torch=torch.__version__,physical_H200=False,triton_boundary="CPU storage model",FA3_boundary="recorded call and independent fp64 outputs",
        cases=cases,paired_forwards=2*len(cases),checked_cache_bytes=sum(x["physical_cache_bytes"] for x in cases),
        attention_output_scalars=sum(x["attention_output_scalars"] for x in cases)),indent=2))
