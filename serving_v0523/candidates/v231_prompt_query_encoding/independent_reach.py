"""Replay V229 and a frozen candidate through Muse, the Breakable callback and the FA3 wrapper."""
import argparse
import ast
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import functools
from functools import cache
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import patch, PropertyMock

import torch
from torch import nn


DIR = Path(__file__).resolve().parent
CANDIDATES = DIR.parent
BASELINE = CANDIDATES / "v229_sharded_decode_serving"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("legacy_stage", nargs="?", type=Path, help=argparse.SUPPRESS)
parser.add_argument("--candidate", type=Path, help="candidate directory containing patch.diff and validation.json")
parser.add_argument("--stage", type=Path, help="also compare each exercised candidate source with this checkout")
parser.add_argument("--manifest", type=Path, help="candidate validation manifest; defaults to validation.json in the candidate directory")
parser.add_argument("--output", type=Path, help="JSON receipt; defaults to the selected candidate directory")
args = parser.parse_args()
if args.stage and args.legacy_stage:
    parser.error("supply only one stage checkout")
STAGE = args.stage or args.legacy_stage
STAGE = STAGE.resolve() if STAGE else None
selected = (args.candidate or (args.manifest.parent if args.manifest else DIR)).resolve()
manifest_path = (args.manifest or selected / "validation.json").resolve()
receipt_name = "independent_reach.json" if selected == DIR else "prompt_query_reach.json"
receipt_path = (args.output or selected / receipt_name).resolve()


def checked_patch(candidate, manifest):
    data = (candidate / "patch.diff").read_bytes()
    digest = sha256(data).hexdigest()
    if digest != manifest["patch_sha256"]:
        raise ValueError(f"{candidate.name} full patch digest {digest} differs from {manifest['patch_sha256']}")
    return data.decode("utf-8"), digest


def shell_command():
    command = ["env"]
    for key in ("PYTHONDONTWRITEBYTECODE", "PYTHONPATH"):
        if key in os.environ:
            command.append(f"{key}={os.environ[key]}")
    command.extend([sys.executable, str(Path(__file__).resolve()), "--candidate", str(selected),
                    "--manifest", str(manifest_path), "--output", str(receipt_path)])
    if STAGE:
        command.extend(["--stage", str(STAGE)])
    return shlex.join(command)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fx = load("v231_prefill_fixture", CANDIDATES/"v223_batched_direct_prompt/independent_replay.py")
reader = load("v231_reader", CANDIDATES/"v217_captured_verify_context/independent_model_entry.py")
reader.SRT = ""
manifest = json.loads(manifest_path.read_bytes())
baseline_manifest = json.loads((BASELINE/"validation.json").read_bytes())
if manifest["base"] != reader.PIN or baseline_manifest["base"] != reader.PIN:
    raise ValueError(f"Both patches must be relative to the pinned serving source {reader.PIN}")
reader.PARENT, parent_digest = checked_patch(BASELINE, baseline_manifest)
reader.CHILD, candidate_digest = checked_patch(selected, manifest)
reader.STAGE = STAGE
source_hashes = {"V229": {}, "candidate": {}}
with patch.object(sys, "argv", [__file__, str(DIR)]):
    harness = load("v231_pool", CANDIDATES/"v178_direct_fp8_prompt_kv/check.py")


@cache
def source(path, child):
    text = reader.read_source(path, child)
    data = text.encode("utf-8")
    if child and STAGE and data != (STAGE/path).read_bytes():
        raise ValueError(f"Candidate source differs from stage bytes for {path}")
    source_hashes["candidate" if child else "V229"][path] = sha256(data).hexdigest()
    return text


fx.source, fx.harness = source, harness
for method in ("is_decode", "is_idle", "is_mixed", "is_decode_or_idle", "is_draft_extend_v2"):
    setattr(harness.Mode, method, lambda self:False)
harness.Mode.is_extend = lambda self:True
harness.Mode.is_extend_or_draft_extend_or_mixed = lambda self,**kwargs:True
harness.ATTN.DECODER = "decoder"
BCG = fx.SR + "model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py"
FP8 = torch.float8_e4m3fn


def define(path, name, env, child, cls=None):
    body = ast.parse(source(path, child)).body
    if cls:
        body = next(n.body for n in body if getattr(n, "name", "") == cls)
    node = next(n for n in body if getattr(n, "name", "") == name)
    if name != "attention_input_scope":
        node.decorator_list = []
    exec("from __future__ import annotations\n" + ast.unparse(node), env)
    return env[name]


def module(name, **objects):
    inst = ModuleType(name)
    inst.__dict__.update(objects)
    return inst


def capture_backend(env, child):
    capvar, streamvar = ContextVar("test_capture", default=None), ContextVar("test_stream", default=None)
    env.update(functools=functools, _current_capture_var=capvar, _current_stream_var=streamvar,
               get_device_module=lambda: NS(current_stream=lambda: None))
    for fn in ("_weak_ref_if_tensor", "_copy_output", "_EagerOnGraphCallable", "eager_on_graph", "BreakableCUDAGraph"):
        define(BCG, fn, env, child)
    graph = env["BreakableCUDAGraph"]()
    phase = []
    cap = NS(cuda_graph=graph, _barrier_fn=None, _begin_new_segment=lambda:phase.append("begin"),
             _end_current_segment=lambda:phase.append("end"))
    return capvar, cap, phase


def run(child, rows, prefixes, left, guard, scale):
    fx.child = child
    define_fx = fx.define
    H, D, QH = fx.H, fx.D, fx.QH
    live, count, pad = sum(rows), len(rows), 13
    qkv, totals, tables, banks = fx.data(rows, prefixes, pad)
    lengths = torch.tensor(totals, dtype=torch.int32)
    writes = [torch.cat([tab[i,p:p+n] for i,(p,n) in enumerate(zip(prefixes,rows))] +
                        [torch.zeros(pad,dtype=torch.int32)]).long() for tab in tables]
    ks, vs = (torch.tensor([.75]), torch.tensor([1.625])) if scale else (None, None)
    graph = {}
    exec(source(fx.CTX, child), graph)
    env = dict(torch=torch, nn=nn, contextmanager=contextmanager, 
        is_in_breakable_cuda_graph=lambda:True, attention_query_tokens=lambda batch,n:n,
        attention_output_tokens=lambda batch,n:n,padded_extend_real_tokens=lambda *a:None,
        **{n:graph[n] for n in ("is_in_prefill_graph_scope","is_in_full_prefill_graph","get_prefill_raw_num_tokens")})
    capvar, cap, phase = capture_backend(env, child)
    for fun in ("attention_input_scope", "_zero_padded_tokens", "allocate_attention_outputs"):
        define(fx.GRAPH, fun, env, child)
    radix_forward = define(fx.CALLBACK, "forward", env, child, "RadixAttention")
    break_body = define(fx.CALLBACK, "_eager_attention", env, child, "RadixAttention")
    radix_cls = type("RadixAttention",(nn.Module,),{"forward":radix_forward, "_eager_attention":env["eager_on_graph"](break_body)})
    layer = radix_cls()
    layer.__dict__.update(layer_id=0, tp_v_head_num=H, tp_k_head_num=H, tp_q_head_num=QH, v_head_dim=D,
        head_dim=D, qk_head_dim=D, is_cross_attention=False, attn_type="decoder", sliding_window_size=left,
        k_scale=ks,v_scale=vs,logit_cap=2.,scaling=D**-.5,use_prefill_attention_wrapper=True)
    def injected(ns, path, names, cls=None, old=True):
        ns.update({key:env[key] for key in ("is_in_prefill_graph_scope","is_in_full_prefill_graph","get_prefill_raw_num_tokens")})
        if path==harness.HELP and "prepare_direct_prompt_kv" in names:
            names=[*names, "prepare_batched_prompt_kv"]
        define_fx(path,names,ns,cls)
    with patch.object(harness,"src",side_effect=lambda path,*a:source(path,child)), patch.object(harness,"add",side_effect=injected):
        backend,pool,paged,stores=harness.mk(False,*banks,qkv,*tables,left,ks,vs)
    attention_globals=type(backend).forward_extend.__globals__
    original_paged=attention_globals["flash_attn_with_kvcache"]
    sentinel=1.0078125
    def write_paged(**kw):
        result=original_paged(**kw)
        result.fill_(sentinel)
        return result
    attention_globals["flash_attn_with_kvcache"]=write_paged
    base_env=dict(debug_kernel_api=lambda func:func)
    define_fx(fx.BASE, ["forward"], base_env, "AttentionBackend")
    setattr(type(backend),"forward",base_env["forward"])
    backend._hopper_verify_sms=132
    backend._muse_verify_encodings = torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(FP8).view(torch.uint8)
    env.update(get_attn_backend=lambda:backend,get_token_to_kv_pool=lambda:pool,get_memory=lambda:NS(enable_unified_memory=False),
               TRTLLMHAAttnBackend=type("TRT",(),{}), FlashAttentionBackend=type(backend), ForwardMode=harness.Mode,
               AttentionType=harness.ATTN,is_cp_active=lambda batch:False)
    class BaseFusedOp:
        def _forward_backend_dynamic(self,*args,**kwargs):
            return self.forward_native(*args,**kwargs)
    candidate=object()
    normcls=type("FusedQKNorm",(nn.Module,BaseFusedOp),{})
    for fun in ("_slices","_rope_native","forward_native"):
        setattr(normcls,fun,define(fx.NORM,fun,env,child,"FusedQKNorm"))
    normcls.forward=normcls.forward_native
    normcls.backend_eligible=lambda self,*args: True
    norm=normcls()
    norm._num_q_heads, norm._num_kv_heads, norm._head_dim, norm._eps=QH,H,D,1e-6
    norm.is_torch_compile=False
    norm._forward_method=norm._forward_backend_dynamic
    norm._dynamic_backend_candidates=(candidate,)
    class ScalarQKKernel:
        def __getitem__(self,_):
            def launch(raw,qout,kout,pos,cos,*args,**kw):
                assert kw["HAS_KV_STORE"] is False and kw["QUERY_ENCODINGS"] is backend._muse_verify_encodings
                q,k=norm.forward_native(raw,pos if kw["HAS_ROPE"] else None,cos if kw["HAS_ROPE"] else None)
                qout.copy_(kw["QUERY_ENCODINGS"][(q.view(torch.int16).long() & 65535)])
                kout.copy_(k)
            return launch
    pkg="sglang.kernels.ops.attention.muse_spark_qk_norm_rope_store"
    modules={
        "sglang.kernels.fused_op": module("fused_op",BaseFusedOp=BaseFusedOp,get_fused_op_backend=lambda:None,
                                         _trace_enabled=False),
        pkg:module(pkg,FusedQKNorm=normcls),
        "sglang.kernels.spec":module("spec",KernelBackend=NS(TRITON=candidate)),
        "sglang.srt.layers.radix_attention":module("radix",RadixAttention=radix_cls),
        "sglang.srt.model_executor.runner_utils.prefill_graph":module("prefill_graph",**graph),
    }
    if child:
        env.update(VerifyFP8Store=type("Verify",(),{}),is_arch_support_pdl=lambda:False,
                   triton=NS(cdiv=lambda a,b:(a+b-1)//b, next_power_of_2=lambda x:1<<(x-1).bit_length()),
                   _qk_rms_norm_fwd_kernel=ScalarQKKernel())
        define(fx.NORM, "_triton_config",env,child)
        setattr(normcls,"_forward_triton",define(fx.NORM,"_forward_triton",env,child,"FusedQKNorm"))
        setattr(type(backend),"muse_prompt_query_encodings",define(harness.BACK,"muse_prompt_query_encodings",env,child,"FlashAttentionBackend"))
    model=type("Muse",(),{"forward":define(fx.MODEL,"forward",env,child,"MuseSpark1xAttention"),
                          "_fused_kv_store":define(fx.MODEL,"_fused_kv_store",env,child,"MuseSpark1xAttention")})()
    seq=torch.arange(live+pad)
    freq=torch.arange((live+pad)*(D//2)).view(live+pad,-1)/10000.
    rope=NS(cos_sin_cache=torch.cat((freq.cos(),freq.sin()),dim=1)) if left>=0 else None
    downstream=[]
    def consume(x,g=None):
        downstream.append(x.dtype)
        assert x.dtype==torch.bfloat16
        return x
    model.__dict__.update(qkv_proj=lambda x:(x,None),q_size=QH*D,kv_size=H*D,use_o_gate=False,rotary_emb=rope,
                          _fused_qk_norm=norm,attn=layer,_fused_o_norm_gate=consume,o_proj=lambda x:(consume(x),None))
    fb=NS(seq_lens_cpu=lengths,forward_mode=harness.Mode.EXTEND,batch_size=count,dcp_kv_mask=None,
          extend_prefix_lens_cpu=list(prefixes),extend_seq_lens_cpu=list(rows),out_cache_loc=writes[0],out_cache_loc_is_physical=False,
          positions=seq, global_num_token_non_padded_cpu=live,mha_return_lse=False,_attn_output=None,req_pool_indices_cpu=None)
    env["get_forward_batch"]=lambda:fb
    fx.initialize(backend,pool,fb,tables,writes,count,lengths,rows,None)
    if guard=="writer":
        pool.set_kv_buffer=pool.set_kv_buffer
    if guard=="norm_hook":
        norm.register_forward_pre_hook(lambda m,a:None)
    dense=[]
    def fake_fa(**kw):
        dense.append(kw)
        result=kw["out"] if kw["out"] is not None else kw["q"].bfloat16()
        return result.fill_(sentinel)
    backend.flash_attn_varlen_func=fake_fa
    pipe=dict(torch=torch,triton=NS(cdiv=lambda a,b:(a+b-1)//b),_pack_prompt_rows=fx.KernelBoundary())
    fn=define("python/sglang/kernels/ops/attention/batched_prompt_kv.py","pack_prompt_rows",pipe,child)
    modules[fx.PACK]=module(fx.PACK,pack_prompt_rows=fn)
    observed=[]
    native=type(backend).forward
    def observe(self, query,k,v,*args,**kw):
        observed.append((query.dtype,k.dtype,v.dtype,query.data_ptr()))
        return native(self,query,k,v,*args,**kw)
    type(backend).forward=observe
    with torch.inference_mode(),patch.dict(sys.modules,modules),patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True),patch.object(torch.cuda,"is_current_stream_capturing",return_value=False):
        token=capvar.set(cap)
        try:
            with graph["prefill_graph_scope"](full_graph=guard=="full_graph",raw_num_tokens=None):
                out=model.forward(seq,qkv,fb)
        finally:
            capvar.reset(token)
        assert len(cap.cuda_graph._break_fns)==1 and phase==["end","begin"]
        assert len(stores)==1 and len(dense)+len(paged)==1
        dense.clear(); paged.clear(); stores.clear()
        cap.cuda_graph._segments=[NS(replay=lambda:None),NS(replay=lambda:consume(out))]
        with graph["prefill_graph_scope"](full_graph=guard=="full_graph",raw_num_tokens=live):
            cap.cuda_graph.replay()
    assert len(stores)==1 and len(dense)+len(paged)==1
    assert observed[0]==observed[1] and observed[0][1:3]==(torch.bfloat16, torch.bfloat16)
    expected_q=FP8 if child and guard not in ("norm_hook","full_graph") else torch.bfloat16
    assert observed[0][0]==expected_q,(guard,observed,expected_q)
    call=(dense or paged)[0]
    direct = bool(dense) or call.get("page_table") is None
    assert direct==(guard not in ("writer","full_graph")),(guard,count,left,list(call))
    assert call["q"].dtype==FP8 and call["q"].shape==(live,QH,D)
    assert call["out"].dtype==torch.bfloat16 and call["out"].data_ptr()==out.data_ptr()
    assert (out[:live]==sentinel).all() and torch.tensor(sentinel).to(FP8).float()!=sentinel
    assert not torch.count_nonzero(out[live:]) and torch.equal(stores[0][0],writes[int(left>=0)][:live])
    if ks is not None:
        assert torch.equal(call["k_descale"],ks.expand(count,H)) and torch.equal(call["v_descale"],vs.expand(count,H))
    keys=["q"]+(["k","v"] if dense else ["k_cache","v_cache"])
    arrays=[call[key].view(torch.uint8) for key in keys]
    arrays.extend(x.view(torch.uint8) for x in (*stores[0],*pool.full_kv_pool.data,*pool.swa_kv_pool.data))
    return [(x.shape,sha256(x.contiguous().numpy().tobytes()).hexdigest()) for x in arrays], {
        "rows":rows,"prefixes":prefixes,"window":left,"guard":guard,"scales":[.75,1.625] if scale else None,
        "producer_q":str(observed[0][0]),"callback_kv":"torch.bfloat16","fa3_q":"torch.float8_e4m3fn",
        "captured_result":"torch.bfloat16","bf16_only_output_values":out[:live].numel(),
        "producer_to_callback_same_storage":observed[0][-1]==observed[1][-1],
        "fa3_entry":"dense_varlen" if dense else "dense_single" if direct else "paged",
        "compared_bytes":sum(x.numel() for x in arrays),"fa3_q_bytes":arrays[0].numel()}


if __name__=="__main__":
    print(shell_command(), flush=True)
    torch.set_num_threads(1)
    cases=[]
    for rows, prefixes in [([4099],[721]),([2340,1770],[2195,13])]:
        for left in (-1,2047):
            for guard in (None,"writer","norm_hook","full_graph"):
                scale=left>=0 or guard=="writer"
                parent,_=run(False,rows,prefixes,left,guard,scale)
                child,case=run(True,rows,prefixes,left,guard,scale)
                assert parent==child,(rows,prefixes,left,guard)
                cases.append(case)
    result=dict(passed=True,cases=cases,comparisons=len(cases),query_bytes=sum(c["fa3_q_bytes"] for c in cases),
                compared_bytes=sum(c["compared_bytes"] for c in cases),patch_sha256=candidate_digest,
                candidate=manifest.get("candidate",selected.name),candidate_tree=manifest["tree"],
                baseline="V229",baseline_patch_sha256=parent_digest,base=reader.PIN,source_sha256=source_hashes,
                notes="CPU execution of producer dispatch, Triton host wrapper, graph callback recording/replay, metadata, native cache writer and FA3 wrapper. Scalar kernel and FA3 boundaries; H200 numerics/latency unmeasured.")
    receipt_path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("cases", "source_sha256")}))
