"""Exercise both target methods against the V120b parent with recorded FA3 calls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

BACKEND = "python/sglang/srt/layers/attention/flashattention_backend.py"
MUSE = "python/sglang/srt/models/muse_spark.py"
PAGED = "python/sglang/srt/mem_cache/layout/paged_view.py"
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
PARENT = "ea427b45fdbf5d363687107f02e5659bed2caea5"


class Hopper(torch.Tensor):
    # CPU storage, ordinary PyTorch shape/stride/dtype operations.
    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cuda(self):
        return True


def method(source, name, cls=None):
    tree = ast.parse(source)
    if cls:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "from __future__ import annotations\n" + ast.unparse(fn)


def src(stage, file, tree=None):
    if tree:
        return subprocess.check_output(["git", "-C", str(stage), "show", tree + ":" + file], text=True)
    return (stage / file).read_text()


def counter():
    return NS(prepare=[], fa=[], nodes=[])


def lengths(args):
    return (tuple(args["cache_seqlens"].tolist()), tuple(args["cu_seqlens_q"].tolist()))


def prepare(trace, **kw):
    assert "cu_seqlens_k_new" not in kw and "cu_seqlens_k" not in kw
    plan = NS(args=kw, live=None, semaphore=0)
    def node():
        plan.live = lengths(kw)
        plan.semaphore = 0
    node()
    trace.nodes.append(node)
    trace.prepare.append(plan)
    return plan


def attention(trace, **kw):
    trace.fa.append(kw)
    plan = kw.get("scheduler_metadata")
    if plan is not None:
        def node():
            assert plan.live == lengths(kw)
            assert plan.semaphore == 0 and plan.args["window_size"] == kw["window_size"]
            assert plan.args["max_seqlen_k"] == kw["page_table"].shape[1] * kw["k_cache"].shape[1]
            assert plan.args["max_seqlen_q"] == kw["max_seqlen_q"]
            assert plan.args["cache_seqlens"] is kw["cache_seqlens"]
            assert plan.args["cu_seqlens_q"] is kw["cu_seqlens_q"]
            plan.semaphore += 1
            plan.semaphore = 0  # simulate native split combine
        node()
        trace.nodes.append(node)
    result = kw["q"]
    if kw["return_softmax_lse"]:
        return result, torch.empty(result.shape[1], result.shape[0])
    return result


class Mode:
    EXTEND = 1
    def __init__(self, verify=True): self.verify = verify
    def is_target_verify(self): return self.verify


def build(stage, tree=None):
    source = src(stage, BACKEND, tree)
    trace = counter()
    env = dict(torch=torch, ForwardMode=Mode, AttentionType=NS(ENCODER_ONLY="enc", DECODER_BIDIRECTIONAL="bi"),
               is_cp_active=lambda fb: False, flash_attn_with_kvcache=lambda **kw: attention(trace, **kw))
    for name in ["paged_view", "paged_kv_view"]:
        exec(method(src(stage, PAGED, BASE), name), env)
    methods = ["get_paged_mha_kv_cache", "prepare_paged_mha_query", "forward_extend", "_mxfp8_sf_kwargs"]
    if tree is None: methods.append("_verify_schedule")
    for name in methods: exec(method(source, name, "FlashAttentionBackend"), env)
    backend = type("Backend", (), {k: env[k] for k in methods})()
    backend.__dict__.update(fa_impl_ver=3, _pack_swa_fp8_prefill=True, _verify_schedule_sms=132,
                            use_mla=False, local_attn_builder=None, topk=1, fa_skip_kv_cache=False,
                            kv_cache_is_mxfp8=False, use_sliding_window_kv_pool=True,
                            kv_cache_dtype=torch.float8_e4m3fn, kv_cache_dtype_str="fp8_e4m3", num_splits=0)
    backend._get_scheduler_metadata = lambda **kw: prepare(trace, **kw)
    exec(method(src(stage, MUSE, tree), "forward", "MuseSparkTextForCausalLM"), env)
    owner = NS(_share_verify_schedule=True, capture_aux_hidden_states=False, lm_head=None,
               logits_processor=lambda ids, hid, head, fb, aux: hid)
    return backend, owner, env["forward"], trace


def tensors(b, w, page, cols, qheads, kvheads):
    def device(x): return x.as_subclass(Hopper)
    table = device(torch.zeros((b, cols), dtype=torch.int32))
    def meta(short=0):
        return NS(page_table=table, max_seq_len_q=w,
                  cache_seqlens_int32=device(torch.arange(b,dtype=torch.int32)+600-short),
                  cu_seqlens_k=device(torch.arange(b+1, dtype=torch.int32)*700),
                  cu_seqlens_q=device(torch.arange(b+1, dtype=torch.int32)*w))
    md=meta()
    md.swa_page_table=table.clone(); md.image_token_ranges=None; md.swa_spec_metadata=None
    md.swa_out_cache_loc=None
    fb=NS(batch_size=b, forward_mode=Mode(), spec_info=NS(num_tokens_per_req=w,ragged_verify_layout=None),
          _fa3_verify_schedules=None, dcp_kv_mask=None)
    layer=NS(tp_q_head_num=qheads,tp_k_head_num=kvheads,tp_v_head_num=kvheads,
             head_dim=64,v_head_dim=64,layer_id=0,is_cross_attention=False,attn_type="decoder",
             logit_cap=0.0,scaling=0.125,k_scale=device(torch.tensor(.75)),v_scale=device(torch.tensor(.5)))
    q=device(torch.zeros(b*w,qheads,64,dtype=torch.float8_e4m3fn))
    kv=tuple(device(torch.zeros(page,kvheads,64,dtype=torch.float8_e4m3fn)) for _ in range(2))
    return md, fb, layer, q, kv, meta


def signatures(trace):
    def describe(t):
        return (t.data_ptr(), tuple(t.shape),tuple(t.stride()),str(t.dtype)) if isinstance(t,torch.Tensor) else t
    return [{k: describe(v) for k,v in call.items() if k != "scheduler_metadata"} for call in trace.fa]


def main(stage):
    current = build(stage)
    parent = build(stage,PARENT)
    stream = NS(cuda_stream=111)
    totals = dict(pairs=0, qualified_62_layer_calls=0, qualified_prepares=0, excluded_layers=0,
                  recorded_execution_replays=0)
    scenarios = {"static","alternate_swa","streams","short_table","short_window","half_wave",
                 "ragged","wrong_width","cross_query","q_head","cap","mask","flag","compile","sinks","lse","mode",
                 "wrong_device", "strided_lengths","hidden_scope","view_rows"}
    with patch.object(torch.cuda,"current_device",lambda: 0), patch.object(torch.cuda,"current_stream",lambda _:stream):
        for page in (1,160):
            for case in sorted(scenarios):
                for b in ((1,4,16,32) if case=="static" else (4,)):
                    qheads,kvheads,width=16,2,8
                    if case=="half_wave": b=34
                    if case=="q_head": qheads=24
                    cols=(400 if case=="short_table" else 16384)//page
                    md,fb,layer,q,kv,make_meta=tensors(b,width,page,cols,qheads,kvheads)
                    if case=="alternate_swa": md.swa_spec_metadata=make_meta(9)
                    if case=="ragged": fb.spec_info.ragged_verify_layout=object()
                    if case=="wrong_width": fb.spec_info.num_tokens_per_req=7
                    if case=="cross_query": q=q[:-1]
                    if case=="view_rows": q=q[:,:qheads//2].contiguous()
                    if case=="cap": layer.logit_cap=3.0
                    if case=="mask": fb.dcp_kv_mask=True
                    if case=="mode": fb.forward_mode=Mode(False)
                    if case=="strided_lengths": md.cache_seqlens_int32=md.cache_seqlens_int32.repeat_interleave(2)[::2]
                    for backend,owner,forward,trace in (current,parent):
                        trace.fa.clear();trace.prepare.clear();trace.nodes.clear()
                        backend.page_size=page;backend.forward_metadata=md
                        backend.token_to_kv_pool=NS(get_kv_buffer=lambda _:kv)
                        owner._share_verify_schedule=case!="flag"
                        def body(ids,pos,batch,input_embeds=None):
                            for i in range(62):
                                stream.cuda_stream = (111+i%2) if case=="streams" else 111
                                layer.sliding_window_size=(-1 if i%4==0 else 3 if case=="short_window" else 2047)
                                extra={"sinks":q} if case=="sinks" else {"return_lse":True} if case=="lse" else {}
                                output=backend.forward_extend(q,None,None,layer,batch,**extra)
                            return output
                        owner.model=body
                        with patch.object(torch.compiler, "is_compiling",lambda: case=="compile"), \
                             patch.object(torch.cuda, "current_device",lambda: 1 if case=="wrong_device" else 0):
                            if case=="hidden_scope": body(q,None,fb)
                            else: forward(owner,q,None,fb)
                        assert fb._fa3_verify_schedules is None
                    assert signatures(current[3])==signatures(parent[3]),case
                    tc=current[3]
                    count=sum("scheduler_metadata" in call for call in tc.fa)
                    expected=62 if case in ("static","streams","alternate_swa") else 16 if case=="short_window" else 0
                    assert count==expected,(case,count,expected)
                    expected_plans=3 if case=="streams" else 2 if expected==62 else 1 if expected>0 else 0
                    assert len(tc.prepare)==expected_plans,(case,len(tc.prepare),expected_plans)
                    if count:
                        for i in range(4):
                            md.cache_seqlens_int32.add_(i+1)
                            if md.swa_spec_metadata: md.swa_spec_metadata.cache_seqlens_int32.add_(i+11)
                            for node in tc.nodes: node()
                            totals["recorded_execution_replays"]+=1
                    if case=="static":
                        totals["qualified_62_layer_calls"]+=1;totals["qualified_prepares"]+=len(tc.prepare)
                    totals["excluded_layers"]+=62-count;totals["pairs"]+=1
        # The model always restores the surrounding scope on exception or recursion.
        outer={"already":"owned"};fb._fa3_verify_schedules=outer
        owner,forward=current[1:3];owner._share_verify_schedule=True;fb.forward_mode=Mode()
        def explode(*args,**kwargs):
            assert args[2]._fa3_verify_schedules=={}
            raise LookupError("synthetic")
        owner.model=explode
        try: forward(owner,q,None,fb)
        except LookupError: pass
        else: raise AssertionError("lost exception")
        assert fb._fa3_verify_schedules is outer
    return dict(passed=True, torch=torch.__version__, complete_parent_and_candidate_methods=True,
                cases=sorted(scenarios), **totals, bounds="CPU recording, modeled stream order and semaphore",
                source_sha256={p:hashlib.sha256(src(stage,p).encode()).hexdigest() for p in (MUSE,BACKEND)})


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",type=Path)
    print(json.dumps(main(p.parse_args().stage),indent=2))
