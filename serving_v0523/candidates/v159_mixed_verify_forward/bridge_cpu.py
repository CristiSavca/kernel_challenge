"""Compare a native prepared combined forward with per-role outputs and declined-proposal ownership."""
import ast
from datetime import timedelta
import tempfile
import hashlib
import importlib.util
import json
import sys
from copy import copy
from pathlib import Path
from types import SimpleNamespace as N
from typing import NamedTuple, Optional, Tuple
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1])
def module(name, file):
    spec=importlib.util.spec_from_file_location(name,file)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
v156=module("v156",here.parent/"v156_mixed_verify_settlement/device_cpu.py")
v155=v156.ref
pl=module("v157",here.parent/"v157_mixed_graph_admission/cpu_check.py")
pl.load()
path=stage/"python/sglang/srt/speculative"
def functions(file, env):
    tree=ast.parse(file.read_text())
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    exec(compile("from __future__ import annotations\n"+ast.unparse(ast.Module(body=body,type_ignores=[])),str(file),"exec"),env)
class VI(pl.DFlashVerifyInput):
    def __init__(self,**kw):
        super().__init__(**kw,topk=1,custom_mask=None,ragged_verify_layout=None,num_tokens_per_req=kw['draft_token_num'])
pl.DFlashVerifyInput=VI;pl.ForwardMode=v155.Mode;pl.CaptureHiddenMode=v155.Capture
functions(path/"dflash_mixed_graph_admission.py",pl.__dict__)
env=dict(torch=torch,copy=copy,json=json,hashlib=hashlib,ForwardMode=v155.Mode,ForwardBatch=v155.FB,CaptureHiddenMode=v155.Capture,
    DFlashVerifyInput=VI,MixedDFlashResult=v156.Combined, get_exec=v155.env['get_exec'],
    get_schedule=lambda:pl.SETTINGS,plan_mixed_dflash_target=pl.plan_mixed_dflash_target,agree_mixed_dflash_target=pl.agree_mixed_dflash_target)
env.update(PrefillCudaGraphRunner=pl.PrefillCudaGraphRunner,DecodeCudaGraphRunner=pl.DecodeCudaGraphRunner)
functions(path/"dflash_mixed_forward.py",env)
v156.env.update(NamedTuple=NamedTuple,Tuple=Tuple,Optional=Optional)
proposal_cls=v156.build("speculative/dflash_worker_v2.py","DFlashBlockProposal")
ownership=v156.build("speculative/dflash_worker_v2.py","DFlashWorkerV2",{"is_current_decode_proposal","consume_decode_proposal"})
def run(backend,t,i,cpu_mirror,veto=None):
    w,p,d,old,ctx,_,entries,steps=v156.prepare(backend,t,i,True)
    runner=w.target_worker.model_runner
    policy=pl.model((list(range(1,129)),[1,2,3,4,8]))
    for nm in ("prefill_cuda_graph_runner","decode_cuda_graph_runner"):
        graph=getattr(policy,nm);setattr(runner,nm,graph);graph.model_runner=runner
    runner.spec_algorithm.is_dflash=lambda:True
    runner.req_to_token_pool=policy.req_to_token_pool
    runner.page_size=1;runner.model_config.is_encoder_decoder=False;runner.ngram_embedding_manager.engram_hasher=None
    runner.spec_algorithm.is_ngram=lambda:False
    for b in (p,d):
        b.req_pool_indices+=1
        if b is d:b.req_pool_indices+=len(p.reqs)
        b.req_pool_indices_cpu=b.req_pool_indices.clone()
        for rid,req in zip(b.req_pool_indices.tolist(),b.reqs):
            req.rid=str(rid);req.origin_input_ids=[];req.finished=lambda:False;req.is_retracted=False
            req.kv=N(req_pool_idx=rid,kv_committed_len=0,kv_allocated_len=64)
        b.orig_seq_lens=b.seq_lens.int();b.sampling_info.grammars=None;b.has_grammar=False
    q=p.extend_num_tokens; p.out_cache_loc=old.out_cache_loc[:q];p.input_ids=old.input_ids[:q]
    for r,n in zip(d.reqs,d.seq_lens.tolist()): r.kv.kv_committed_len=n
    if not cpu_mirror:d.seq_lens_cpu=None
    d.spec_info=object()
    slots=old.out_cache_loc[q:].reshape(t,8); positions=d.seq_lens[:,None]+torch.arange(8)
    base=runner.kv_index_translator;oldplan=base.own_plan
    def mk(fb):
        x=oldplan(fb);x._source=base;return x
    base.own_plan=mk
    plan=v155.Plan(N(out_cache_loc=slots.flatten()));plan.req_pool_indices=d.req_pool_indices;plan._source=base
    token=w._pending_proposal_token=object()
    proposal=proposal_cls(old.input_ids[q:].view(t,8),positions,d.seq_lens,d.seq_lens_cpu,slots,plan,None,d.spec_info,d,token)
    w.is_current_decode_proposal=ownership.is_current_decode_proposal.__get__(w)
    w.consume_decode_proposal=ownership.consume_decode_proposal.__get__(w)
    # The CPU runtime's renderer asserts saved prompt identity in V156's own
    # fixture. V159 constructs a private copy of the context, so verify its metadata here.
    def sampled(**kw):
        f=kw["forward_batch"]
        assert f.req_pool_indices is p.req_pool_indices and f.seq_lens is p.seq_lens
        assert f.sampling_info is not p.sampling_info
        steps.append("native_prompt_sampler")
    runner.ngram_embedding_manager.update_after_decode=sampled
    if veto=="no_bucket":runner.prefill_cuda_graph_runner.backend._graphs={}
    if veto=="no_headroom":d.reqs[0].kv.kv_allocated_len=int(d.seq_lens[0])+8
    if veto=="sampling":d.sampling_info.is_all_greedy=False
    if veto=="cost":pl.SETTINGS.max_prefill_tokens=q+8*t-1
    if veto=="stale":w._pending_proposal_token=object()
    if veto=="rank_disagrees":env["agree_mixed_dflash_target"]=lambda *a,**kw:None
    pids=p.input_ids.clone();base_lens=d.seq_lens.clone();original=(d.seq_lens_cpu,d.spec_info,p.sampling_info,p.prefix_lens,p.extend_lens)
    callbacks=[]
    try:
        result=env["try_forward_proposed_mixed"](w,p,d,proposal,on_publish=callbacks.append)
    finally:
        env["agree_mixed_dflash_target"]=pl.agree_mixed_dflash_target
    assert torch.equal(p.input_ids,pids) and torch.equal(d.seq_lens,base_lens)
    assert all(a is b for a,b in zip(original,(d.seq_lens_cpu,d.spec_info,p.sampling_info,p.prefix_lens,p.extend_lens)))
    if veto:
        assert result is None and not entries and not callbacks and not steps,(veto,steps)
        assert w.is_current_decode_proposal(d,proposal)==(veto!="stale")
        if veto!="stale":w.consume_decode_proposal(d,proposal)
        return
    assert result is not None,(q,t,backend)
    assert not w.is_current_decode_proposal(d,proposal) and len(entries)==1
    f=entries[0]; assert f.forward_mode==v155.Mode.EXTEND and f.capture_hidden_mode==v155.Capture.FULL
    assert torch.equal(f.input_ids[q:],proposal.tokens.flatten()) and torch.equal(f.positions[q:],proposal.positions.flatten())
    assert torch.equal(f.kv_loc_plan.slots,old.out_cache_loc)
    assert callbacks[0] is result.new_seq_lens and len(callbacks)==1
    desired=[1+(i+3*n)%8 for n in range(t)]
    a,b=result.prompt_result,result.decode_result
    assert b.accept_lens.tolist()==desired
    prompt_gold=v156.pred(p.input_ids)[torch.tensor(p.extend_lens).cumsum(0)-1]
    assert torch.equal(a.next_token_ids,prompt_gold)
    bonus=v156.pred(proposal.tokens[torch.arange(t),torch.tensor(desired)-1])
    assert torch.equal(result.next_draft_input.bonus_tokens,torch.cat((prompt_gold,bonus)))
    assert torch.equal(b.new_seq_lens,d.seq_lens+torch.tensor(desired)) and torch.equal(a.new_seq_lens,p.seq_lens)
    assert a.logits_output.next_token_logits is None and b.logits_output.next_token_logits is None
    return q,t

def standalone_fallback():
    v158=module("v158",here.parent/"v158_native_dflash_proposal/proposal_cpu.py")
    runs=0
    # V158's native worker simulator exercises both the draft and verify entries.
    # An unsupported sampled cohort must reach the ordinary target exactly once.
    for folded in (False,True):
        for t in (1,2,4):
            args=("sampled_selector",folded,t,False,"eager","resolved")
            expected=v158.run(v158.candidate,*args)
            def declined(w,b,table):
                calls=[]
                standalone=w.target_worker.forward_batch_generation
                def target(*a,**kw):
                    calls.append("target")
                    return standalone(*a,**kw)
                w.target_worker.forward_batch_generation=target
                proposal=w.propose_decode_block(b)
                w.can_settle_mixed_verify=lambda p,d:False
                assert env["try_forward_proposed_mixed"](w,object(),b,proposal) is None
                assert not calls and w.is_current_decode_proposal(b,proposal)
                r=w.verify_proposed_decode_block(b,proposal,on_publish=lambda x:calls.append("publish"))
                assert calls==["target","publish"] and not w.is_current_decode_proposal(b,proposal)
                return r.accept_lens.tolist(),r.next_token_ids.tolist(),r.next_draft_input.bonus_tokens.tolist()
            assert v158.run(v158.candidate,*args,action=declined)==tuple(expected[1:])
            runs+=1
    return runs

def gloo_worker(rank,size,uri):
    dist.init_process_group("gloo",rank=rank,world_size=size,init_method=uri,timeout=timedelta(seconds=40))
    pl.PARALLEL=N(tp_group=N(world_size=size,cpu_group=dist.group.WORLD))
    plan=pl.DFlashMixedGraphPlan(64,16,64)
    try:
        for local,signature,expected in [(plan,91,True),(None,-1,False),(plan,92,False),
                         (pl.DFlashMixedGraphPlan(64,16,80),91,False)]:
            answer=pl.agree_mixed_dflash_target(plan if rank==0 else local,cohort_signature=(1,2,24,91,3,4,5) if rank==0 else (1,2,24,signature,3,4,5))
            assert (answer is not None)==expected
    finally:
        dist.destroy_process_group()

if __name__=="__main__":
    n=0;true_mirror=0
    with torch.inference_mode():
        for graph in ("FULL","BREAKABLE"):
            for tails in (1,2,4):
                for it in range(8):
                    for host in (False,True):
                        run(graph,tails,it,host);n+=1;true_mirror+=host
        failures=["no_bucket","no_headroom","sampling","cost","stale","rank_disagrees"]
        for v in failures:run("FULL",2,2,False,v)
        with patch.object(torch.Tensor,"record_stream",lambda *args:None):
            fallback=standalone_fallback()
    for world in (2,4,8):
        with tempfile.TemporaryDirectory() as temp:
            mp.start_processes(gloo_worker,(world,"file://"+str(Path(temp)/"gloo")),nprocs=world,start_method="fork",join=True)
    print(json.dumps(dict(exec=n,cpu_mirror_present=true_mirror,gpu_only_prefix_emulated=n-true_mirror,declines=failures,
            actual_standalone_fallbacks=fallback,gloo_world_sizes=[2,4,8],gloo_patterns=4,native_targets_per_accept=1,native_targets_per_decline=0,physical_H200=False),indent=2))
