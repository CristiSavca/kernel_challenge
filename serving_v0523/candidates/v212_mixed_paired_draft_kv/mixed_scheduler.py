"""Drive the mixed scheduler, two target implementations and five physical draft K/V layers on CPU."""
import ast
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
from functools import lru_cache
from hashlib import sha256
import importlib.util
import inspect
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.modules import module as module_hooks

HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()
ROOT=STAGE/"python/sglang/srt"
V211=HERE.parent/"v211_prompt_expert_tiles_mixed_serving"
PARENT=json.loads((V211/"validation.json").read_text())
assert sha256((V211/"patch.diff").read_bytes()).hexdigest()==PARENT["patch_sha256"]
FROZEN=json.loads((HERE/"validation.json").read_text())
assert sha256((HERE/"patch.diff").read_bytes()).hexdigest()==FROZEN["patch_sha256"]
CHANGED={p.removeprefix("python/sglang/srt/") for p in FROZEN["changed_from_parent"]}
assert CHANGED=={"models/muse_spark.py","speculative/dflash_worker_v2.py","speculative/dflash_mixed_forward.py"}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@lru_cache(None)
def source(version,path):
    tree=(PARENT if version=="V211" else FROZEN)["tree"]
    raw=subprocess.check_output(["git","-C",str(STAGE),"show",tree+":python/sglang/srt/"+path])
    if version=="V212":assert raw==(ROOT/path).read_bytes()
    return raw


def entries(version,path,cls=None):
    body=ast.parse(source(version,path)).body
    if cls:
        body=next(n.body for n in body if getattr(n,"name",None)==cls)
    return {n.name:n for n in body if isinstance(n,ast.FunctionDef)}


def bind(nodes,namespace,filename):
    result=dict(namespace)
    exec(compile("from __future__ import annotations\n"+ast.unparse(ast.Module(nodes,[])),filename,"exec"),result)
    return result


def kv_model(real,rank,hidden):
    layers=[]
    rope=real.Rope()
    rope.head_size=rope.rotary_dim=2
    rope.is_neox_style=True
    phase=torch.arange(2048)%4
    rope.cos_sin_cache=torch.stack((torch.tensor([1,0,-1,0])[phase],torch.tensor([0,1,0,-1])[phase]),dim=-1).float()
    for idx in range(5):
        a=real.Attention()
        a.q_size=a.kv_size=a.head_dim=2
        a.num_kv_heads=1
        a.rotary_emb=rope
        a.v_scale=None
        a.attn=NS(layer_id=idx,k_scale=None,v_scale=None)
        weights=torch.zeros((6,hidden),dtype=torch.bfloat16)
        for col in range(6):
            weights[col,(7*idx+col*3+rank*5)%hidden]=1 if col%2 else -1
        a.qkv_proj=NS(weight=weights,bias=None)
        norm=a.k_norm=real.Norm()
        norm.weight=torch.ones(2,dtype=torch.bfloat16)
        norm.variance_epsilon=1.e-5
        norm.override_orig_dtype=norm.variance_size_override=None
        norm.hidden_size=2
        norm.cast_x_before_out_mul=False
        layers.append(NS(self_attn=a))
    helper=real.Helper(layers,rope,1,2,torch.device("cpu"),max_position_hint=2048)
    return layers,helper


def model_input(ref,ids):
    hidden,taps=ref.body_tokens(ids)
    return hidden,[x.repeat(1,2) for x in taps]


def rank_main(rank,size,folder):
    logging.disable(logging.WARNING)
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(Path(folder)/"init"),rank=rank,
                            world_size=size,timeout=timedelta(seconds=75))
    try:
        ranked(rank,size,folder)
    finally:
        dist.destroy_process_group()


def ranked(rank,size,folder):
    candidates=HERE.parent
    test=load("v212_scheduler",candidates/"v160_mixed_dflash_serving/independent_round.py")
    context=load("v212_context",candidates/"v202_mixed_target_serving/independent_mixed.py")
    vocab=load("v212_vocab",candidates/"v203_mixed_sharded_verify_head/independent_gloo.py")
    real=load("v212_physical",candidates/"v185_paired_prompt_draft_kv/worker_check.py")
    p2p=load("v212_p2p",next(candidates.glob("v113_*/cpu_check.py")))
    ref=test.ref
    spy={}
    test.sched.module("sglang.srt.speculative.dflash_mixed_forward",
                      start_mixed_prefix_readback=test.bridge.env["start_mixed_prefix_readback"])
    mixed_module=sys.modules["sglang.srt.speculative.dflash_mixed_forward"]
    comm=p2p.Comm(rank)
    def gather(dest,value):
        with torch.inference_mode(False):
            parts=[torch.empty(tuple(value.shape),dtype=value.dtype) for _ in range(size)]
            dist.all_gather(parts,value.contiguous().clone())
        dest.view(size,*value.shape).copy_(torch.stack(parts))
        spy["head_gathers"].append(tuple(value.shape))
    group=NS(world_size=size,rank_in_group=rank,pynccl_comm=comm,cpu_group=dist.group.WORLD,all_gather_into_tensor=gather)
    head_ops=vocab.build_processor(group)
    ref.PAR.tp_group=group
    test.admit["dist"]=dist
    old_policy=test.pl.model
    def policy(grids):
        mr=old_policy((list(range(1,2049)),grids[1]))
        test.pl.PARALLEL.tp_group=group
        return mr
    test.pl.model=policy
    projection=context.projection
    projection._supports_shard_norm=lambda *_: True
    def paired(rows,count,group):
        if case=="rank_gemm_oom" and rank==size-1:
            with patch.object(torch,"mm",side_effect=torch.OutOfMemoryError("rank-local GEMM workspace")):
                return projection.project_paired_draft_kv(rows,count,group)
        return projection.project_paired_draft_kv(rows,count,group)
    ref.env.update(get_forward=lambda:NS(set=lambda *_:None,scoped=lambda **_:nullcontext()),
                   project_paired_draft_kv=paired,
                   project_dflash_prefill_shard=projection.project_dflash_prefill_shard)
    original=test.settle.prepare
    tree=ast.parse(inspect.getsource(original))
    stmt=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
              isinstance(n.targets[0],ast.Name) and n.targets[0].id=="lengths")
    stmt.value=ast.Name(id="sizes",ctx=ast.Load())
    make=dict(vars(test.settle))
    exec(compile(ast.fix_missing_locations(tree),"<synthetic lengths>","exec"),make)
    def prepare(backend,tails,iteration,do_veto):
        make["sizes"]=sizes
        result=make["prepare"](backend,tails,iteration,do_veto)
        worker,prompt,decode,schedule=result[:4]
        target=worker.target_worker
        runner=target.model_runner
        model=runner.model
        graph=inspect.getclosurevars(runner.forward).nonlocals["g"]
        processor=model.logits_processor.__self__
        head=context.NativeHead(processor.forward)
        model.logits_processor=head
        marker=torch.nn.Module()
        model._forward_hooks=marker._forward_hooks
        model._forward_pre_hooks=marker._forward_pre_hooks
        draft=context.Draft(worker.draft_model)
        worker.draft_model=draft
        hidden=16 if case=="traffic_veto" else 64
        draft.fc=torch.nn.Linear(80,hidden,bias=False,dtype=torch.bfloat16)
        draft.hidden_norm=context.RMSNorm(hidden)
        with torch.no_grad():
            w=torch.arange(80*hidden).view(hidden,80)
            draft.fc.weight.copy_((((w*11)%17)-8).float()/64)
        layers,helper=kv_model(real,rank,hidden)
        draft.layers=layers
        worker._use_fused_kv_materialize=True
        worker._fused_kv_helper=helper
        if case=="rank_retired" and rank==size-1:
            worker._use_fused_kv_materialize=False
            worker._fused_kv_helper=None
        if case=="head_hook":
            def observe(_module,_args,out):
                spy["observed_width"]=out.next_token_logits.shape[1]
            head.register_forward_hook(observe)
        if case=="rank_norm_hook" and rank==size-1:
            draft.hidden_norm.register_forward_pre_hook(lambda *_:None)
        if case=="rank_unavailable" and rank==size-1:
            comm.available=False
        if case=="value_scale":
            layers[2].self_attn.v_scale=1.25
            # Scaling semantics on CUDA are checked independently by the paired-writer fixture.
            # For each path, compare the same native fused-cache result against V211.
        ctx=(draft,draft.fc,draft.hidden_norm,draft.project_target_hidden,group,marker,head)
        worker._sp_prefill_context=ctx if case!="context_off" else None
        for attr in ("is_deferred_shard","finish_deferred_gather","deferred_argmax"):
            setattr(head,attr,getattr(head_ops,attr))
        worker._target_logits_processor=head
        sw=(ref.V+size-1)//size
        head_ops.defer_verify_gather=True
        head_ops.final_logit_softcapping=25.
        head_ops.vocab_size=ref.V
        processor.do_tensor_parallel_all_gather=True
        processor.defer_verify_gather=True
        processor._can_gather_softcap_fused=lambda *_:True
        processor._gather_softcap_fused=head_ops._gather_softcap_fused
        processor._compute_lm_head=lambda h,w,b,out:torch.nn.functional.pad(
            h.float()@w.T,(0,sw*size-ref.V))[:,rank*sw:(rank+1)*sw].contiguous()
        def shard(whole):
            span=(len(whole)+size-1)//size
            return torch.nn.functional.pad(whole,(0,0,0,span*size-len(whole)))[rank*span:(rank+1)*span]
        def body(ids):
            h,taps=model_input(ref,ids)
            return shard(h),[shard(t) for t in taps]
        def materialize(x,num_tokens):
            with torch.inference_mode(False):
                copies=[torch.empty(tuple(x.shape),dtype=x.dtype) for _ in range(size)]
                dist.all_gather(copies,x.contiguous().clone())
            return torch.cat(copies)[:num_tokens]
        def selected(x,indices,count):
            return materialize(x,count)[indices]
        ref.env["layernorm_sp"]=NS(runs_sp=lambda m:m==ref.Mode.EXTEND,sp_exit_gather=materialize,
                sp_gather_selected_rows=selected,maybe_exit_gather=lambda **kw:(
                    materialize(kw["hidden_states"],len(kw["input_ids"])),kw["hidden_states_before_norm"]))
        graph.layer_model.forward=lambda ids,*args,**kwargs:body(ids)
        graph.backend.replay=lambda key,fb,**kwargs:body(fb.input_ids)
        pool=worker.draft_model_runner.token_to_kv_pool
        buf=[torch.full((pool.k_buffer[0].shape[0],1,2),-999.,dtype=torch.bfloat16) for _ in range(5)]
        pool.k_buffer=buf
        pool.v_buffer=[x.clone() for x in buf]
        did_fail=False
        def store(layer,loc,k,v,*args,**kwargs):
            nonlocal did_fail
            assert (loc>0).all()
            idx=layer.layer_id
            pool.k_buffer[idx][loc]=k
            pool.v_buffer[idx][loc]=v
            if case=="partial_tail" and not did_fail and len(loc)<32 and idx==2:
                did_fail=True
                raise RuntimeError("synthetic failure after committing a third tail layer")
        pool.set_kv_buffer=store
        def expected_rows(raw):
            normalized=draft.hidden_norm(draft.fc(raw))
            kv=(normalized@helper.flat_kv_weight_t).reshape(len(raw),5,2,2)
            return normalized,kv
        native=target.forward_batch_generation
        def target_call(*args,**kwargs):
            fb=kwargs["forward_batch"]
            assert kwargs["batch"] is None and kwargs["is_verify"]
            out=native(*args,**kwargs)
            logits=out.logits_output
            raw=torch.cat(model_input(ref,fb.input_ids)[1],dim=-1)
            normalized,kv=expected_rows(raw)
            projected=logits.dflash_prefill_kv_projected
            done=logits.dflash_prefill_context_done
            expect=kv.reshape(len(raw),20) if projected else (normalized if done else raw)
            assert torch.equal(logits.hidden_states,expect), (version,backend,case,logits.hidden_states.shape)
            spy.update(projected=projected,context=done,head_cols=logits.next_token_logits.shape[1],
                       expected_kv=kv,positions=fb.positions,rows=len(raw))
            return out
        target.forward_batch_generation=target_call
        worker.forward_mixed_verify=lambda *_,**__: (_ for _ in ()).throw(AssertionError("alternate worker entry"))
        spy.update(worker=worker,schedule=schedule)
        return result
    test.settle.prepare=prepare
    snapshots={}
    runs=[]
    for version in ("V211","V212"):
        path="speculative/dflash_mixed_forward.py"
        mix=bind(list(entries(version,path).values()),dict(test.bridge.env,module_hooks=module_hooks,
                 dflash_sp_prefill_scope=projection.dflash_sp_prefill_scope),path)
        mixed_module.try_forward_proposed_mixed=mix["try_forward_proposed_mixed"]
        path="models/muse_spark.py"
        nodes=entries(version,path,"MuseSparkTextForCausalLM")
        # Global layernorm_sp is rebound at each synthetic forward.
        old_namespace=ref.Muse.forward.__globals__
        exec("from __future__ import annotations\n"+ast.unparse(nodes["forward"]),old_namespace)
        ref.Muse.forward=old_namespace["forward"]
        path="speculative/dflash_worker_v2.py"
        nodes=entries(version,path,"DFlashWorkerV2")
        methods=["settle_mixed_verify","_append_target_hidden_to_draft_kv_by_loc","_append_paired_prompt_kv",
                 "_append_target_hidden_sequential","_append_target_hidden_fused","_prefill_kv_helper"]
        base=dict(test.settle.env,logger=logging.getLogger("V212"),_is_npu=False)
        for name in methods:
            if name in nodes:
                setattr(test.settle.DF,name,bind([nodes[name]],base,path)[name])
        cases=["paired","rank_retired","rank_unavailable","head_hook","rank_norm_hook","partial_tail",
               "short","value_scale","context_off","long_unaligned","wide_bucket_padding","traffic_veto","rank_gemm_oom"]
        for backend in ("eager","FULL","BREAKABLE"):
            for j,case in enumerate(cases):
                # Target rows for aligned runs are exactly 16*(8*TP+2), enough to amortize the weight exchange.
                n=128*size+{"long_unaligned":29,"wide_bucket_padding":17}.get(case,32)
                sizes=[n-23,7] if case!="short" else [9,7]
                before=comm.sent
                comm.available=True
                spy.clear()
                spy["head_gathers"]=[]
                with torch.inference_mode():
                    report=test.replay(backend,2,j,j%2==0)
                    worker=spy["worker"]
                    sched=spy["schedule"]
                    q=sum(sizes)
                    commits=torch.tensor([1+(j+3*i)%8 for i in range(2)])
                    accepted=torch.cat((torch.ones(q,dtype=torch.bool),(torch.arange(8)<commits[:,None]).flatten()))
                    slots=sched.out_cache_loc[accepted]
                    kv=spy["expected_kv"]
                    pos=spy["positions"]
                    pool=worker.draft_model_runner.token_to_kv_pool
                    digest=sha256()
                    for layer in range(5):
                        attn=worker.draft_model.layers[layer].self_attn
                        k=attn.apply_k_rope(pos,attn.apply_k_norm(kv[:,layer,0].contiguous())).reshape(-1,1,2)
                        v=kv[:,layer,1].reshape(-1,1,2)
                        for actual,gold in ((pool.k_buffer[layer],k),(pool.v_buffer[layer],v)):
                            wanted=torch.full_like(actual,-999.)
                            wanted[slots]=gold[accepted]
                            assert torch.equal(wanted,actual),(version,backend,case,layer,(wanted!=actual).sum().item())
                            digest.update(actual.view(torch.int16).numpy().tobytes())
                touched_transport=comm.sent>before
                oom_fallback=(version=="V212" and case=="rank_gemm_oom")
                assert touched_transport==(spy["projected"] or oom_fallback)
                eligible_case=(case in ("paired","partial_tail","long_unaligned")
                               or (case=="wide_bucket_padding" and backend!="FULL"))
                expect_pair=eligible_case and version=="V212"
                assert spy["projected"]==expect_pair,(version,backend,case,spy["projected"])
                expected_local=case!="head_hook"
                assert spy["head_cols"]==((ref.V+size-1)//size if expected_local else ref.V)
                if case=="head_hook":assert spy["observed_width"]==ref.V
                sw=(ref.V+size-1)//size
                assert spy["head_gathers"]==([(2,sw),(16,2)] if expected_local else [(18,sw)])
                key=(backend,case)
                record=(report,digest.hexdigest())
                if version=="V212":
                    assert snapshots[key]==record
                else:
                    snapshots[key]=record
                runs.append(dict(version=version,backend=backend,case=case,projected=spy["projected"],
                                 context=spy["context"],head_cols=spy["head_cols"],committed=report[2],
                                 specific_commits=commits.tolist(),failure_after_layer_recovered=case=="partial_tail" and not worker._use_fused_kv_materialize,
                                 rejected=int((~accepted).sum()),wire_bytes=comm.sent-before,
                                 cache_words=sum(t.numel() for t in pool.k_buffer+pool.v_buffer),rows=spy["rows"]))
    if rank==0:
        Path(folder,"result.json").write_text(json.dumps(runs))


def main():
    payload={}
    for size in (4,8):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(rank_main,args=(size,tmp),nprocs=size,join=True)
            cases=json.loads(Path(tmp,"result.json").read_bytes())
        counts=Counter(r["case"] for r in cases if r["projected"])
        payload[size]=dict(scheduler_calls=len(cases),v211_v212_pairs=len(cases)//2,
            paired_reroutes=dict(counts),cache_words_checked_per_rank=sum(r["cache_words"] for r in cases),
            checked_rejected_slots=sum(r["rejected"] for r in cases),
            accepted_lengths=sorted({k for r in cases for k in r["specific_commits"]}),
            tail_partial_write_recoveries=sum(r["failure_after_layer_recovered"] for r in cases),
            after_exchange_workspace_declines=sum(r["case"]=="rank_gemm_oom" and r["wire_bytes"]>0 and not r["projected"] for r in cases),
            examples=[r for r in cases if r["backend"]=="FULL" and r["case"]=="paired"])
    file_hashes={v:{p:sha256(source(v,p)).hexdigest() for p in sorted(CHANGED)} for v in ("V211","V212")}
    print(json.dumps(dict(torch=torch.__version__,actual_gloo=True,physical_H200=False,
                          parent_sha256=PARENT["patch_sha256"],candidate_sha256=FROZEN["patch_sha256"],
                          tested_sources=file_hashes,worlds=payload),indent=2))


if __name__=="__main__":
    main()
