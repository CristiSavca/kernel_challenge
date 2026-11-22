"""Replay the actual original and extracted DFLASH worker bodies with CPU devices."""
import ast
import hashlib
import json
import logging
from functools import lru_cache
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as N
from typing import NamedTuple, Optional, Tuple
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[3]
BASE="297e54fd105e974cdf401e5b4d20253a22c16f90"
PATH="python/sglang/srt/speculative/dflash_worker_v2.py"
original=subprocess.check_output(["git","-C",str(ROOT/"serving_v0523/sglang"),"show",BASE+":"+PATH],text=True)
candidate=(Path(sys.argv[1])/PATH).read_text()
cpu_accept_src=subprocess.check_output(["git","-C",str(ROOT/"serving_v0523/sglang"),"show",
            BASE+":python/sglang/kernels/ops/speculative/dflash.py"],text=True)
W,V,H=8,103,7
torch.set_num_threads(1)
class DraftInput(N):pass
class Mode:
    def is_extend(self):return False
    def is_idle(self):return False
class Plan(N):
    def bind(self,fb,translator):fb.physical=self.write_virtual+translator.base
    def reads_from(self,translator,**kwargs):
        self.events.append("compact read projection")
        return self
    def write_ids(self,translator):return self.write_virtual+translator.base
class Verify(N):
    def prepare_for_verify(self,batch,target,**kwargs):
        target.events.append(("prepare_verify",None if batch.seq_lens_cpu is None else batch.seq_lens_cpu.tolist(),batch.seq_lens_sum,kwargs))
        if getattr(target,"fail_prepare",False):
            raise RuntimeError("injected metadata failure")
        return N(inputs=self.draft_token,positions=self.positions,plan=self.kv_loc_plan,
                 original_global_num_tokens_cpu=[0,len(batch.reqs)] if len(batch.reqs)==4 else None,can_run_decode_cuda_graph=True),None

@lru_cache
def programs(source):
    funcs={"_commit_accept","_resolve_dflash_embedding_module","_is_all_greedy","compute_dflash_correct_drafts_and_bonus"}
    methods={"_ensure_draft_block_buffers","_draft_write_ids","_accept_block","_make_next_draft_input_decode","forward_batch_generation",
             "propose_decode_block","is_current_decode_proposal","consume_decode_proposal","verify_proposed_decode_block"}
    compiled=[]
    for raw in (cpu_accept_src,source):
        for node in ast.parse(raw).body:
            keep=getattr(node,"name",None) in funcs|{"DFlashBlockProposal","DFlashWorkerV2"}
            if not keep:continue
            if node.name=="DFlashWorkerV2":
                node.body=[m for m in node.body if isinstance(m,ast.FunctionDef) and m.name in methods]
            compiled.append(compile("from __future__ import annotations\n"+ast.unparse(node),PATH,"exec"))
    return compiled


def module(source,record):
    def trace(name):return lambda *a,**k:record.append(name)
    def accept_fmt(**kw):
        return kw["candidates"],kw["target_predict"]
    def logits_adjust(**kwargs):record.append("logits adjusted")
    env=dict(torch=torch,NamedTuple=NamedTuple,Optional=Optional,Tuple=Tuple,BaseSpecWorker=object,
        ScheduleBatch=N,KVLocPlan=Plan,DFlashDraftInputV2=DraftInput,SpeculativeAlgorithm=N(DFLASH="draft"),
        CaptureHiddenMode=N(NULL=0,FULL=2),ForwardMode=N(TARGET_VERIFY="target_verify"),
        ForwardBatch=N,DFlashVerifyInput=Verify,GenerationBatchResult=N,_is_npu=False,
        unwrap_lora_layer=lambda x:x,draft_tp_context=lambda g:nullcontext(),
        enable_num_token_non_padded=lambda:True,SIMULATE_ACC_LEN=0,
        is_dflash_sampling_verify_available=lambda:False,get_parallel=lambda:N(attn_dp_enabled=False,tp_rank=0,tp_group=N()),
        SpecTpSyncSite=N(DFLASH_SELECTOR="selector",DFLASH_ACCEPT_GREEDY="greedy"),
        GrammarTree=N(from_linear_chain=lambda t:(record.append("grammar tree"),t)[1]),
        build_grammar_vocab_mask=lambda **k:N(apply=trace("grammar mask")),apply_dflash_verify_logits_adjustments=logits_adjust,
        SpeculativeSamplingMaskCapture=N(from_logits=lambda *a,**k:None),compute_spec_logprobs=trace("logprobs"),
        logger=logging.getLogger("no-output"),target_input_embeddings=lambda m:m.get_input_embeddings(),
        domino_greedy_rollout=lambda **k:next_ids(k["bonus_tokens"]),)
    env["logger"].disabled=True
    def assign(**k):
        row=k["req_pool_indices"];base=k["start_offset"]
        return k["req_to_token"][row[:,None],base[:,None]+torch.arange(W)].flatten()
    env["assign_extend_cache_locs_func"]=assign
    def prep(**k):
        if env["fail_triton"]:raise RuntimeError("injected Triton fallback")
        b=k["bonus_tokens"];p=k["prefix_lens"]
        k["block_ids_out"].fill_(k["mask_token_id"]);k["block_ids_out"][:,0]=b
        k["positions_out"].copy_(p[:,None]+torch.arange(W))
        k["cache_loc_out"].copy_(k["req_to_token"][k["req_pool_indices"][:,None],k["positions_out"]])
    env["_prepare_dflash_draft_block_unchecked"]=prep
    env["propose_lilicorr_block"]=lambda **k:(next_ids(k["anchor"]), None,None)
    for code in programs(source):exec(code,env)
    env["make_draft_input_v2"]=lambda **k:N(**k)
    return env["DFlashWorkerV2"](),env

def next_ids(roots):
    return (roots[:,None]+torch.arange(1,W))%V

def run(source,mode,folded,bs,compact,prepare,cpu_kind,*,external=False,action=None):
    events=[];w,env=module(source,events)
    table=torch.randperm(20*80,generator=torch.Generator().manual_seed(13)).reshape(20,80)
    members=torch.arange(bs-1,-1,-1)+2;prefix=torch.arange(bs,dtype=torch.int32)+13
    roots=torch.arange(bs)+2;pred=next_ids(roots);candidates=torch.cat((roots[:,None],pred),dim=1)
    sampled=mode in ("sampled_selector","sampled_lilicorr")
    env["get_parallel"]=lambda:N(attn_dp_enabled=bs==4,tp_rank=0,tp_group=N())
    sample=N(is_all_greedy=not sampled,temperature=torch.tensor([1.0]))
    batch=N(forward_mode=Mode(),is_extend_in_batch=False,seq_lens=prefix,
        seq_lens_cpu=prefix.clone() if cpu_kind=="resolved" else None,
        seq_lens_sum=int(prefix.sum()) if cpu_kind=="resolved" else None,req_pool_indices=members,
        reqs=[N(kv=N(kv_allocated_len=int(p)+2*W)) for p in prefix],sampling_info=sample,
        has_grammar=bs%2==0,return_logprob=bs>=4,can_run_decode_cuda_graph=folded)
    draft=DraftInput(bonus_tokens=roots,nxt_kv_lens_cpu=prefix+2*W if cpu_kind!="allocated" else None,
                     nxt_kv_lens_sum=int((prefix+2*W).sum()),max_top_k=1,uniform_top_k_value=1)
    batch.spec_info=draft
    weights=torch.arange(V*H).reshape(V,H).to(torch.bfloat16)
    target_model=N(get_input_embeddings=lambda:lambda ids:weights[ids],lm_head=N(weight=weights))
    def plan(**kw):
        events.append(("plan",kw["read_extent"],kw["write_virtual"].tolist()))
        return Plan(**kw,events=events)
    def target(batch,forward_batch,**kwargs):
        events.append(("target",forward_batch.inputs.tolist(),forward_batch.positions.tolist(),forward_batch.can_run_decode_cuda_graph,kwargs))
        ids=forward_batch.inputs.reshape(bs,W);ids2=(ids+1)%V
        length=(torch.arange(bs)+3)%W
        ids2[torch.arange(bs),length]=(ids2[torch.arange(bs),length]+5)%V
        logits=torch.full((bs,W,V),-4.0)
        logits.scatter_(2,ids2[:,:,None],4.0)
        return N(logits_output=N(next_token_logits=logits.reshape(-1,V),hidden_states=weights[ids].reshape(-1,H)),
                 can_run_cuda_graph=folded,routed_experts_output=None,indexer_topk_output=None)
    w.target_worker=N(model_runner=N(model=target_model,kv_index_translator=N(plan=plan)),
                          forward_batch_generation=target,events=events)
    w.model_runner=N(req_to_token_pool=N(req_to_token=table),model_config=N(vocab_size=V))
    w.draft_model=N(prefix_gru=N(),embed_proj=N(),shift_label=False)
    w._noise_embed_scale=1.25
    w._full_embed_gpu=weights if compact else None
    w._draft_block_ids_buf=None;w._pending_proposal_token=None;w.block_size=W;w.device="cpu"
    w._block_pos_offsets=torch.arange(W);w._mask_token_id=3;w._use_triton_prepare_block=prepare!="eager"
    env["fail_triton"]=prepare=="raise"
    w.use_compact_draft_cache=compact
    w._compute_compact_draft_seq_lens=lambda vals:torch.clamp(vals,max=3)
    w._fill_compact_seq_lens_cpu_bound=lambda **k:k["out"].fill_(3+W)
    w._rebuild_compact_draft_cache=lambda **k:events.append(("compact",k["draft_prefix_lens"].tolist()))
    w.draft_parallel_layout=N();w._draft_block_spec_info=N()
    w.selector=N() if "selector" in mode else None;w.lilicorr=N() if "lilicorr" in mode else None
    w._is_domino=mode=="domino";w.domino_candidate_pool_size=2
    w._selector_sample=None;w._selector_sampling_enabled=True;w._lilicorr_sampling_enabled=mode=="sampled_lilicorr"
    w._warned_lilicorr_eager=False;w._lilicorr_anchor=roots;w._target_logits_processor=None
    w._need_mamba_verify_commit=bs==2;w._use_triton_accept_bonus=False
    w._update_target_mamba_state_after_verify=lambda **k:events.append(("mamba",k["seq_lens_pre_verify"].tolist(),k["seq_lens_post_verify"].tolist()))
    if folded:
        w._target_logits_processor=N(is_deferred_shard=lambda x:True,
            deferred_argmax=lambda x:(events.append("deferred argmax"),x.argmax(-1))[1],
            finish_deferred_gather=lambda x:(events.append("vocabulary gather"),x)[1])
        w._can_skip_verify_logits=lambda b,s: not b.has_grammar and not b.return_logprob and not sampled
    w._validate_phase1_sampling_support=lambda b:None
    def q_data():
        return torch.stack((pred,(pred+2)%V),dim=-1),torch.full((bs,W-1,2),.5)
    if mode=="sampled_lilicorr":
        env["propose_lilicorr_block"]=lambda **kw:(pred,*q_data())
    w._draft_sampler=N(out=pred.flatten().clone(),candidate_out=q_data()[0],q_out=q_data()[1],
         stage_sampling_params=lambda **k:events.append(("stage sampling",k["bs"]))) if folded else None
    def draft_forward(fb):
        events.append(("draft",fb.input_ids.tolist(),fb.positions.tolist(),fb.physical.tolist(),
                       fb.seq_lens.tolist(),fb.seq_lens_cpu.tolist(),fb.seq_lens_sum,fb.input_embeds.float().sum().item()))
        return N(logits_output=N(hidden_states=weights[candidates].reshape(-1,H)),can_run_graph=folded)
    w.draft_model_runner=N(kv_index_translator=N(base=7),forward=draft_forward)
    def propose_selector(**kw):
        if sampled:w._selector_sample=q_data()
        return pred
    w._propose_selector_block=propose_selector
    w._greedy_sample_from_vocab_parallel_head=lambda **k:pred.flatten()
    def sampling_accept(**kw):
        events.append(("selector correction",kw["candidate_ids"].tolist(),kw["q_rows"].tolist()))
        return env["compute_dflash_correct_drafts_and_bonus"](candidates=kw["candidates"],
                    target_predict=kw["next_token_logits"].argmax(-1).view(bs,W))
    w._selector_sampling_accept=sampling_accept
    w._tp_sync=N(sync=lambda site,t:events.append(("tp",site,t.tolist())))
    w._append_target_hidden_to_draft_kv_by_loc=lambda **k:events.append(("draft KV",k["cache_loc_2d"].tolist(),k["commit_lens"].tolist(),k["positions"].tolist()))
    publish=lambda x:events.append(("publish",x.tolist()))
    if action is not None:return action(w,batch,table)
    if external:
        pkt=w.propose_decode_block(batch)
        assert pkt.tokens.shape==pkt.positions.shape==pkt.virtual_slots.shape==(bs,W)
        assert pkt.virtual_slots.is_contiguous() and pkt.kv_loc_plan.write_virtual.data_ptr()==pkt.virtual_slots.data_ptr()
        assert pkt.prefix_lens is batch.seq_lens and pkt.prefix_lens_cpu is batch.seq_lens_cpu
        assert pkt.source_batch is batch and pkt.draft_input is draft and pkt.selector_sample is w._selector_sample
        assert torch.equal(pkt.virtual_slots,table[members[:,None],prefix[:,None]+torch.arange(W)])
        assert not any(e=="grammar tree" or isinstance(e,tuple) and e[0] in ("publish","target","draft KV") for e in events)
        result=w.verify_proposed_decode_block(batch,pkt,on_publish=publish)
    else:result=w.forward_batch_generation(batch,on_publish=publish)
    assert torch.equal(result.new_seq_lens,prefix+result.accept_lens)
    assert (batch.seq_lens_cpu is None)==(cpu_kind!="resolved")
    assert batch.seq_lens_sum==(int(prefix.sum()) if cpu_kind=="resolved" else None)
    return events,result.accept_lens.tolist(),result.next_token_ids.tolist(),result.next_draft_input.bonus_tokens.tolist()


def ownership(w,b,table):
    p=w.propose_decode_block(b)
    assert w.is_current_decode_proposal(b,p)
    foreign=type(w)();foreign._pending_proposal_token=object()
    assert not foreign.is_current_decode_proposal(b,p)
    q=w.propose_decode_block(b)
    assert not w.is_current_decode_proposal(b,p) and w.is_current_decode_proposal(b,q)
    import copy
    clone=copy.copy(b)
    assert not w.is_current_decode_proposal(clone,q)
    for field in ("seq_lens","seq_lens_cpu","req_pool_indices","spec_info"):
        saved=getattr(b,field)
        setattr(b,field,copy.copy(saved))
        assert not w.is_current_decode_proposal(b,q)
        setattr(b,field,saved)
    assert w.is_current_decode_proposal(b,q)
    w.consume_decode_proposal(b,q)
    assert not w.is_current_decode_proposal(b,q)
    try:w.verify_proposed_decode_block(b,q)
    except ValueError:pass
    else:raise AssertionError("duplicate verification")
    t=w.propose_decode_block(b);cpu=b.seq_lens_cpu;total=b.seq_lens_sum
    w.target_worker.fail_prepare=True
    try:w.verify_proposed_decode_block(b,t)
    except RuntimeError as ex:assert "injected metadata failure" in str(ex)
    else:raise AssertionError("missing injected error")
    assert b.seq_lens_cpu is cpu and b.seq_lens_sum==total and not w.is_current_decode_proposal(b,t)
    return True


if __name__=="__main__":
    count=0
    with torch.inference_mode(), patch.object(torch.Tensor,"record_stream",lambda self,stream:None):
        for i,mode in enumerate(("head","selector","sampled_selector","domino","lilicorr","sampled_lilicorr")):
            for folded in (False,True):
                for compact in (False,True):
                    for j,prepare in enumerate(("eager","success","raise")):
                        for k,cpu in enumerate(("resolved","upper","allocated")):
                            bs=(1,2,4)[(i+j+k)%3]
                            args=(mode,folded,bs,compact,prepare,cpu)
                            expected=run(original,*args)
                            assert run(candidate,*args)==expected,(args,"ordinary")
                            assert run(candidate,*args,external=True)==expected,(args,"external then verify")
                            count+=1
        assert run(candidate,"sampled_selector",True,4,False,"eager","resolved",action=ownership)
    print(json.dumps(dict(torch=torch.__version__,base=BASE,original_sha256=hashlib.sha256(original.encode()).hexdigest(),
             worker_sha256=hashlib.sha256(candidate.encode()).hexdigest(),cases=count,
             compared_ordinary_worker_steps=count,compared_external_then_verify_steps=count,
             sampler_paths=["direct head","selector","sampled selector","Domino","LiLiCorr","sampled LiLiCorr"],
             execution=["folded graph","eager"],context=["ordinary","compact"],
             prepare=["native fallback","recorded Triton success","injected Triton failure"],
             host_lens=["committed mirror","allocator safe bound","per-request allocated length"],
             batch_sizes=[1,2,4],identical_observations=["GRAMMAR and TP order","selector IDs/probability pair",
             "draft inputs, positions, slots, host sum, device prefix","all target inputs","commits/bonus/published lengths",
             "prefix-filtered draft KV writer inputs","host length restoration","DP idle-peer graph fallback","Mamba commit","deferred argmax/vocabulary gather"],
             ownership="old generation, other batch, replaced CPU/GPU lens or req/spec info, duplicate and metadata failure checked",
             device_substitutions="CPU stand-ins for CUDA streams, physical kernels, model execution, attention and communications"),indent=2))
