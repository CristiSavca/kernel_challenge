"""Exercise the mixed worker on CPU through native V155 head, sampler, accept, draft writer and host packing."""
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as N

import torch

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v155_cpu", root / "v155_mixed_verify_graph_rows/worker_cpu.py")
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
env = ref.env
srt = Path(sys.argv[1]) / "python/sglang/srt"
hashes = {}
def build(file, name, methods=None, *, keep_class=False):
    raw = (srt / file).read_text()
    obj = next(n for n in ast.parse(raw).body if getattr(n, "name", None) == name)
    if methods is not None:
        obj.body = [n for n in obj.body if getattr(n, "name", None) in methods]
        if not keep_class:
            obj.bases = []
            obj.decorator_list = []
    hashes[name] = hashlib.sha256(ast.unparse(obj).encode()).hexdigest()
    exec(compile("from __future__ import annotations\n"+ast.unparse(obj), file, "exec"),env)
    return env[name]

util="managers/utils.py"
env.update(dataclass=dataclasses.dataclass, _is_cuda=False, _is_hip=False,
           _is_npu=False, _use_aiter=False, is_cuda=lambda:True, SIMULATE_ACC_LEN=0,
           _trace_e2e_sampler=lambda *a,**kw:None,maybe_detect_nan=lambda *a:None,
           sanitize_nan_logits=lambda *a:None,SYNC_TOKEN_IDS_ACROSS_TP=False,
           draft_tp_context=lambda _:nullcontext(),DFlashDraftInputV2=N,
           SpecTpSyncSite=N(DFLASH_TARGET="prompt",DFLASH_ACCEPT_GREEDY="accept"),KVWriteLoc=lambda loc,physical:loc)
Result=build(util,"GenerationBatchResult")
Combined=build(util,"MixedDFlashResult")
um=types.ModuleType("sglang.srt.managers.utils")
um.MixedDFlashResult=Combined
sys.modules[um.__name__]=um
build("../kernels/ops/speculative/dflash.py","compute_dflash_correct_drafts_and_bonus")
file="speculative/dflash_worker_v2.py"
build(file,"_commit_accept")
build(file,"_is_all_greedy")
build("speculative/draft_worker_common.py","make_draft_input_v2")
DF=build(file,"DFlashWorkerV2",{"_can_skip_verify_logits","can_settle_mixed_verify","forward_mixed_verify",
    "settle_mixed_verify","_accept_block","_draft_write_ids","_append_target_hidden_to_draft_kv_by_loc",
    "_append_target_hidden_sequential"})
MR=build("model_executor/model_runner.py","ModelRunner",{"sample","_preprocess_logits"})
Sampler=build("layers/sampler.py","Sampler",{"forward","_preprocess_logits","_sync_token_ids_across_tp"})
Pool=build("mem_cache/memory_pool.py","MHATokenToKVPool",{"set_kv_buffer_prefix_valid"})
def pred(ids):
    h,taps=ref.body_tokens(ids)
    logit=25*torch.tanh((h.float()@ref.WEIGHT.T)*.75/25)
    return logit.argmax(-1)

SamplingInfo=ref.build("sampling/sampling_batch_info.py","SamplingBatchInfo",None)

def sampling():
    z=SamplingInfo(temperatures=torch.ones(1), top_ps=torch.ones(1), top_ks=torch.ones(1),
                   min_ps=torch.zeros(1), is_all_greedy=True, is_any_greedy=True,
                   need_top_p_sampling=False, need_top_k_sampling=False, need_min_p_sampling=False,
                   vocab_size=ref.V)
    z.update_regex_vocab_mask=lambda:None
    z.apply_logits_bias=lambda _:None
    return z

def role(batch):
    batch.decoding_reqs=None
    batch.return_hidden_states=batch.has_grammar=False
    batch.sampling_info=sampling()
    for r in batch.reqs:
        r.return_routed_experts=r.return_indexer_topk=r.return_sampling_mask=False
    batch.copy=lambda:copy.copy(batch)
    return batch

def prepare(backend, count, iteration, do_veto=False):
    w=8; lengths=[1+(iteration+i*3)%6 for i in range(1+iteration%3)]
    p=len(lengths);q=sum(lengths);t=count
    mixed=ref.schedule(lengths+[w]*t,offset=iteration)
    prompt=role(ref.schedule(lengths,offset=iteration))
    tail=role(ref.schedule([w]*t))
    tail.forward_mode=ref.Mode.DECODE
    tail.seq_lens=torch.tensor(mixed.prefix_lens[p:],dtype=torch.long)
    tail.seq_lens_cpu=tail.seq_lens.clone()
    mixed.reqs=prompt.reqs+tail.reqs
    z=ref.Mode
    prompt_fb=N(forward_mode=z.EXTEND,batch_size=p,req_pool_indices=prompt.req_pool_indices,
                seq_lens=prompt.seq_lens,extend_num_tokens=q,extend_seq_lens=torch.tensor(lengths,dtype=torch.int32),
                positions=torch.cat([torch.arange(start,start+n) for start,n in zip(prompt.prefix_lens,lengths)]),
                sampling_info=prompt.sampling_info,top_logprobs_nums=None,token_ids_logprobs=None,return_logprob=False)
    desired=[1+(iteration+3*i)%w for i in range(t)]
    for i,live in enumerate(desired):
        chain=[(iteration*3+7*i+1)%ref.V]
        for j in range(1,w):
            next_id=int(pred(torch.tensor([chain[-1]])).item())
            chain.append((next_id+(j==live))%ref.V)
        mixed.input_ids[q+i*w:q+(i+1)*w]=torch.tensor(chain)
    # Every target row, including each proposal that will be rejected, has a
    # valid private physical slot. None points at the pool's zero sentinel.
    gen=torch.Generator().manual_seed(889+iteration)
    slots=5+torch.randperm(3*(q+t*w),generator=gen)[:q+t*w]
    mixed.out_cache_loc=slots
    target,entries,_=ref.test_worker(backend,4096)
    runner=target.model_runner
    runner.sampling_observer=runner.watermark_state=None
    runner.device=torch.device("cpu")
    sampler=Sampler()
    runner._preprocess_logits=types.MethodType(MR._preprocess_logits,runner)
    runner.sample=types.MethodType(MR.sample,runner)
    runner.sampler=sampler.forward
    steps=[]
    def record_sample(**kw):
        assert kw['forward_batch'] is prompt_fb
        steps.append("native_prompt_sampler")
    runner.ngram_embedding_manager.update_after_decode=record_sample
    pool=Pool();pool.k_buffer=[torch.full((3*(q+t*w)+6,40),-1234.,dtype=torch.bfloat16)]
    v=pool.k_buffer[0].clone();pool.v_buffer=[v]
    def store(layer,loc,k,v,ks,vs,**kw):
        assert not bool((loc==0).any())
        pool.k_buffer[0][loc]=k.reshape(-1,40)
        pool.v_buffer[0][loc]=v.reshape(-1,40)
    pool.set_kv_buffer=store
    expected_k=pool.k_buffer[0].clone();expected_v=v.clone()
    df=DF();df._is_domino=df._need_mamba_verify_commit=df.use_compact_draft_cache=False
    df.lilicorr=df._selector_sample=None;df.block_size=w;df._block_pos_offsets=torch.arange(w)
    df._tp_sync=N(sync=lambda site,ids:steps.append(site))
    df._use_triton_accept_bonus=df._use_fused_kv_materialize=False
    df._fused_kv_helper=None;df.draft_parallel_layout=None
    attn=N(attn=N(k_scale=None,v_scale=None,layer_id=0),num_kv_heads=1,head_dim=40,
           apply_k_norm=lambda x:x,apply_k_rope=lambda pos,x:x+pos.to(x.dtype)[:,None],
           kv_proj_only=lambda x:(x,x*2))
    df.draft_model=N(project_target_hidden=lambda x:x,prepare_context_hidden_for_kv=lambda layer,x:x,
                     layers=[N(self_attn=attn)])
    translator=object();df.draft_model_runner=N(kv_index_translator=translator,token_to_kv_pool=pool)
    def write_ids(plan,reader):
        assert reader is translator
        return plan.slots
    ref.Plan.write_ids=write_ids
    df.model_runner=runner;df.target_worker=target
    df._wait_for_fused_hicache_load=lambda b:steps.append("hicache")
    callbacks=[]
    def publish(lens):
        callbacks.append(lens)
        steps.append("publish")
    def execute():
        return df.forward_mixed_verify(mixed_batch=mixed, prompt_batch=prompt,
                    decode_batch=tail,prompt_forward_batch=prompt_fb,on_publish=publish)
    if do_veto:
        return df,prompt,tail,mixed,prompt_fb,execute,entries,steps
    mixed_output=execute()
    a,b=mixed_output.prompt_result,mixed_output.decode_result
    assert len(entries)==1 and entries[0].forward_mode==ref.Mode.EXTEND
    assert mixed_output.next_draft_input.new_seq_lens is callbacks[0] is mixed_output.new_seq_lens
    assert a.logits_output.next_token_logits is None and b.logits_output.next_token_logits is None
    assert steps==["hicache","native_prompt_sampler","prompt","accept","publish"]
    chain=mixed.input_ids[q:].view(t,w)
    assert b.accept_lens.tolist()==desired
    prompt_gold=pred(mixed.input_ids[:q])[torch.tensor(lengths).cumsum(0)-1]
    bonus=pred(chain[torch.arange(t),torch.tensor(desired)-1])
    assert torch.equal(a.next_token_ids,prompt_gold)
    assert torch.equal(mixed_output.next_draft_input.bonus_tokens,torch.cat([prompt_gold,bonus]))
    assert torch.equal(b.new_seq_lens,tail.seq_lens+torch.tensor(desired))
    assert torch.equal(a.new_seq_lens,prompt.seq_lens)
    assert a.new_seq_lens.untyped_storage().data_ptr()==b.new_seq_lens.untyped_storage().data_ptr()
    for i,n in enumerate(desired):
        actual=b.next_token_ids.view(t,w)[i,:n]
        assert torch.equal(actual,torch.cat((chain[i,1:n],bonus[i:i+1])))
    _,taps=ref.body_tokens(mixed.input_ids)
    h=torch.cat(taps,dim=-1)
    live=torch.cat((torch.ones(q,dtype=torch.bool),(torch.arange(w)<torch.tensor(desired)[:,None]).flatten()))
    positions=torch.cat((prompt_fb.positions,(tail.seq_lens[:,None]+torch.arange(w)).flatten()))
    expected_k[slots[live]]=(h+positions.to(h.dtype)[:,None])[live]
    expected_v[slots[live]]=(h*2)[live]
    assert torch.equal(pool.k_buffer[0],expected_k) and torch.equal(v,expected_v)
    return q+t*w,int(live.sum()),q

def negative():
    cases=[
        ("df","lilicorr",1),("df","_selector_sample",1),("df","use_compact_draft_cache",True),
        ("df","_is_domino",True),("df","_need_mamba_verify_commit",True),
        ("runner","sampling_observer",object()),("ngram","enabled",True),
        ("par","attn_cp_size",2),("par","attn_dcp_size",2),("par","dp_size",2),("par","pp_size",2),
        ("p","has_grammar",True),("t","return_logprob",True),("p","return_hidden_states",True),
        ("p","beam_tail",1),("t","is_prefill_only",True),("p","decoding_reqs",[1]),
        ("si","is_all_greedy",False),("si","has_custom_logit_processor",True),
        ("si","acc_additive_penalties",torch.zeros(1)),("si","acc_scaling_penalties",torch.zeros(1)),
        ("si","penalizer_orchestrator",N(is_required=True)),("si","logit_bias",torch.zeros(1)),
        ("si","grammar_mask",1),("si","grammars",[object()]),("si","sampling_mask_batch_indices",torch.ones(1)),
        ("si","sampling_support_logprobs_capture_indices",1),("si","has_watermark_candidates",True),
        ("req","return_routed_experts",True),("req","return_indexer_topk",True),
        ("req","return_sampling_mask",True),("req","beam_group",1),
        ("m","out_cache_loc",None),("m","input_ids",None),("m","spec_info",1),
        ("m","hicache_consumer_index",99),("m","reqs",[]),("t","seq_lens_cpu",None),
        ("ctx","sampling_info",object()),("ctx","req_pool_indices",torch.zeros(1)),
        ("ctx","extend_num_tokens",0),("ctx","batch_size",0),
    ]
    for who,name,value in cases:
        df,p,t,m,c,execute,entries,steps=prepare("eager",2,0,True)
        obj={"df":df,"p":p,"t":t,"m":m,"ctx":c,"req":t.reqs[0],"si":p.sampling_info,
             "par":ref.PAR,"runner":df.target_worker.model_runner,"ngram":df.target_worker.model_runner.ngram_embedding_manager}[who]
        old=getattr(obj,name);setattr(obj,name,value)
        assert execute() is None and not entries and not steps,(who,name)
        setattr(obj,name,old)
    return len(cases)

if __name__=="__main__":
    with torch.inference_mode():
        totals=[0]*3; batches=0
        for backend in ("eager","FULL","BREAKABLE"):
            for n in (1,2,4):
                for i in range(8):
                    row=prepare(backend,n,i);totals=[x+y for x,y in zip(totals,row)]
                    batches+=1
        veto=negative()
    print(json.dumps(dict(torch=torch.__version__,h200=False,executions=batches,
                         backends=["eager","FULL body replay emulator","Breakable body replay emulator"],
                         pretarget_declines=veto,actual_rows=totals[0],committed_rows=totals[1],
                         prompt_rows=totals[2],chain_commits=list(range(1,9)),source_hashes=hashes),indent=2))
