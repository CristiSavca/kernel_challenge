"""Run from the challenge repository with CPU PyTorch. No SGLang checkout edits."""
import ast
import hashlib
import json
import subprocess
from enum import IntEnum, auto
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
PARENT = "7a3d4dc6c0e526b60ca50bc975a6036da7796d8eb05f9e8afb70423164425437"
PARENT_PATCH = (ROOT / "serving_v0523/candidates/v147_integrated_experts_prefill/patch.diff").read_bytes()
assert hashlib.sha256(PARENT_PATCH).hexdigest() == PARENT
SRT = "srt/"
NATIVE = {}


@lru_cache
def source(path):
    fullname = "python/sglang/" + path
    assert f"diff --git a/{fullname} ".encode() not in PARENT_PATCH
    return subprocess.check_output(
        ["git", "-C", str(ROOT / "serving_v0523/sglang"), "show", PIN + ":" + fullname],
        text=True,
    )


def load(path, node_name, owner=None, **bindings):
    tree = ast.parse(source(path))
    body = tree.body if owner is None else next(n.body for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in body if getattr(n, "name", None) == node_name)
    text = ast.get_source_segment(source(path), node)
    NATIVE[f"{path}/{node_name}"] = hashlib.sha256(text.encode()).hexdigest()
    code = ast.Module(body=[node], type_ignores=[])
    env = dict(torch=torch, IntEnum=IntEnum, auto=auto, **bindings)
    exec(compile("from __future__ import annotations\n" + ast.unparse(code), path, "exec"), env)
    return env[node_name]


mode = load(SRT+"model_executor/forward_batch_info.py", "ForwardMode")
native_accept = load("kernels/ops/speculative/dflash.py", "compute_dflash_correct_drafts_and_bonus")
native_commit = load(SRT+"speculative/dflash_worker_v2.py", "_commit_accept")
native_project = load(SRT+"models/dflash.py", "project_target_hidden", "DFlashDraftModel")
native_reserve = load(SRT+"mem_cache/allocation_sizing.py", "page_aligned_decode_alloc_lens")
native_prune = load(SRT+"layers/logits_processor.py", "_get_pruned_states", "LogitsProcessor")
native_mix = load(SRT+"managers/schedule_batch.py", "mix_with_running", "ScheduleBatch", ForwardMode=mode)
native_resolve = load(SRT+"managers/overlap_utils.py", "resolve_mixed_spec_tails", "FutureMap")
pcg = SRT+"model_executor/runner/prefill_cuda_graph_runner.py"
native_bufrows = load(pcg, "_prefill_logits_buffer_rows", "PrefillCudaGraphRunner")
native_trim = load(pcg,"_trim_logits_output","PrefillCudaGraphRunner",LogitsProcessorOutput=NS)
res = SRT+"managers/scheduler_components/batch_result_processor.py"
stride = load(res,"_get_speculative_output_stride")
native_result = load(res,"_resolve_spec_v2_tokens","SchedulerBatchResultProcessor",_get_speculative_output_stride=stride)

attn = SRT+"layers/attention/torch_native_backend.py"
native_sdpa = load(attn,"_run_sdpa_forward_extend","TorchNativeAttnBackend",
                   scaled_dot_product_attention=F.scaled_dot_product_attention)
native_mask = load(attn,"_make_sliding_window_mask","TorchNativeAttnBackend")

W, H, V = 8, 12, 43
DT = torch.bfloat16
torch.set_num_threads(1)


def concat(x):
    return torch.cat(x, dim=0)


def split(tensor, lens):
    return list(tensor.split(lens, dim=0))


class SmallTarget:
    """Real SDPA, five layer taps, physical FP8 KV, finite full-vocabulary head.

    The weights are synthetic and shared by both executions. FP64 SDPA inputs
    isolate logical masking from hardware reduction order, with BF16 rows
    preserved across model layer boundaries.
    """
    def __init__(self):
        torch.manual_seed(173)
        self.embed = torch.randn(V,H).to(DT)
        self.w = [torch.randn(3,H,H).to(DT)/4 for _ in range(5)]
        self.head = torch.randn(V,H).to(DT)
        self.draft = NS(fc=torch.nn.Linear(5*H,H,bias=False,dtype=DT),
                        hidden_norm=torch.nn.RMSNorm(H,dtype=DT),quantizable=False)
        self.kv = [torch.randn(2*H,H).to(DT)/3 for _ in range(5)]
        self.table = torch.randperm(5*64).reshape(5,64)
        self.swa_xlate = torch.randperm(self.table.numel())

    def new_cache(self):
        return torch.full((5,2,self.table.numel(),H), float("nan"),dtype=torch.float32)

    def body(self, ids, prefixes, inputs, cache, *, crosscheck_native=False):
        lens = [len(x) for x in inputs]
        owner = concat([torch.full((n,),rid) for rid,n in zip(ids,lens,strict=True)])
        pos = concat([torch.arange(p,p+n) for p,n in zip(prefixes,lens,strict=True)])
        slots = self.table[owner,pos]
        slots_k = concat([self.table[r,:p+n] for r,p,n in zip(ids,prefixes,lens,strict=True)])
        owner_k = concat([torch.full((p+n,),r) for r,p,n in zip(ids,prefixes,lens,strict=True)])
        pos_k = concat([torch.arange(p+n) for p,n in zip(prefixes,lens,strict=True)])
        x = self.embed[concat(inputs)]
        taps = []
        for layer,w in enumerate(self.w):
            q,k,v = [F.linear(x,wi) for wi in w]
            # Quantize the actual KV that both paths will read, not hidden activations.
            dst_slots = self.swa_xlate[slots] if layer % 2 else slots
            src_slots = self.swa_xlate[slots_k] if layer % 2 else slots_k
            cache[layer,0,dst_slots] = k.to(torch.float8_e4m3fn).float()
            cache[layer,1,dst_slots] = v.to(torch.float8_e4m3fn).float()
            visible = (owner[:,None] == owner_k) & (pos[:,None] >= pos_k)
            if layer % 2:
                visible &= pos[:,None] - pos_k <= 4
            ctx = F.scaled_dot_product_attention(q.double()[None,None],
                cache[layer,0,src_slots].double()[None,None],
                cache[layer,1,src_slots].double()[None,None],attn_mask=visible[None,None])[0,0].to(DT)
            if crosscheck_native:
                native_out = torch.empty((len(q),1,H),dtype=torch.float64)
                native_sdpa(NS(_make_sliding_window_mask=native_mask),q.double()[:,None],native_out,
                     cache[layer,0,:,None],cache[layer,1,:,None],
                     self.swa_xlate[self.table] if layer%2 else self.table,torch.tensor(ids),
                     torch.tensor(prefixes)+torch.tensor(lens),torch.tensor(prefixes),torch.tensor(lens),
                     causal=True,sliding_window_size=4 if layer%2 else None)
                assert torch.equal(native_out[:,0].to(DT),ctx)
            x = F.rms_norm((x+ctx).float(), (H,)).to(DT)
            taps.append(x)
        # A deterministic full-K projection tests all vocabulary entries, including prompt heads.
        return F.linear(x.double(),self.head.double()).float(), torch.cat(taps,dim=-1), slots

    def draft_write(self, taps, slots, dst):
        ctx = native_project(self.draft, taps)
        for layer,w in enumerate(self.kv):
            dst[layer,slots] = F.linear(ctx,w)


def result_tokens(owners,out,committed):
    reqs = [NS(is_retracted=False,finished=lambda:False,grammar=None,kv=NS(kv_committed_len=p),
               spec_verify_ct=0,spec_num_correct_drafts=0,update_spec_correct_drafts_histogram=lambda n:None)
            for p in owners]
    consumer = NS(metrics_reporter=NS(snapshot_spec_decode_metrics=lambda *x:None),
               model_worker=NS(on_verify_complete_cpu=lambda *a,**k:None),advance_grammar_fsm=lambda *x:None)
    result = NS(next_token_ids=out.flatten(),accept_lens=committed,speculative_output_stride=W,
                num_non_draft_tokens_per_req=1,block_accept_lens=None,cap_lens=None)
    return native_result(consumer,result,NS(reqs=reqs)), [r.kv.kv_committed_len for r in reqs]


def one_batch(model,seed,commit_case):
    rng = torch.Generator().manual_seed(seed)
    owners = torch.randperm(5,generator=rng).tolist()
    num_prompts = 1 + (seed % 2)
    prompt_ids, tail_ids = owners[:num_prompts], owners[num_prompts:]
    bases = [int(torch.randint(1,13,(1,),generator=rng)) for _ in owners]
    pbase,tbase = bases[:num_prompts],bases[num_prompts:]
    prefix = [torch.randint(V,(n,),generator=rng) for n in bases]
    prompt = [torch.randint(V,(1+(seed+3*i)%11,),generator=rng) for i in range(num_prompts)]
    roots = [torch.randint(V,(1,),generator=rng) for _ in tail_ids]
    cache = model.new_cache()
    model.body(owners,[0]*len(owners),prefix,cache)
    seq = [x for x in roots]
    tmp = cache.clone()
    for n in range(W):
        logits,_,_ = model.body(tail_ids,[p+n for p in tbase],[x[-1:] for x in seq],tmp)
        seq=[torch.cat((x,y.view(1))) for x,y in zip(seq,logits.argmax(-1),strict=True)]
    proposals=[]
    desired=[]
    for j,x in enumerate(seq):
        c=1+(commit_case+j)%W
        desired.append(c)
        z=x[:W].clone()
        if c<W:
            z[c] = (z[c]+1)%V
        proposals.append(z)
    inputs=prompt+proposals
    lens=[len(x) for x in inputs]
    together_cache=cache.clone()
    together,paid_taps,slots=model.body(owners,bases,inputs,together_cache,crosscheck_native=True)
    by_req_cache=cache.clone()
    pieces=[model.body([r],[p],[x],by_req_cache) for r,p,x in zip(owners,bases,inputs,strict=True)]
    assert torch.equal(concat([z[0] for z in pieces]),together)
    assert torch.equal(concat([z[1] for z in pieces]),paid_taps)
    assert torch.equal(slots,concat([z[2] for z in pieces]))
    assert torch.equal(together_cache.view(torch.int32),by_req_cache.view(torch.int32))
    # Each prompt has one head row, each tail has W. All requests share the single body.
    n_prompt_rows=sum(lens[:num_prompts])
    remap=torch.cat((torch.tensor(lens[:num_prompts]).cumsum(0)-1,
                     torch.arange(n_prompt_rows,sum(lens))))
    heads=together[remap]
    chain=torch.stack(proposals)
    verify=heads[num_prompts:].argmax(-1).view(-1,W)
    correct,bonus=native_accept(candidates=chain,target_predict=verify)
    out,commit=native_commit(chain,correct,bonus)
    assert commit.tolist()==desired
    actual,committed_base=result_tokens(tbase,out,commit)
    assert actual==[z[1:n+1].tolist() for z,n in zip(seq,desired,strict=True)]
    assert committed_base==[p+n for p,n in zip(tbase,desired,strict=True)]
    prompt_bonus=heads[:num_prompts].argmax(-1)
    assert prompt_bonus.tolist()==[int(z[0][-1].argmax()) for z in pieces[:num_prompts]]
    draft=torch.full((5,model.table.numel(),2*H),float("nan"),dtype=DT)
    gold=draft.clone()
    # The prompt is always materialized. Only accepted ORIGINAL tail-input rows go into draft KV.
    valid=torch.cat((torch.ones(n_prompt_rows,dtype=torch.bool),(torch.arange(W)<commit[:,None]).flatten()))
    model.draft_write(paid_taps[valid],slots[valid],draft)
    for i,(r,p,part) in enumerate(zip(owners,bases,pieces,strict=True)):
        n=lens[i] if i<num_prompts else desired[i-num_prompts]
        model.draft_write(part[1][:n],model.table[r,p:p+n],gold)
    assert torch.equal(draft.view(torch.int16),gold.view(torch.int16))
    # Poison rejected target slots. Next attention must see accepted prefix and new target bonus only.
    poisoned=together_cache.clone()
    for layer in range(5):
        rejected = slots[~valid]
        poisoned[layer,:,model.swa_xlate[rejected] if layer % 2 else rejected]=float("nan")
    nextx=[x.view(1) for x in bonus]
    next_joint,_,_=model.body(tail_ids,committed_base,nextx,poisoned)
    isolated=[model.body([r],[p],[x],by_req_cache)[0] for r,p,x in zip(tail_ids,committed_base,nextx,strict=True)]
    assert torch.equal(next_joint,concat(isolated))
    return len(owners),len(tail_ids),sum(lens),len(remap),sum(desired),int((~valid).sum())


def existing_boundaries(model):
    # One prompt and two running requests. Actual overlap advances tails by different lengths.
    prompt,ids = [4],torch.tensor([3,1])
    reqs=[NS(seqlen=base+1,refresh_fill_ids=lambda:None) for base in (9,13)]
    pool=NS(req_to_token=model.table)
    spec=NS(is_none=lambda:False)
    mix=NS(seq_lens=torch.tensor([6]),seq_lens_cpu=torch.tensor([6]),req_to_token_pool=pool,
           out_cache_loc=model.table[4,2:6],mamba_track_indices=None,spec_algorithm=spec,
           prefix_lens=[2],extend_lens=prompt.copy(),extend_num_tokens=4,extend_logprob_start_lens=[0])
    def merge(run):
        mix.seq_lens=torch.cat((mix.seq_lens,run.seq_lens))
        mix.seq_lens_cpu=None
    mix.merge_batch=merge
    running=NS(reqs=reqs,batch_size=lambda:len(reqs),req_pool_indices=ids,
               req_pool_indices_cpu=ids.clone(),seq_lens=torch.tensor([9,13]))
    native_mix(mix,running)
    assert mix.extend_lens==[4,1,1] and mix.seq_lens.tolist()==[6,10,14]
    relay=torch.zeros(5,dtype=torch.long)
    relay[ids]=torch.tensor([15,14])
    native_resolve(NS(new_seq_lens_buf=relay,req_to_token=model.table,publish_ready=None,
                      fwd_prepare_d2h_stream=None),mix)
    assert mix.prefix_lens==[2,15,14] and mix.seq_lens.tolist()==[6,16,15]
    assert torch.equal(mix.out_cache_loc[-2:],model.table[ids,relay[ids]])
    assert len(mix.out_cache_loc)==6 and len(mix.out_cache_loc)!=4+2*W
    # Allocator reserve already covers all 8 real rows even with up to 8-token unresolved prior commit.
    reserve_cases=0
    for old in range(3,12):
        for lag in range(W+1):
            for slack in (0,2,9,16,23):
                _,nxt,needed=native_reserve([NS(kv=NS(kv_committed_len=old,kv_allocated_len=old+slack))],
                                            reserve=2*W,page_size=1)
                assert old+lag+W<=nxt[0] and needed==nxt[0]-old-slack
                reserve_cases+=1
    actual_lens=[4,W,W]
    rows=torch.arange(sum(actual_lens)).reshape(-1,1)
    meta=NS(forward_mode=mode.MIXED,extend_return_logprob=False,extend_seq_lens=torch.tensor(actual_lens))
    pruned=native_prune(None,rows,None,None,meta)[0][:,0].tolist()
    assert pruned==[3,11,19]
    meta.forward_mode=mode.TARGET_VERIFY
    meta.draft_extend_select_index=None
    assert torch.equal(native_prune(None,rows[4:],None,None,meta)[0],rows[4:])
    want=[3]+list(range(4,20))
    assert native_bufrows(None,NS(return_logprob=False,batch_size=3))==3
    output=NS(next_token_logits=torch.tensor(want),hidden_states=rows,last_hidden_states=None,
              mm_input_embeds=None,input_token_logprobs=None,input_top_logprobs_val=None,
              input_top_logprobs_idx=None,input_token_ids_logprobs_val=None,input_token_ids_logprobs_idx=None,
              input_logprobs_copy_done=None,customized_info=None,hidden_states_token_indices=None)
    rows_per_backend={}
    for name,full in (("FULL",True),("BREAKABLE",False)):
        runner=NS(model_runner=NS(spec_algorithm=NS(is_speculative=lambda:True)),
                  raw_bs=3,raw_num_tokens=len(rows),_is_full_backend=full)
        trimmed=native_trim(runner,output)
        rows_per_backend[name]=len(trimmed.next_token_logits)
        assert len(trimmed.hidden_states)==20
    assert rows_per_backend=={"FULL":3,"BREAKABLE":17}
    return dict(real_prompt_rows=4,stale_tail_bases=[9,13],fresh_tail_bases=relay[ids].tolist(),
                single_slot_tail_destinations=mix.out_cache_loc[-2:].tolist(),
                native_mixed_rows=mix.extend_lens,needed_rows=actual_lens,legacy_logit_indices=pruned,
                required_logit_indices=want,prefill_shared_logit_buffer_rows=3,
                native_full_and_breakable_logit_trim_rows=rows_per_backend,
                page_one_double_reserve_cases=reserve_cases)


if __name__=="__main__":
    with torch.inference_mode():
        model=SmallTarget()
        samples=[one_batch(model,seed,k) for seed in range(1,9) for k in range(W)]
        labels=("request_rows","verified_requests","target_token_rows","required_logit_rows",
                "committed_tail_rows","discarded_tail_rows")
        print(json.dumps(dict(torch=torch.__version__,sglang=PIN,v147_patch=PARENT,
                  cases=len(samples),native_sdpa_layer_comparisons=5*len(samples),totals=dict(zip(labels,map(sum,zip(*samples)),strict=True)),
                  boundaries=existing_boundaries(model),native_function_hashes=NATIVE),indent=2))
