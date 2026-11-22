"""Quantify cross-round draft overlap using pinned source, CPU model examples, and archived public H200 timings."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from typing import Tuple
import torch

ROOT=Path(__file__).resolve().parents[3]
S=ROOT/'serving_v0523'; PIN='297e54fd105e974cdf401e5b4d20253a22c16f90'
PARENT=S/'candidates/v198_segmented_verify_prompt_experts'
PATCH=(PARENT/'patch.diff').read_bytes()
assert hashlib.sha256(PATCH).hexdigest()=='fa91157f4e5e3c9704664047ba177af4ddabf2f5d2ab0af2560e1320902a59e4'
def src(name):
    f='python/sglang/'+name
    base=subprocess.check_output(['git','-C',str(S/'sglang'),'show',PIN+':'+f]).splitlines(keepends=True)
    for seg in PATCH.split(b'diff --git ')[1:]:
        rows=seg.splitlines(keepends=True)
        if rows[0].decode().rstrip()!=f'a/{f} b/{f}':continue
        out=[];i=0;inhunk=False
        for row in rows[1:]:
            if row.startswith(b'@@'):
                n=int(re.match(rb'@@ -(\d+)(?:,\d+)? \+',row).group(1))-1
                assert n>=i;out.extend(base[i:n]);i=n;inhunk=True
            elif inhunk:
                if row[:1] in (b' ',b'-'):
                    assert row[1:]==base[i];i+=1
                if row[:1] in (b' ',b'+'):out.append(row[1:])
        return b''.join(out+base[i:]).decode()
    return b''.join(base).decode()


def fn(code, name, cls=None):
    nodes=ast.parse(code).body
    if cls is not None:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls).body
    n=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'torch':torch,'Tuple':Tuple}
    exec(compile('from __future__ import annotations\n'+ast.unparse(n),name,'exec'),env)
    return env[name]


def dependencies():
    paths=['srt/speculative/dflash_worker_v2.py','srt/models/dflash.py','kernels/ops/speculative/dflash.py',
           'srt/speculative/draft_parallel.py','srt/speculative/spec_utils.py']
    worker,model,accept_src,layout,scope=[src(p) for p in paths]
    n=next(x for x in ast.walk(ast.parse(worker)) if isinstance(x,ast.FunctionDef) and x.name=='forward_batch_generation')
    body=ast.get_source_segment(worker,n)
    # Bind the native order in the combined parent. Positions / physical aliases are populated before draft replay.
    order=['bonus_tokens=draft_input.bonus_tokens.view(-1)','self.draft_model_runner.forward(forward_batch)',
           'self.target_worker.forward_batch_generation(','commit_lens=commit_lens','hidden = logits_output.hidden_states',
           'self._append_target_hidden_to_draft_kv_by_loc(','bonus_tokens=bonus,']
    pos=body.index('prefix_lens = batch.seq_lens');loc=[]
    for key in order:pos=body.index(key,pos);loc.append(n.lineno+body[:pos].count('\n'));pos+=len(key)
    assert 'DraftParallelLayout(tp_group=None, attn_tp_size=parallel.attn_tp_size)' in layout
    assert 'if layout.tp_group is None:' in scope and 'return contextlib.nullcontext()' in scope
    assert 'input_embeds = self.embed_tokens(input_ids)' in model and 'self.layers' in model and 'class DFlash2DraftModel(DFlashDraftModel)' in model
    accept=fn(accept_src,'compute_dflash_correct_drafts_and_bonus');commit=fn(worker,'_commit_accept')
    project=fn(model,'project_target_hidden','DFlashDraftModel')
    B,W,H=19,8,16
    inp=torch.arange(B*W).view(B,W)+1
    same=inp[:,1:].clone()
    ans=[]
    for bonus in (47,71):
        target=torch.cat((same,torch.full((B,1),bonus)),dim=1)
        accepted,next_token=accept(candidates=inp,target_predict=target)
        out,count=commit(inp,accepted,next_token)
        assert torch.all(count==8) and torch.all(next_token==bonus)
        ans.append((out,next_token))
    assert torch.equal(ans[0][0][:,:7],ans[1][0][:,:7]) and torch.all(ans[0][1]!=ans[1][1])
    # The already known seven accepted tokens give neither the next bonus nor its initial embedding.
    embedding=torch.nn.Embedding(256,H,dtype=torch.bfloat16)
    with torch.no_grad():
        embedding.weight.zero_();embedding.weight[47,0]=1;embedding.weight[71,1]=1
    assert torch.all((embedding(ans[0][1])!=embedding(ans[1][1])).any(1))
    # Both worlds use the same first four tap blocks and the same target bonus. Varied last tap
    # changes the real full-width FC->normalization call and hence valid first-layer draft attention.
    fc=torch.nn.Linear(5*H,H,bias=False,dtype=torch.bfloat16)
    with torch.no_grad():
        fc.weight.zero_();fc.weight[:,:H]=torch.eye(H,dtype=torch.bfloat16)
        fc.weight[:,4*H:]=torch.eye(H,dtype=torch.bfloat16)
    m=SimpleNamespace(fc=fc,quantizable=False,hidden_norm=torch.nn.RMSNorm(H,eps=1e-6,dtype=torch.bfloat16))
    results=[]
    for sign in (-1,1):
        taps=torch.zeros(B*W,5*H,dtype=torch.bfloat16);taps[:,0]=1;taps[:,4*H+1]=sign
        v=project(m,taps).reshape(B,W,H);k=torch.zeros_like(v)
        prev=torch.zeros((B,31,H),dtype=torch.bfloat16);q=torch.zeros((B,8,H),dtype=torch.bfloat16)
        # All eight freshly committed rows are attended inside the published 2,048-row draft window.
        merged=torch.nn.functional.scaled_dot_product_attention(q[:,None],torch.cat((prev,k),1)[:,None],
            torch.cat((prev,v),1)[:,None],is_causal=False)
        results.append(merged)
    assert torch.all(results[0][:,:,:,1]<0) and torch.all(results[1][:,:,:,1]>0)
    return dict(source_sha256={p:hashlib.sha256(src(p).encode()).hexdigest() for p in paths},
                V198_decode_order=dict(zip(order,loc)),full_accept_with_distinct_real_bonus_rows=B,
                same_bonus_distinct_last_tap_queries=B*W,torch=torch.__version__)


def rates():
    scored=json.loads((S/'runs/abd217baf0ed49fd.json').read_text())['result']['rungs']
    v68=json.loads((S/'runs/04b01f958aa5487c.json').read_text())['result']['rungs'][0]
    table=[]
    for idx,x in [('V64',r) for r in scored]+[('V68',v68)]:
        a=x['acceptance'];B=x['concurrency'];upper=(a-1)/7
        lo,hi=0.,1.
        for _ in range(65):
            q=(lo+hi)/2
            if sum(q**i for i in range(8))<a:lo=q
            else:hi=q
        constant=lambda d:1/(1-upper*d)
        table.append(dict(candidate=idx,batch=B,acceptance=a,p50_tps=x['tok_s_per_user_p50'],
                   period_estimate_ms=1000*a/x['tok_s_per_user_p50'],full_probability_upper=upper,
                   stationary_iid_token_match=q,illustrative_full_probability=q**7,
                   independent_requests_upper_all_full=upper**B,
                   stationary_iid_full_batch_probability=q**(7*B),
                   fixed_cost_ideal_gain={str(d):constant(d) for d in [.1,.25,.5,1.]}))
    # Frequency does not bound time share when the duration of a draft depends on acceptance.
    hits=3/7;weighted_good=(hits*10)/(hits*10+(1-hits)*.01+0.1)
    assert weighted_good>.97
    return dict(rungs=table,correlated_example_time_fraction=weighted_good)


def split_proxy():
    records=[]
    for p in sorted((ROOT/'experiments/candidates/muse_decode_layer/profiles').glob('*.json')):
        d=json.loads(p.read_text())
        if not d['summary'].get('passed'):continue
        out={}
        for x in d['benchmarks']:
            case=x['case'];b=case['batch']
            if case['load']!=f'verify {b}' or b not in (16,32):continue
            if not all(k in x for k in ('median_raw_ns','baseline_raw_ns','replay_overhead')):continue
            layer=case['layer'];w=case['weight'];assert w==(16 if layer=='full' else 46)
            over=x['replay_overhead']['overhead_ns']
            out.setdefault(b,[]).append((w*(x['median_raw_ns']-over)/1e6,w*(x['baseline_raw_ns']-over)/1e6))
        if set(out)!={16,32} or any(len(a)!=2 for a in out.values()):continue
        dense={b:[sum(v) for v in zip(*ls)] for b,ls in out.items()}
        records.append(dict(job=p.stem.split('.')[-1],source_sha256=p.stem.split('.')[0],
                 b16_candidate_ms=dense[16][0],b32_candidate_ms=dense[32][0],b16_baseline_ms=dense[16][1],
                 b32_baseline_ms=dense[32][1],extra_split_candidate_ms=2*dense[16][0]-dense[32][0],
                 extra_split_baseline_ms=2*dense[16][1]-dense[32][1]))
    assert len(records)>5 and all(x['extra_split_candidate_ms']>0 and x['extra_split_baseline_ms']>0 for x in records)
    return records


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.inference_mode():
        print(json.dumps(dict(parent='V198',parent_sha256=hashlib.sha256(PATCH).hexdigest(),
            dependencies=dependencies(),bound=rates(),split_proxy=split_proxy(),
            first_layer_prefix=dict(old_rows=2047,rank_kv_heads=1,rank_query_heads=8,dim=128,
                potential_mask_only_rows=6,possible_query_positions=13,extra_old_kv_bytes_per_request=2047*2*128*2,
                useful_mask_old_attention_flops_per_request=4*6*8*128*2047,old_kv_still_needed_for_root=True),
            physical_new_run=False,patch_exported=False),indent=2))
