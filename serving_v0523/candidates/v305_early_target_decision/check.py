"""Test exact global-prefix certificates and committed draft-KV preparation on V302 source."""
import ast
from collections import Counter
from hashlib import sha256
import itertools
import json
from types import SimpleNamespace as NS
import torch
from source import CP, ROOT, PARENT, sources


def load(src,name,env):
    obj=next(m for m in ast.walk(ast.parse(src)) if isinstance(m,ast.FunctionDef) and m.name==name)
    obj.decorator_list=[]
    exec("from __future__ import annotations\n"+ast.unparse(obj),env)
    return env[name]


def capped_page_certificate(post,rank_count=8,page=4):
    # post is numeric or +inf for first-NaN precedence. Read equal prefixes
    # within existing contiguous TP shards. Stop only with the smallest possible
    # winning ID. An unread earlier shard can still supply a tied winner.
    width=post.numel()//rank_count
    def seen(n):return [t for t in range(post.numel()) if t%width<n]
    for n in range(page,width+1,page):
        ids=seen(n)
        top=max(post[i].item() for i in ids)
        winner=next(i for i in ids if post[i].item()==top)
        if n==width or (top==1.0 and winner< n):
            return winner,n


def check():
    torch.set_num_threads(1)
    with sources() as get:
        paths=["srt/speculative/dflash_worker_v2.py","srt/layers/logits_processor.py",
               "kernels/ops/speculative/dflash.py","kernels/ops/speculative/fused_kv_materialize.py",
               "srt/model_executor/runner/decode_cuda_graph_runner.py","kernels/ops/sampling/softcap_row_argmax.py"]
        w,lp,ops,fused,runner,local=[get(p) for p in paths]
    src_hash={p:sha256(s.encode()).hexdigest() for p,s in zip(paths,(w,lp,ops,fused,runner,local))}
    assert "verify_fc.capture(out, forward_batch)" in runner
    assert w.index("target_out = self.target_worker.forward_batch_generation(")<w.index("target_predict = target_lp.deferred_argmax(",w.index("def verify_proposed_decode_block("))
    assert "on_publish(new_seq_lens)" in w and "token_to_kv_pool.set_kv_buffer_prefix_valid(" in w and "FIRST_NAN" in local
    accept=load(ops,"compute_dflash_correct_drafts_and_bonus",dict(torch=torch))
    commit=load(w,"_commit_accept",dict(torch=torch))
    width=16; ranks=8; vocab=width*ranks; page=4
    state=torch.tensor(list(itertools.product((False,True),repeat=7)))
    b=len(state); rows=b*8
    gen=torch.Generator().manual_seed(305)
    raw=torch.randn((rows,vocab),generator=gen)
    for i in range(rows):
        if i%8!=0:raw[i,(17*i)%vocab]=100.
        if i%3==0:raw[i,(43*i)%vocab]=100.
    post=torch.tanh(raw)
    expected=post.argmax(1).view(b,8)
    def local_select(z,cap,valid,offset,first_nan=False):
        score,idx=torch.tanh(z[:,:valid]/cap).mul(cap).max(1)
        if first_nan:score=torch.where(score.isnan(),torch.inf,score)
        return torch.stack((score,(idx+offset).float()),dim=1)
    proc=NS(vocab_size=vocab,final_logit_softcapping=1.)
    parts=[local_select(raw[:,i*width:(i+1)*width],1.,width,i*width) for i in range(ranks)]
    receive=torch.stack(parts);grp=NS(world_size=ranks,rank_in_group=0)
    def gather(out,local):
        assert torch.equal(local,parts[grp.rank_in_group]);out.copy_(receive.view(-1,2))
    grp.all_gather_into_tensor=gather
    head=load(lp,"deferred_argmax",dict(torch=torch,get_parallel=lambda:NS(tp_group=grp),softcap_row_argmax=local_select))
    for i in range(ranks):
        grp.rank_in_group=i
        assert torch.equal(head(proc,raw[:,i*width:(i+1)*width]).view(b,8),expected)
    result=[capped_page_certificate(s,ranks,page) for s in post]
    got=torch.tensor([v for v,_ in result]).view(b,8)
    assert torch.equal(got,expected)
    proposal=torch.cat((torch.zeros(b,1,dtype=torch.long),torch.where(state,expected[:,:7],(expected[:,:7]+1)%vocab)),1)
    def decoded(decision):
        n,bonus=accept(candidates=proposal,target_predict=decision)
        return commit(proposal,n,bonus)
    original,n=decoded(expected); certified,n2=decoded(got)
    assert torch.equal(n,n2) and torch.equal(original,certified) and set(n.tolist())==set(range(1,9))
    for row, (answer,lim) in zip(post,result):
        if lim<width:assert answer<lim and row[answer].item()==1.
    # The observed first pages are identical. A later lower ID can tie a
    # saturated nonzero-rank winner, or later beat any finite unsaturated one.
    satA=torch.zeros(vocab);satA[width]=1.;satB=satA.clone();satB[page]=1.
    plainA=torch.zeros(vocab);plainA[0]=.5;plainB=plainA.clone();plainB[page]=.75
    visible=lambda x:x.view(ranks,width)[:,:page]
    assert torch.equal(visible(satA),visible(satB)) and torch.equal(visible(plainA),visible(plainB))
    assert [capped_page_certificate(x)[0] for x in (satA,satB,plainA,plainB)]==[16,4,0,4]
    # First-NaN mode ranks a NaN above a saturated numeric value.
    bad=torch.zeros(1,vocab);bad[0,0]=100.;bad[0,7]=torch.nan
    sc,idx=local_select(bad,1.,vocab,0,True)[0];assert torch.isinf(sc) and idx.item()==7
    # Exercise the real fused helper's CPU BF16 GEMM and private output buffers.
    # Only the Triton norm/RoPE math and final device write are substituted.
    layers=5; heads=2; dim=4; hidden=32; model=NS()
    model.n_layers=layers; model.layer_out_dim=2*heads*dim;model.num_kv_heads=heads;model.head_dim=dim;model.rotary_dim=dim
    model.device=torch.device('cpu');model.max_position_hint=131080; model._ensure_rope_cache=lambda x:None
    model.k_norm_weights=model.eps_values=None
    model.flat_kv_weight_t=torch.randn(hidden,layers*model.layer_out_dim,generator=gen).bfloat16()
    shape=(layers,rows,heads,dim)
    model._proj_workspace=torch.empty(rows,layers*model.layer_out_dim,dtype=torch.bfloat16)
    model._k_workspace=torch.empty(shape,dtype=torch.bfloat16); model._v_workspace=torch.empty_like(model._k_workspace)
    model._ensure_workspace=lambda n,d:None;model._mm_out_supported=True
    def norm(proj,kw,eps,cs,pos,h,d,r,k_out,v_out):
        pre=proj.view(rows,layers,2,heads,dim).permute(1,0,2,3,4)
        k_out.copy_(pre[:,:,0]);v_out.copy_(pre[:,:,1]);return k_out,v_out
    materialize=load(fused,"materialize",dict(torch=torch,_fused_norm_rope_stacked=norm))
    h=torch.randn(rows,hidden,generator=gen).bfloat16();pos=torch.arange(rows)
    saved=[];idx=torch.arange(rows).view(b,8); mask=torch.arange(8)[None,:]<n[:,None];live=idx[mask]
    baseline=torch.full((2,*shape),-71,dtype=torch.bfloat16)
    def publish(bank,l,k,v):
        bank[0,l,live]=k[live];bank[1,l,live]=v[live]
    materialize(model,h,pos,lambda l,k,v:publish(baseline,l,k,v))
    materialize(model,h,pos,lambda l,k,v:saved.append((l,k.clone(),v.clone())))
    reused=[]
    materialize(model,h,pos,lambda l,k,v:reused.append((l,k,v)))
    materialize(model,-h,pos,lambda *unused:None)  # another user of graph scratch
    staged=torch.full_like(baseline,-71);aliased=staged.clone()
    for l,k,v in saved:publish(staged,l,k,v)
    for l,k,v in reused:publish(aliased,l,k,v)
    assert torch.equal(staged.view(torch.int16),baseline.view(torch.int16))
    corrupt=(aliased.view(torch.int16)!=baseline.view(torch.int16)).sum().item();assert corrupt>0
    periods=json.loads((CP/'v140_decode_overlap_feasibility/check.json').read_bytes())['model']['rungs']
    fc=json.loads((CP/'v217_captured_verify_context/geometry.json').read_bytes())['cold_fc_weight_bytes_per_rank']['v217']
    kv_row=json.loads((CP/'v249_private_draft_swa/capacity.json').read_bytes())['bytes_per_draft_row']
    headbytes=25256*8192*2
    kv=8192*kv_row # five 8192 x local(K+V) projections, BF16
    assert (fc,kv,headbytes)==(80*2**20,20*2**20,413794304)
    boom={}
    for z in periods:
        m=z['verify_rows'];u=z['users'];t=z['estimated_period_ms'];rawpool=z['h200_pooler_raw_us']
        # Equal BF16 first-touch weight arithmetic intensity m FLOP/byte for these products.
        # Ignoring activations/collectives, the combined roofline floor equals the serial sum.
        floor=lambda weights:max(weights/4.8e12,m*weights/989.5e12)*1000
        assert abs(floor(headbytes+fc+kv)-floor(headbytes)-floor(fc+kv))<1e-9
        prep_ms=rawpool*(fc+kv)/(64*2**20)/1000;head_ms=rawpool*headbytes/(64*2**20)/1000
        boom[u]=dict(rows=m,ms=t,model_head_ms=head_ms,model_fc_plus_kv_ms=prep_ms,
            optimistic_overlap_gain=t/(t-min(head_ms,prep_ms)),cold_shared_roofline_ms=floor(headbytes+fc+kv),
            write_buffers_bytes=kv_row*m,bytes_per_rank_per_vocabulary_sweep=headbytes)
    rank0_needed=sum(int(all(result[r*8+j][1]<width for j in range(int(n[r])))) for r in range(b))
    print(json.dumps(dict(parent=json.loads((PARENT/'validation.json').read_bytes())['tree'],source_sha256=src_hash,
        prefix=dict(native_decisions=rows,ranks=ranks,patterns=b,commits_hist=Counter(n.tolist()),numeric_rows_short=Counter(l for _,l in result),
        requests_all_needed_rows_short=rank0_needed,identical_partial_pages_with_different_bonuses=True,first_nan_control=True),
        kv=dict(layers=layers,rows=rows,matched_words=staged.numel(),changed_if_scratch_reused=corrupt),bands=boom),indent=2))


if __name__=="__main__":
    check()
