"""Native DFLASH output dependence, exact row-packing parity, and cold-bank sensitivity."""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

SERVING = Path(__file__).resolve().parents[2]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
W = 8


def source(path):
    return subprocess.check_output(["git","-C",SERVING/"sglang","show",f"{PIN}:python/sglang/srt/{path}"],text=True)


def method(src,name):
    n=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name==name)
    n.decorator_list=[]
    ns=dict(torch=torch,F=F)
    exec(compile(ast.Module([n],type_ignores=[]),name,"exec"),ns)
    return ns[name]


def native_checks():
    from random import Random
    rng=Random(145)
    worker=source("speculative/dflash_worker_v2.py")
    draft=source("models/dflash.py")
    commit=method(worker,"_commit_accept");conv=method(draft,"_grouped_conv")
    assert "hidden_states = self.mlp(hidden_states)" in draft
    assert "hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)" in draft
    assert "ctx_hidden = self.draft_model.project_target_hidden(target_hidden)" in worker
    assert "cache_loc_2d=cache_loc.view(verify_out_cache_loc_2d.shape)" in worker
    assert "bonus_tokens=bonus," in worker
    # Native two-tap post-MLP convolution. Set the learned delta to zero; both
    # base taps are one. All operands below have exactly representable BF16 values.
    h,group=64,4
    out=torch.zeros((W,h),dtype=torch.bfloat16)
    out[0,0]=4
    out[1,1]=2
    delta=torch.zeros((W,2,h//group),dtype=out.dtype)
    base=torch.ones((2,h),dtype=out.dtype)
    words=torch.zeros((17,h),dtype=out.dtype)
    words[0,0]=1;words[1,1]=1
    words[2:,1]=.25
    def result(x):
        vectors=conv(x,delta,base,W,h//group,group,2)
        row=F.rms_norm(vectors[1:2].float(),(h,),eps=1e-6).to(out.dtype)
        logits=F.linear(row,words)
        return int(logits.argmax()),set(logits.topk(16)[1].view(-1).tolist())
    full,keep=result(out)
    zero=out.clone();zero[0].zero_()
    trimmed,cut=result(zero)
    assert (full,trimmed)==(0,1) and 0 in keep and 0 not in cut
    # Exhaust every possible four-user confirmed-prefix vector with native
    # buffer packing. Norm/FC/KV outputs are compared only at committed rows.
    patterns=list(itertools.product(range(1,W+1),repeat=4))
    seq=slots=0;hist=Counter()
    g=torch.Generator().manual_seed(145)
    features=torch.randint(-2,3,(4*W,5*32),generator=g,dtype=torch.int16).bfloat16()
    fc=torch.randint(-2,3,(32,160),generator=g,dtype=torch.int16).bfloat16()
    kv=torch.randint(-2,3,(5*2*4,32),generator=g,dtype=torch.int16).bfloat16()
    native=lambda x:F.linear(F.rms_norm(F.linear(x,fc).float(),(32,),eps=1e-6).bfloat16(),kv)
    projected=native(features)
    candidates=torch.arange(4*W).view(4,W)+1000
    bonus=torch.tensor([992,993,994,995])
    for v in patterns:
        lengths=torch.tensor(v,dtype=torch.int32)
        ids,sz=commit(candidates,lengths-1,bonus)
        assert sz.tolist()==list(v)
        mask=(torch.arange(W)[None,:]<lengths[:,None]).flatten()
        assert torch.equal(native(features[mask]),projected[mask])
        for b,n in enumerate(v):
            assert ids[b,:n].tolist()==[*candidates[b,1:n].tolist(),int(bonus[b])]
        hist[math.ceil(sum(v)/16)]+=1;slots+=sum(v);seq+=4
    # Every new write is anchored at the *previous* prefix plus its accepted
    # length. Early returning a bonus creates no duplicate context row.
    heads=[rng.randrange(2000) for _ in range(13)]
    writes=[set() for _ in heads]
    for _ in range(400):
        selected=rng.sample(range(len(heads)),rng.randrange(1,len(heads)+1))
        lengths=torch.tensor([rng.randint(1,W) for _ in selected])
        idx=torch.arange(len(selected)*W).view(-1,W)+8000
        _,actual=commit(idx,lengths-1,torch.zeros_like(lengths))
        for b,i in enumerate(selected):
            coords=set(range(heads[i],heads[i]+int(actual[b])))
            assert not coords&writes[i]
            writes[i]|=coords;heads[i]+=int(actual[b])
    return dict(complete_four_user_patterns=len(patterns),requests=seq,physical_live_rows=slots,
                tile_counts_under_uniform_pattern_enumeration=dict(hist),
                independently_shuffled_steps=400,unique_positions=sum(map(len,writes)),
                anchor_winner_before=full,anchor_winner_after=trimmed,
                first_full_top16_contains_token0=True,trimmed_top16_contains_token0=False)


def distribution(rate):
    lo,hi=0.,1.
    for _ in range(80):
        q=(lo+hi)/2
        if sum(q**k for k in range(W))<rate:lo=q
        else:hi=q
    q=(lo+hi)/2
    return q,[q**(n-1)*(1-q) if n<W else q**(W-1) for n in range(1,W+1)]


def tile_extrema(batch,mean):
    # All distributions of the simultaneous sum S with bounded support and
    # specified mean. A linear objective attains its extremes on two support points.
    lower=[];upper=[]
    for a in range(batch,math.floor(mean)+1):
        for b in range(math.ceil(mean),batch*W+1):
            w=1.0 if a==b else (b-mean)/(b-a)
            n=w*math.ceil(a/16)+(1-w)*math.ceil(b/16)
            lower.append(n);upper.append(n)
    return min(lower),max(upper)


def model():
    official=json.loads((SERVING/"runs/abd217baf0ed49fd.json").read_text())["result"]
    fc_k,fc_n=40960,8192
    kv_k,kv_n=8192,5*2*128
    v64=(fc_k*fc_n+kv_k*kv_n)*2
    shard=(fc_k*(fc_n//8)+kv_k*kv_n)*2
    r=[]
    for row in official["rungs"]:
        b=row["concurrency"];n=b*W;a=row["acceptance"];s=b*a
        q,p=distribution(a)
        mass={0:1.}
        for _ in range(b):
            new=Counter()
            for total,v in mass.items():
                for i,prob in enumerate(p,1):new[total+i]+=v*prob
            mass=dict(new)
        assert abs(sum(mass.values())-1)<1e-12
        def floors(bank):
            return [max(bank/4.8e12,bank*rows/989e12)*1e6 for rows in (n,s)]
        r.append(dict(users=b,width=W,dense_rows=n,mean_live=s,all_weight_bytes_v64=v64,
                      all_weight_bytes_v120b=shard,skipped_v64_gflop=(n-s)*v64/1e9,
                      skipped_v120b_gflop=(n-s)*shard/1e9,dense_m16_tiles=math.ceil(n/16),
                      min_max_expected_m16_tiles_at_reported_mean=tile_extrema(b,s),
                      expected_m16_tiles_if_independent=sum(math.ceil(k/16)*v for k,v in mass.items()),
                      probability_at_least_one_m16_tile_dropped_if_independent=sum(
                          v for k,v in mass.items() if math.ceil(k/16)<math.ceil(n/16)),
                      assumed_geometric_match=q,
                      v64_cold_fc_and_kv_floor_us_dense_compact=floors(v64),
                      v120b_cold_fc_and_kv_floor_us_dense_compact=floors(shard),
                      period_proxy_ms=1000*a/row["tok_s_per_user_p50"]))
    return r


def measured_attention():
    records=json.loads((SERVING.parent/"experiments/runs/8f9e5971f95841a0.json").read_text())["response"]["result"]["records"]
    out={}
    for users in (1,16,32):
        rows=[r for r in records if r["event"]=="benchmark" and r["case"].get("q_len",1)==8
              and r["case"]["batch"]==users]
        assert {r["case"]["layer"] for r in rows}=={"full","sliding"}
        def weighted(key):
            return sum(r[key]*(16 if r["case"]["layer"]=="full" else 46) for r in rows)/1e6
        out[users]={"stock_ms":weighted("baseline_ns"),"candidate_ms":weighted("median_ns"),
                    "difference_ms":weighted("baseline_ns")-weighted("median_ns")}
    return out


if __name__=="__main__":
    torch.set_num_threads(1)
    p=SERVING/"candidates/v64_humming_grouped_sequence_parallel/patch.diff"
    assert hashlib.sha256(p.read_bytes()).hexdigest()=="f6aaf93ce06ccfce9461e06bc3f71dea05dd901c01497c8be31bd2c515da689a"
    result=dict(pin=PIN,torch=torch.__version__,native=native_checks(),rungs=model(),historical_isolated_attention_h200=measured_attention(),physical_gpu=False)
    print(json.dumps(result,indent=2))
