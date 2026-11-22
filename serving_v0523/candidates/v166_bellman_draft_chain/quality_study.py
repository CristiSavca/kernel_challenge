"""Synthetic expected commits, exhaustive short sequences and actual native target acceptance."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v127_native_prefix_verify'))
from prefix_study import load_function


def probs(logits,beta):
    scaled=beta*logits
    e=np.exp(scaled-np.max(scaled,axis=-1,keepdims=True))
    return e/e.sum(-1,keepdims=True)


def choices(p):
    b,d,k,_=p.shape;r=np.arange(b);v=np.zeros((b,k));decisions=[]
    for j in reversed(range(d)):
        u=p[:,j]*(1+v[:,None,:]);v=u.max(2)
        tie=np.where(u==v[:,:,None],p[:,j],-np.inf)
        decisions.insert(0,tie.argmax(2))
    outputs=[]
    for chain in (p.argmax(3),np.stack(decisions,axis=1)):
        prev=np.zeros(b,dtype=np.int32);path=[]
        for j in range(d):
            prev=chain[r,j,prev];path.append(prev)
        outputs.append(np.stack(path,1))
    return outputs


def commits(p,chain):
    b,d=chain.shape;r=np.arange(b);prev=np.zeros(b,dtype=int)
    reach=np.ones(b);m=np.ones(b)
    for j in range(d):
        nxt=chain[:,j];reach*=p[r,j,prev,nxt];m+=reach;prev=nxt
    return m


def exhaustive(rng):
    count=0
    for d,k in ((2,4),(3,4),(4,3),(5,3)):
        n=32;s=rng.normal(0,1.4,(n,d,k,k));p=probs(s,1);g,opt=choices(p)
        maximum=np.full(n,-np.inf)
        for ch in itertools.product(range(k),repeat=d):
            score=commits(p,np.tile(ch,(n,1)));maximum=np.maximum(maximum,score);count+=n
        assert np.allclose(maximum,commits(p,opt),rtol=1e-13,atol=1e-13)
    return count


def native_commit(rng,chains,ids):
    native=load_function('kernels/ops/speculative/dflash.py','compute_dflash_correct_drafts_and_bonus')
    commit=load_function('srt/speculative/dflash_worker_v2.py','_commit_accept')
    b,d=chains[0].shape;cases=0
    for path in chains:
        proposal=ids[np.arange(b)[:,None],np.arange(d)[None,:],path]
        root=rng.integers(2**34,2**37,(b,1),dtype='i8')
        c=torch.from_numpy(np.column_stack((root,proposal)))
        for length in range(d+1):
            target=rng.integers(2**39,2**40,(b,d+1),dtype='i8')
            target[:,:length]=proposal[:,:length]
            # Additional predictions past the first error may agree accidentally
            # and still cannot be committed past that error.
            if length+1<d:target[:,length+1:d]=proposal[:,length+1:d]
            t=torch.from_numpy(target);a,bonus=native(candidates=c,target_predict=t)
            out,lens=commit(c,a,bonus)
            assert torch.equal(lens,torch.full_like(lens,length+1))
            assert torch.equal(out[:,:length+1],t[:,:length+1]) and torch.equal(bonus.long(),t[:,length].long())
            cases+=b
    return cases


def main():
    rng=np.random.default_rng(166047);x=exhaustive(rng);b,d,k=768,7,16
    logits=rng.normal(size=(b,d,k,k))*rng.uniform(.6,3.2,(b,d,k,1))
    p=probs(logits,1);g,dp=choices(p)
    rows=[]
    def row(label,target):
        old=commits(target,g);new=commits(target,dp)
        return dict(target=label,native=float(old.mean()),candidate=float(new.mean()),ratio=float(new.mean()/old.mean()),
                    requests_better=int(np.sum(new>old+1e-12)),requests_worse=int(np.sum(new<old-1e-12)))
    for beta in (.5,1,2,4,10):
        rows.append(row('temperature inverse '+str(beta),probs(logits,beta)))
    lo,hi=1.,12.
    for _ in range(32):
        mid=(lo+hi)/2
        if commits(probs(logits,mid),g).mean()<4:lo=mid
        else:hi=mid
    in_range=lo>1 and hi<12
    assert in_range
    rows.append(row('target sharpened to synthetic native mean of four',probs(logits,mid)))
    native_ids=rng.integers(2**35,2**39,(b,d,k),dtype='i8');n=native_commit(rng,(g,dp),native_ids)
    # Standard native depth and width with two finite successors, reused afresh
    # from each synthetic position. Row zero represents the verified anchor.
    demo=np.zeros((1,d,k,k));demo[:,:,:,0]=.51;demo[:,:,:,1]=.49
    demo[:,1:,1,0]=.01;demo[:,1:,1,1]=.99
    ga,da=choices(demo);d0=commits(demo,ga)[0];d1=commits(demo,da)[0]
    return dict(seed=166047,synthetic=True,lattices=b,depth=d,top_k=k,exhaustive_paths_checked=x,
                native_commit_anchor_cases=n,changed_requests=int(np.any(g!=dp,axis=1).sum()),
                sensitivity=rows,sharpened_case_beta=mid,
                demo=dict(native=ga.tolist()[0],candidate=da.tolist()[0],model_expected=[d0,d1],
                          deterministic_all_zero_target_commits=[8,1]),
                assumptions='Markov target probabilities at each speculative anchor; no real customer prompts, target traces, or serving latency.')

if __name__=='__main__':print(json.dumps(main(),indent=2))
