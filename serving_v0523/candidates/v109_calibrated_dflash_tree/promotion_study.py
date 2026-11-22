"""Compare promotion thresholds with held-out synthetic selector lattices."""
import json
from pathlib import Path
import numpy as np
from replay import model_trees, weights
from cpu_check import paths


def traces(scores,beta,seeds,w,k):
    # Paired target continuations for every policy within this constructed workload.
    worlds=seeds.shape[:2]
    rank=np.zeros(worlds,np.int32);prefix=np.zeros(worlds,np.int64)
    history=[prefix.copy()]
    p=weights(scores,beta) if beta!='alternating' else None
    for d in range(w-1):
        if p is None:
            p0,p1=weights(scores,.5),weights(scores,2.)
            phases=(np.arange(worlds[0])[:,None]+np.arange(worlds[1])[None,:]//8)%2
            dist=np.where(phases[:,:,None]==0,p0[d,rank],p1[d,rank])
        else:dist=p[d,rank]
        rank=(seeds[:,:,d,None]>=np.cumsum(dist,axis=-1)).sum(axis=-1)
        prefix=prefix*k+rank
        history.append(prefix.copy())
    return history


def all_rewards(trees,h,k):
    shape=h[0].shape
    gold=np.ones((*shape,len(trees)),np.int32)
    for j,tree in enumerate(trees):
        for path in paths(tree)[1:]:
            encoded=0
            for item in path:encoded=encoded*k+(item-10)
            gold[:,:,j]+=h[len(path)]==encoded
    return gold


def live(gold,common,threshold):
    b,t,n=gold.shape;w=n-2
    st=np.zeros((b,w),np.int32);votes=np.zeros((b,3),np.int32)
    allr=np.arange(b)
    totals=np.zeros(t,np.int64)
    for i in range(t):
        util=np.cumsum(st,axis=1)-st[:,:1]
        best=1+np.argmax(util[:,1:-1],axis=1)
        base=np.where(util[allr,best]>np.maximum(0,util[:,-1]), best,np.where(st[:,0]>0,w-1,0))
        if threshold==100:
            variant=base
        else:
            bwin, swin = votes[:,0]>=threshold,votes[:,1]>=threshold
            prefer=(votes[:,2]>0)|((votes[:,2]==0)&(votes[:,0]>=votes[:,1]))
            variant=np.where(bwin&(~swin|prefer),w,np.where(swin,w+1,base))
        observed=gold[allr,i,variant]; totals[i]=observed.sum()
        lower=np.minimum(gold[:,i,:],observed[:,None]+1)
        incumbent=lower[allr,base]
        old=st.copy()
        st[:,0]=np.clip(old[:,0]-np.sign(old[:,0])+2*np.sign(lower[:,w-1]-lower[:,0]),-w,w)
        for j in range(1,w):
            a,z=lower[:,j],lower[:,j-1];pr=old[:,j]
            unknown=(j>common)&(observed+1<w)&(a==z)&(a==observed+1)
            st[:,j]=np.where(unknown,pr,np.sign(pr)*(3*np.abs(pr)//4)+4*(a-z))
        for col,(a,z) in enumerate(((lower[:,w],incumbent),(lower[:,w+1],incumbent),(lower[:,w],lower[:,w+1]))):
            pr=votes[:,col];unknown=(a==z)&(a==observed+1)&(a<w)
            votes[:,col]=np.where(unknown,pr,np.clip(pr-np.sign(pr)+2*np.sign(a-z),-w,w))
    return totals


def main():
    w,k=8,4;ids=np.tile(10+np.arange(k),(w-1,1)); profiles=(.5,1.,2.,'alternating')
    thresholds=(100,1,3,4,5)
    totals={p:{t:np.zeros(48,np.int64) for t in thresholds} for p in profiles}
    for seed in range(1010,1042):
        rng=np.random.default_rng(seed);scores=rng.normal(0,1.15,(w-1,k,k)).astype('f4')
        family,trees,chain=model_trees(scores,ids,9)
        common=max(i for i in range(w) if tuple(chain[1:i+1]) in family[0][1])
        draw=rng.random((64,48,w-1))
        for p in profiles:
            gold=all_rewards([t for t,_ in family]+trees[1:],traces(scores,p,draw,w,k),k)
            for t in thresholds:totals[p][t]+=live(gold,common,t)
    results={str(p):{str(t):{'relative_to_v101b_48_calls':float(v.sum()/totals[p][100].sum()),
                             'relative_to_v101b_first_8':float(v[:8].sum()/totals[p][100][:8].sum()),
                             'mean_48':float(v.sum()/(32*64*48))}
                     for t,v in per.items()} for p,per in totals.items()}
    print(json.dumps(dict(synthetic=True,unseen_lattice_seeds=[1010,1041],paired_requests_per_lattice=64,
                          calls=48,profiles=results),indent=2))


if __name__=='__main__':main()
