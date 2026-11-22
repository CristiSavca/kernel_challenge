"""Execute actual tempered proposals, selection, commit and both feedback kernels on CPU."""
import argparse
import itertools
import json
from pathlib import Path
import runpy

import numpy as np


D=Path(__file__).resolve().parent.parent
H=runpy.run_path(str(D/'v70_dflash_selector_tree/cpu_check.py'))
S=runpy.run_path(str(D/'v101_adaptive_draft_spine/spine_cpu.py'))
ptr,array=H['Pointer'],H['array']
fnames=('_edge_probabilities','_propose_tree','_native_prefix_nodes','_select_spine_tree','_update_spine_margins',
        '_spine_choice','_known_tree_reward','_temperature_vote','_accept_tree')


def paths(tree):
    rows=[()]
    for t in range(1,tree.shape[1]):
        p=tree[2,t]
        assert 0<=p<t and tree[1,t]==tree[1,p]+1
        assert tree[3,t]==tree[3,p]|(1<<t)
        rows.append(rows[p]+(int(tree[0,t]),))
    return rows


def reward(tree,seq):
    # A full path identifies the predecessor. Tree rank order resolves siblings with equal tokens.
    nodes=0
    for i,t in enumerate(seq):
        children=np.flatnonzero((tree[2]==nodes)&(tree[0]==t))
        if not children.size: return min(i+1,tree.shape[1])
        nodes=children[0]
    return min(len(seq)+1,tree.shape[1])


def choice(prior):
    broad,sharp,rel=prior
    if broad>=3 and (sharp<3 or rel>0 or (rel==0 and broad>=sharp)):return 1
    if sharp>=3:return 2
    return 0


def mutation(prior,a,b,obs,w):
    if a==b==obs+1<w:return int(prior)
    return int(np.clip(prior-np.sign(prior)+2*np.sign(a-b),-w,w))


def make_program(src):
    pid,functions=H['program'](src,fnames)
    tl=functions[0].__globals__['tl'];tl.gather=lambda x,ix,axis:array(np.take(x,ix,axis=axis));tl.static_range=range
    return pid,dict(zip(fnames,functions))


def check(src):
    rng=np.random.default_rng(10087);pid,ks=make_program(src)
    c={'proposals':0,'incumbent_width_plus_three_strides':0,'selected_and_accepted':0,
       'ternary_pairwise_observations':0,'padding_or_sentinel_checks':0}
    for w,k,cap in itertools.product(range(2,9),(2,5,16),(3,5)):
        live=3;block=1<<(w-1).bit_length();edgeblock=1<<(w*k-1).bit_length()
        for case in range(3):
            scores=rng.normal(0,1.4,(live,w-1,k,k)).astype(np.float32)
            if case==1:scores.fill(0.)
            if case==2:scores[...,0]=np.nan;scores[0,0,0,1]=np.inf
            ids=np.stack([np.stack([rng.permutation(k)+(1<<33) for _ in range(w-1)]) for _ in range(live)]).astype(np.int64)
            anchors=np.arange(live,dtype=np.int64)*7+49
            orig=np.full((4,cap,w),-424242,dtype=np.int64);alts=np.full((2,4,cap,w),-424242,dtype=np.int64)
            prior=rng.integers(-8,9,(live,w+3),dtype=np.int32)
            # Independent poison patterns distinguish the extra policies.
            prior[:,w:]=np.array([[0,-3,8],[4,3,2],[4,4,-1]],dtype=np.int32)
            chain=np.zeros((live,w),dtype=np.int64)
            for r in range(live):
                pid[0]=r;chain[r,0]=anchors[r];rank=0
                for depth in range(w-1):
                    rank=int(np.argmax(np.nan_to_num(scores[r,depth,rank],nan=-np.inf)))
                    chain[r,depth+1]=ids[r,depth,rank]
                for variant,beta in enumerate((1.,.5,2.)):
                    pid[1]=variant
                    ks['_propose_tree'](ptr(ids),ptr(scores),ptr(anchors),*map(ptr,orig),
                         1,w,k,edgeblock,ptr(alts),cap*w)
                    got=orig[:,r] if variant==0 else alts[variant-1,:,r]
                    expected=H['proposal_oracle'](ids[r],scores[r]*beta,anchors[r])
                    np.testing.assert_array_equal(got,expected)
                    c['proposals']+=1
            assert np.all(orig[:,live:]==-424242) and np.all(alts[:,:,live:]==-424242)
            c['padding_or_sentinel_checks']+=1
            orig=np.ascontiguousarray(orig[:,:live])
            base=np.empty_like(orig);chosen=np.empty_like(orig);nxt=np.full_like(prior,-9999)
            for r in range(live):
                pid[0]=r;ks['_select_spine_tree'](*map(ptr,(orig,chain,prior,base)),live*w,w,block,w+3)
                family=S['make_family'](orig[:,r],chain[r])
                utility=np.r_[0,np.cumsum(prior[r,1:w])];s=(w-1 if prior[r,0]>0 else 0)
                if w>2 and max(utility[1:-1])>max(utility[0],utility[-1]):s=1+np.argmax(utility[1:-1])
                np.testing.assert_array_equal(base[:,r],family[s][0])
                c['incumbent_width_plus_three_strides']+=1
            seqs=[];preds=np.empty((live,w),dtype=np.int64)
            for r in range(live):
                pid[0]=r;ks['_select_spine_tree'](*map(ptr,(orig,chain,prior,chosen)),live*w,w,block,w+3,ptr(alts),cap*w)
                selected=choice(prior[r,w:]);options=[base[:,r],alts[0,:,r],alts[1,:,r]]
                np.testing.assert_array_equal(chosen[:,r],options[selected])
                possible=sorted(set.union(*(set(paths(t)) for t in options)))
                p=possible[rng.integers(len(possible))]
                seq=list(p)+[-111]*(w-len(p));seqs.append(seq)
                for i,path in enumerate(paths(chosen[:,r])):
                    preds[r,i]=seq[len(path)] if path==tuple(seq[:len(path)]) else -7732
            lengths=np.zeros(live,dtype=np.int32);bonus=np.empty(live,dtype=np.int64);newlen=bonus.copy()
            prefix=np.arange(live,dtype=np.int64)+50;out=np.full_like(chain,-84743);order=out.copy()
            for r in range(live):
                pid[0]=r
                ks['_accept_tree'](*map(ptr,(chosen[0],chosen[2],preds,prefix,lengths,bonus,out,newlen,order)),w,block)
                obs=int(lengths[r]);assert obs==reward(chosen[:,r],seqs[r])
                np.testing.assert_array_equal(out[r,:obs],seqs[r][:obs]);out[r,obs:]=-11121314
                assert newlen[r]==prefix[r]+obs
                c['selected_and_accepted']+=1
                ks['_update_spine_margins'](*map(ptr,(orig[0],orig[2],chain,out,lengths,prior,nxt)),w,block,w+3)
                assert np.all(nxt[r,w:]==-9999)
                baseline_next=nxt[r,:w].copy()
                ks['_update_spine_margins'](*map(ptr,(orig[0],orig[2],chain,out,lengths,prior,nxt)),w,block,w+3,ptr(alts),cap*w)
                np.testing.assert_array_equal(nxt[r,:w],baseline_next)
                alternatives=[base[:,r],alts[0,:,r],alts[1,:,r]]
                lower=[reward(t,seqs[r][:obs]) for t in alternatives]
                gold=[reward(t,seqs[r]) for t in alternatives]
                for j,(a,b) in enumerate(((1,0),(2,0),(1,2))):
                    if lower[a]!=lower[b]:assert np.sign(lower[a]-lower[b])==np.sign(gold[a]-gold[b])
                    assert nxt[r,w+j]==mutation(prior[r,w+j],lower[a],lower[b],obs,w)
                    c['ternary_pairwise_observations']+=1
    return c


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path)
    print(json.dumps(check(ap.parse_args().source),indent=2))
