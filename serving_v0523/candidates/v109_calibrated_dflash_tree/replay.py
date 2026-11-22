"""Synthetic Markov-target replay. Only current-step verified tokens reach the learners."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

from cpu_check import H, S, choice, make_program, mutation, paths, ptr, reward


def model_trees(scores,ids,anchor):
    w,k=ids.shape[0]+1,ids.shape[1]
    trees=[H['proposal_oracle'](ids,scores*b,anchor) for b in (1.,.5,2.)]
    chain=[anchor];rank=0
    for d in range(w-1):
        rank=int(np.argmax(scores[d,rank]))
        chain.append(ids[d,rank])
    chain=np.array(chain,np.int64)
    return [*S['make_family'](trees[0],chain)],trees,chain


def weights(scores,beta):
    s=scores.astype(np.float64)*beta
    p=np.exp(s-s.max(axis=-1,keepdims=True))
    return p/p.sum(axis=-1,keepdims=True)


def expected(tree,p):
    # Candidate IDs equal per-depth rank+10 in this constructed model.
    total=1.
    for path in paths(tree)[1:]:
        prev=0;prod=1.
        for d,token in enumerate(path):
            rank=token-10;prod*=p[d,prev,rank];prev=rank
        total+=prod
    return total


def choose_spine(state):
    util=np.r_[0,np.cumsum(state[1:])]
    if len(util)>2 and util[1:-1].max()>max(util[0],util[-1]):return int(1+np.argmax(util[1:-1]))
    return len(util)-1 if state[0]>0 else 0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path)
    args=ap.parse_args();pid,kernels=make_program(args.source)
    w,k=8,4;ids=np.tile(10+np.arange(k),(w-1,1));anchor=9
    opportunities=[];best=None
    for seed in range(128):
        rng=np.random.default_rng(seed)
        scores=rng.normal(0,1.15,(w-1,k,k)).astype(np.float32)
        family,trees,chain=model_trees(scores,ids,anchor)
        for beta in (.5,1.,2.):
            p=weights(scores,beta);base=max(expected(t,p) for t,_ in family)
            values=[expected(t,p) for t in trees[1:]]
            i=int(np.argmax(values))
            ratio=values[i]/base
            opportunities.append((beta,ratio))
            if best is None or ratio>best[0]:best=(ratio,seed,beta,i)
    _,seed,_,_=best
    rng=np.random.default_rng(seed);scores=rng.normal(0,1.15,(w-1,k,k)).astype(np.float32)
    family,trees,chain=model_trees(scores,ids,anchor)
    alts=np.ascontiguousarray(trees[1:])[:,:,None,:]
    # Memoize only complete synthetic evaluation statistics in this offline test.
    # Serving does no caching of requests, tokens, or completed continuations.
    all_options=[t for t,_ in family]+trees[1:]
    results={}
    for target_beta in (.5,1.,2.,'alternating'):
        cumulative=np.zeros(3);cho=np.zeros(3,np.int32);short=np.zeros(3)
        for replica in range(16):
            rng=np.random.default_rng(10000+replica)
            history=np.zeros(w+3,dtype=np.int32);inc_hist=np.zeros(w,dtype=np.int32)
            for step in range(48):
                beta=target_beta if target_beta!='alternating' else (.5 if (replica+step//8)%2 else 2.)
                p=weights(scores,beta);pr=0;seq=[]
                for d in range(w-1):
                    pr=int(rng.choice(k,p=p[d,pr]));seq.append(int(ids[d,pr]))
                seq.append(-1)
                n=choose_spine(inc_hist);base_depth=choose_spine(history[:w]);option=choice(history[w:])
                incumbent=family[base_depth][0];packed=np.empty_like(incumbent)
                kernels['_select_spine_tree'](*map(ptr,(trees[0],chain,history,packed)),w,w,w,w+3,ptr(alts),w)
                np.testing.assert_array_equal(packed,[incumbent,*trees[1:]][option])
                v101=reward(family[n][0],seq);v109=reward(packed,seq);stock=reward(trees[0],seq)
                cumulative += (stock,v101,v109);cho[option]+=1
                if step<8:short += (stock,v101,v109)
                for state,actual in ((inc_hist,v101),(history,v109)):
                    observed=np.array([actual],np.int32)
                    committed=np.full(w,-19871,dtype=np.int64);committed[:actual]=seq[:actual]
                    nxt=np.full_like(state,-9191)
                    extra=(ptr(alts),w) if state.size>w else ()
                    kernels['_update_spine_margins'](*map(ptr,(trees[0][0],trees[0][2],chain,committed,observed,state,nxt)),w,w,state.size,*extra)
                    state[:]=nxt
        ratio=cumulative[2]/cumulative[1]
        results[str(target_beta)]={"mean_committed":dict(zip(("fixed_original","v101b","v109"),(cumulative/(48*16)).tolist())),
                                  "first_8_calls":dict(zip(("fixed_original","v101b","v109"),(short/(8*16)).tolist())),
                                  "v109_over_v101b":float(ratio),"selected_counts":cho.tolist()}
    # The winning calibrated source must contain an off-spine prefix outside every V101b family member.
    winner=trees[best[3]+1]
    unavailable=sorted(set(paths(winner))-set.union(*(p for _,p in family)))
    assert unavailable and best[0]>1.03
    population={}
    for beta in (.5,1.,2.):
        vals=np.array([ratio for b,ratio in opportunities if b==beta])
        population[str(beta)]={'mean_best_calibrated_over_best_v101b_fixed_depth':float(np.mean(vals)),
                               'mean_best_of_full_family_over_best_v101b':float(np.mean(np.maximum(vals,1.))),
                               'max':float(max(vals)),'beneficial_seeds':int(np.sum(vals>1.000001))}
    print(json.dumps(dict(synthetic=True,source='random Gaussian native selector lattices',seeds_examined=128,
                          approx_workload_prevalence=None,width=w,top_k=k,oracle_population=population,
                          witness_seed=seed,witness_beta=best[2],witness_best_fixed_ratio=best[0],
                          off_spine_prefixes_unavailable_to_v101b=unavailable,paired_replay=results),indent=2))


if __name__=='__main__':main()
