"""Canonical slot order for equal-token siblings and repeated IDs under different parents."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from guard_cpu import HARNESS, array, choice, ptr


def orbit(tokens, parents, sequence):
    leader=0
    length=0
    selected=[0]
    for tok in sequence:
        kids=np.flatnonzero((parents==leader)&(tokens==tok))
        if not len(kids): break
        leader=int(kids[0])
        selected.append(leader)
        length+=1
    return min(len(tokens),length+1),selected


def reference(base,native,length):
    w=len(native)
    _,seen=orbit(base[0],base[2],native[1:])
    if length < len(seen):
        return base.copy()
    out=np.zeros_like(base)
    by_slot={}
    for n in range(length+1):
        out[:,n]=native[n],n,n-1,(1<<(n+1))-1
        if n<len(seen): by_slot[seen[n]]=n
    dest=length+1
    for n in range(1,w):
        if n in by_slot: continue
        if dest==w: break
        parent=by_slot[base[2,n]]
        out[:,dest]=base[0,n],base[1,n],parent,int(out[3,parent])|(1<<dest)
        by_slot[n]=dest
        dest+=1
    return out


def run(source):
    names=("_native_prefix_nodes","_select_spine_tree","_update_spine_margins","_update_tree_advantage")
    _,(_,select,update,old_update)=HARNESS["program"](source,names)
    select.__globals__["tl"].gather=lambda x,index,axis:array(np.take(x,index,axis=axis))
    w=8
    base=np.empty((4,w),dtype=np.int64)
    base[0]=[81,1,1,2,2,1,1,2] # equal-token pairs under two different parents
    base[2]=[-1,0,0,1,1,2,3,5]
    for n in range(w):
        parent=base[2,n]
        base[1,n]=0 if n==0 else base[1,parent]+1
        base[3,n]=(1<<n)|(0 if n==0 else int(base[3,parent]))
    counts={"layouts":0,"observations":0,"adjacent_margins":0}
    for chain in ([81,1,2,1,3,1,2,2], [81,1,1,2,3,4,1,2], [81,2,1,2,1,1,1,1]):
        native=np.array(chain,dtype=np.int64)
        family=[reference(base,native,s) for s in range(w)]
        for s in range(w):
            state=np.zeros(w,dtype=np.int32)
            if 0<s<w-1:
                state[s]=7; state[s+1]=-7
            elif s==w-1:
                state[0]=4
            assert choice(state)==s
            proposed=np.empty_like(base)
            select(*map(ptr,(base,native,state,proposed)),w,w,w)
            np.testing.assert_array_equal(proposed,family[s])
            counts["layouts"]+=1
            for upcoming in itertools.product((1,2),repeat=3):
                sequence=[*upcoming,3,4,1,1,8]
                relevant,_=orbit(proposed[0],proposed[2],sequence)
                committed=np.full(w,-91993,dtype=np.int64)
                committed[:relevant]=sequence[:relevant]
                known=np.array([orbit(t[0],t[2],sequence[:relevant])[0] for t in family],dtype=np.int32)
                actual=np.array([orbit(t[0],t[2],sequence)[0] for t in family],dtype=np.int32)
                assert np.abs(np.diff(actual)).max()<=1
                nxt=np.empty_like(state)
                ref=np.zeros(1,dtype=np.int32)
                inputs=(base[0],base[2],native,committed,np.array([relevant],dtype=np.int32))
                update(*map(ptr,(*inputs,state,nxt)),w,w)
                old_update(*map(ptr,(*inputs,state[:1],ref)),w,w)
                assert nxt[0]==ref[0]
                for j in range(1,w):
                    same=np.array_equal(family[j],family[j-1])
                    unknown=not same and relevant+1<w and known[j]==known[j-1]==relevant+1
                    if unknown: expected=state[j]
                    else:
                        assert actual[j]-actual[j-1]==known[j]-known[j-1]
                        expected=np.sign(state[j])*(3*abs(state[j])//4)+4*(known[j]-known[j-1])
                    assert nxt[j]==expected
                    counts["adjacent_margins"]+=1
                counts["observations"]+=1
    return counts


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    print(json.dumps(run(parser.parse_args().source),indent=2))
