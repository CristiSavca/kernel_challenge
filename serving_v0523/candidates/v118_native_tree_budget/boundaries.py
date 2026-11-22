"""Certificates for staged draft-head depth, synthetic scheduling costs and KV ownership."""
import json
import math
from pathlib import Path
import numpy as np
from budget import native_tree


def depth_counterexample():
    scores=np.full((7,16,16),-np.inf)
    scores[0,:,0:2]=np.log([.60,.40])
    scores[1:,:,0]=0
    alt=scores.copy()
    alt[-1,:,0:2]=np.log([.50,.50])
    a=native_tree(scores,8);b=native_tree(alt,8)
    def paths(tree):
        result=[()]
        for n in tree[1:]:result.append(result[n.parent]+(n.rank,))
        return result
    x,y=paths(a),paths(b)
    assert np.array_equal(scores[:-1],alt[:-1])
    assert x[:-1]==y[:-1] and x[-1]==(0,)*7 and y[-1]==(1,)
    return {"common_first_six_score_planes":True,"chosen_last_prefix":list(x[-1]),
            "chosen_last_prefix_after_changing_only_plane_seven":list(y[-1])}


def depth_certificates():
    # Work bound only. An unresolved node at depth d needs a child distribution
    # if selected before the last slot and the model admits another depth.
    rng=np.random.default_rng(367887);out={}
    for b in (1,4,16,32):
        part={}
        for w in (8,16,24):
            data=[]
            for _ in range(120):
                trees=[native_tree(rng.normal(0,scale,(7,16,16)),w)
                       for scale in rng.choice([1.25,2.5,4.5,8.],b)]
                needed=[min(7,1+max(n.depth for n in t[:-1])) for t in trees]
                # All ranks need another projection launch if any request does.
                data.append(needed)
            s=np.asarray(data)
            part[str(w)]={str(g):{
                "fraction_individual_requests_proven_complete":round(float((s<=g).mean()),5),
                "fraction_batches_needing_second_projection":round(float((s.max(1)>g).mean()),5),
                "mean_projected_rows_fraction_if_remaining_requests_use_all_seven":
                    round(float((g+(7-g)*(s>g).mean())/7),5)
            } for g in (3,4,5,6)}
        out[str(b)]=part
    return out


def allocation_proof():
    rng=np.random.default_rng(381109)
    cases={"page_one_replay_commits":0,"post_remap_disjoint_requests":0,"anti_alias_samples":0}
    for b in (1,4,16,32):
        for phase in range(12):
            from budget import allocate
            trees=[native_tree(rng.normal(0,f,(7,16,16)),16) for f in rng.uniform(.8,9.,b)]
            widths=allocate(trees,8*b,4)
            # Eight reserved slots per request. Candidate slots above eight can
            # borrow spare *reservation* IDs from another request for this round.
            original=rng.permutation(b*8).reshape(b,8)
            flat=original.flatten().copy()
            start=np.r_[0,np.cumsum(widths)]
            memory={}
            committed=[]
            leftovers=[]
            for r,(t,ww) in enumerate(zip(trees,widths)):
                owned=flat[start[r]:start[r+1]]
                for n,slot in enumerate(owned):
                    memory[int(slot)]=(r,n)
                    mask=t[n].mask
                    live=[i for i in range(ww) if mask >> i & 1]
                    assert all(memory[int(owned[j])]==(r,j) for j in live)
                # An arbitrary verified path. Its target bonus occupies no
                # additional input KV slot and its length never exceeds eight.
                idx=rng.integers(ww);path=[]
                while idx>=0:
                    path.append(int(idx));idx=t[idx].parent
                path.reverse()
                assert len(path)<=8 and len(path)==t[path[-1]].depth+1
                target=[int(owned[j]) for j in path]
                committed.append(target)
                leftovers.extend(int(owned[j]) for j in range(ww) if j not in path)
                assert all(memory[i][0]==r for i in target)
                if list(owned[:len(path)])!=target:cases["anti_alias_samples"]+=1
            # A global permutation or actual copy is mandatory when the committed
            # nodes do not belong to the request's original physical reservation.
            donor=iter(leftovers)
            next_reservation=[slots+[next(donor) for _ in range(8-len(slots))] for slots in committed]
            assert sorted(v for rr in next_reservation for v in rr)==list(range(b*8))
            for r,rr in enumerate(next_reservation):
                assert all(memory[v][0]==r for v in rr[:len(committed[r])])
                cases["post_remap_disjoint_requests"]+=1
            cases["page_one_replay_commits"]+=1
    return cases


def expert_traffic():
    out={}
    for users in (1,4,16,32):
        ys={}
        for w in (8,16,24):
            m=users*w
            probs=np.array([math.comb(m,c)*((1/32)**c)*((31/32)**(m-c)) for c in range(min(m,181)+1)])
            # For the tested m <= 768, excluded count tail above 181 is negligible.
            assert abs(float(probs.sum())-1)<1e-10
            unique=32*(1-probs[0]);passes=32*np.dot(np.ceil(np.arange(len(probs))/8),probs)
            ys[str(w)]={"expected_distinct_local_experts":round(float(unique),5),
                        "expected_eight_row_weight_passes":round(float(passes),5)}
        out[str(users)]=ys
    return out


if __name__=="__main__":
    print(json.dumps({"synthetic":True,"head_counterexample":depth_counterexample(),
                      "fixed_stage_head_model":depth_certificates(),"temporary_slot_model":allocation_proof(),
                      "uniform_top8_expert_traffic":expert_traffic()},indent=2))
