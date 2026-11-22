"""Native-depth DFLASH tree and a fixed-total-row allocator, with an exact expectation oracle."""
from dataclasses import dataclass
import heapq
import itertools
import json
from pathlib import Path
import runpy

import numpy as np


@dataclass
class Node:
    parent: int
    rank: int
    depth: int
    mass: float
    mask: int


def probabilities(scores, scale=1.0):
    scores = np.where(np.isnan(scores), -np.inf, scores) * scale
    peak = scores.max(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        weights = np.where(np.isfinite(peak), np.exp(scores - peak), (scores == peak).astype(float))
    return weights / weights.sum(axis=-1, keepdims=True)


def native_tree(scores, width):
    depths, fanout, _ = scores.shape
    probabilities_by_edge = probabilities(scores)
    root = Node(-1, 0, 0, 1.0, 1)
    nodes = [root]
    queue = []
    def extend(parent):
        p = nodes[parent]
        for rank, prob in enumerate(probabilities_by_edge[p.depth, p.rank]):
            heapq.heappush(queue, (-p.mass*prob, parent*fanout + rank, p.depth + 1))
    extend(0)
    while queue and len(nodes) < width:
        neg_mass, edge, depth = heapq.heappop(queue)
        parent, rank = divmod(edge, fanout)
        nodes.append(Node(parent, rank, depth, -neg_mass, nodes[parent].mask | (1 << len(nodes))))
        if depth < depths:
            extend(len(nodes)-1)
    return nodes


def allocate(trees, budget, floor=1):
    """All requests include an anchor. Remaining rows maximize total predicted commits."""
    caps = np.array([len(tree) for tree in trees], dtype=np.int64)
    widths = np.minimum(caps, floor)
    if not int(widths.sum()) <= budget <= int(caps.sum()):
        raise ValueError("budget outside attainable range")
    queue = [(-tree[w].mass, r) for r,(tree,w) in enumerate(zip(trees,widths)) if w < len(tree)]
    heapq.heapify(queue)
    for _ in range(budget-int(widths.sum())):
        _, request = heapq.heappop(queue)
        widths[request] += 1
        width = widths[request]
        if width < caps[request]:
            heapq.heappush(queue, (-trees[request][width].mass,request))
    return widths


def counterfactual(tree, target):
    out = [1.0]
    for node in tree[1:]:
        parent = tree[node.parent]
        out.append(out[node.parent] * target[parent.depth,parent.rank,node.rank])
    return np.cumsum(out)


def tests():
    rng = np.random.default_rng(318013)
    oracle = runpy.run_path(str(Path(__file__).resolve().parents[1]/"v70_dflash_selector_tree/cpu_check.py"))["proposal_oracle"]
    checked = {"native_v70_trees":0,"allocations_vs_exhaustive":0,"exhaustive_target_paths":0}
    for seeds in range(60):
        score = rng.normal(0.,2.,(7,16,16))
        if seeds%9==0:score[0,0] = -np.inf
        if seeds%9==1:score[0,0,:3] = np.inf
        ids = np.stack([rng.permutation(16)+100*i for i in range(7)])
        tree = native_tree(score,24)
        ks=oracle(ids,score,1234); original=np.array([1234,*[ids[n.depth-1,n.rank] for n in tree[1:8]]])
        for arr,z in ((ks[0],original),(ks[1],[n.depth for n in tree[:8]]),
                      (ks[2],[n.parent for n in tree[:8]]),(ks[3],[n.mask for n in tree[:8]])):
            np.testing.assert_array_equal(arr,z)
        assert all(tree[i].mass <= tree[i-1].mass+1e-12 for i in range(1,len(tree)))
        assert max(n.depth for n in tree)<=7
        checked["native_v70_trees"]+=1
    for _ in range(80):
        scores = [rng.normal(0.,1.5,(3,2,2)) for _ in range(3)]
        trees = [native_tree(s,8) for s in scores]
        for floor in (1,2,3):
            for budget in (10,13,18):
                w = allocate(trees,budget,floor)
                actual = sum(sum(n.mass for n in t[:z]) for t,z in zip(trees,w))
                ref = max(sum(sum(n.mass for n in t[:z]) for t,z in zip(trees,perm))
                          for perm in itertools.product(range(floor,9),repeat=3) if sum(perm)==budget)
                np.testing.assert_allclose(actual,ref,rtol=1e-13)
                checked["allocations_vs_exhaustive"]+=1
        for s,tree in zip(scores,trees):
            target=probabilities(s,0.6);distance=counterfactual(tree,target)[-1];mean=0.
            for path in itertools.product(range(2),repeat=3):
                pred=0;mass=1.;node=0;length=1
                for level,rank in enumerate(path):
                    mass*=target[level,pred,rank];pred=rank
                    if node is not None:
                        node=next((idx for idx,n in enumerate(tree) if n.parent==node and n.rank==rank),None)
                        if node is not None:length+=1
                mean+=mass*length
                checked["exhaustive_target_paths"]+=1
            np.testing.assert_allclose(mean,distance,rtol=1e-13)
    return checked


def study():
    rng=np.random.default_rng(92491)
    k=16;depth=7
    result={}
    for batch in (1,4,16,32):
        samples={}
        for _ in range(32):
            # Heterogeneous synthetic confidence across independent requests. No judge input.
            sigmas=rng.choice([1.25,2.5,4.5,8.0],batch)
            raw=[rng.normal(0,s,(depth,k,k)) for s in sigmas]
            trees=[native_tree(s,24) for s in raw]
            widths={f"pooled_f{f}_cap{cap}":allocate([t[:cap] for t in trees],batch*8,f)
                    for cap in (16,24) for f in (1,4,6)}
            widths.update({f"uniform_{w}":np.full(batch,w) for w in (8,16,24)})
            for beta in (0.5,1.,2.):
                ref=[counterfactual(t,probabilities(s,beta)) for t,s in zip(trees,raw)]
                for name,w in widths.items():
                    sums=[r[s-1] for r,s in zip(ref,w)]
                    old=np.array([r[7] for r in ref]);counts=np.asarray(sums)
                    samples.setdefault((beta,name),[]).extend(zip(counts,old,w))
        result[str(batch)]={}
        for (beta,name),values in samples.items():
            data=np.array(values);ratio=data[:,0]/data[:,1];width=data[:,2]
            result[str(batch)].setdefault(str(beta),{})[name]={
                "mean_commits":round(float(data[:,0].mean()),5),
                "mean_ratio_to_fixed8":round(float(data[:,0].sum()/data[:,1].sum()),5),
                "per_request_p10_ratio":round(float(np.quantile(ratio,.1)),5),
                "fraction_losing_expected_commits":round(float((ratio<1-1e-10).mean()),5),
                "min_max_widths":[int(width.min()),int(width.max())],
            }
    return result


if __name__ == "__main__":
    print(json.dumps({"synthetic":True,"native_depth":7,"fanout":16,"batches_per_geometry":32,
                      "tests":tests(),"results":study()},indent=2))
