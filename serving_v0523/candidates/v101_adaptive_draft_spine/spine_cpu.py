"""Execute the new Triton bodies on CPU and compare to explicit, path-based trees."""
import argparse
import itertools
import json
import runpy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
HARNESS = runpy.run_path(str(HERE.parent / "v70_dflash_selector_tree" / "cpu_check.py"))
ptr, array = HARNESS["Pointer"], HARNESS["array"]


def make_family(original, chain):
    width = len(chain)
    paths = [()]
    for slot in range(1, width):
        paths.append(paths[original[2, slot]] + (int(original[0, slot]),))
    assert len(set(paths)) == width
    reference = []
    for length in range(width):
        native = [tuple(chain[1:depth + 1]) for depth in range(length + 1)]
        if all(path in paths for path in native):
            order = paths
        else:
            order = native + [path for path in paths if path not in native][:width - len(native)]
        data = np.empty((4, width), dtype=np.int64)
        for j, path in enumerate(order):
            parent = order.index(path[:-1]) if path else -1
            data[:, j] = (path[-1] if path else chain[0], len(path), parent,
                          ((data[3, parent] if parent >= 0 else 0) | (1 << j)))
            assert parent < j
        reference.append((data, set(order)))
    return reference


def rewards(family, continuation):
    width = len(family)
    return np.array([min(width, 1 + sum(tuple(continuation[:d]) in paths
                                         for d in range(1, min(width - 1, len(continuation)) + 1)))
                     for _, paths in family], dtype=np.int32)


def updated_oracle(family, prior, sequence, observed):
    width = len(family)
    known = rewards(family, sequence[:observed])
    actual = rewards(family, sequence)
    out = np.zeros_like(prior)
    for s in range(1, width):
        assert abs(actual[s] - actual[s-1]) <= 1
        if known[s] != known[s-1]:
            assert known[s] - known[s-1] == actual[s] - actual[s-1]
        if (family[s][1] != family[s-1][1] and known[s] == known[s-1] == observed+1 < width):
            out[s] = prior[s]
        else:
            assert known[s] - known[s-1] == actual[s] - actual[s-1]
            out[s] = np.sign(prior[s]) * (3*abs(prior[s])//4) + 4*(known[s] - known[s-1])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    names = ("_native_prefix_nodes", "_select_spine_tree", "_update_spine_margins", "_accept_tree")
    pid, (_, select, update, accept) = HARNESS["program"](args.source, names)
    select.__globals__["tl"].gather = lambda x, indexes, axis: array(np.take(x, indexes, axis=axis))
    rng = np.random.default_rng(1729)
    counts = {"tree_topologies": 0, "kernel_score_updates": 0, "target_walks": 0,
              "adjacent_full_reward_bounds": 0, "censored_oracle_updates": 0}
    best = None
    for width, case in itertools.product(range(2, 9), range(16)):
        block = 1 << (width-1).bit_length()
        batch = 3
        original = np.empty((4,batch,width), dtype=np.int64)
        chain = np.empty((batch,width), dtype=np.int64)
        families = []
        for b in range(batch):
            k = (2,4,16)[b]
            # Reusing IDs at multiple depths and under many parents exercises path identity.
            ids = np.vstack([rng.permutation(k) + (1 << 34) for _ in range(width-1)])
            score = rng.normal(size=(width-1,k,k)).astype(np.float32) * (0 if case==0 else 2.5)
            original[:,b] = HARNESS["proposal_oracle"](ids, score, 19)
            rank = 0
            chain[b,0] = 19
            for depth in range(width-1):
                rank = np.argmax(score[depth,rank])
                chain[b,depth+1] = ids[depth,rank]
            families.append(make_family(original[:,b], chain[b]))
        for s in range(width):
            chosen = [(s+b)%width for b in range(batch)]
            margins = np.zeros((batch,width),dtype=np.int32)
            for b, length in enumerate(chosen):
                if length:
                    margins[b,length] = 2
            selected = np.full_like(original, -1717)
            for b in range(batch):
                pid[0]=b
                select(*map(ptr,(original,chain,margins,selected)), batch*width,width,block)
                np.testing.assert_array_equal(selected[:,b],families[b][chosen[b]][0])
                counts["tree_topologies"] += 1

            targets = np.full_like(chain, -45678)
            seqs, preds = [], np.zeros_like(chain)
            for b in range(batch):
                paths = sorted(set.union(*(paths for _,paths in families[b])))
                path = paths[rng.integers(len(paths))]
                seq = [*path, *([-999]*(width-len(path)))]
                seqs.append(seq)
                predicted_paths = [()]
                for j in range(1,width):
                    predicted_paths.append(predicted_paths[selected[2,b,j]]+(selected[0,b,j],))
                for j,path in enumerate(predicted_paths):
                    preds[b,j] = seq[len(path)] if tuple(seq[:len(path)])==path else -888
            prefix = np.arange(batch,dtype=np.int64)*13
            lens=np.zeros(batch,dtype=np.int32)
            bonus=np.zeros(batch,dtype=np.int64)
            newlens=bonus.copy()
            order=np.zeros_like(targets)
            for b in range(batch):
                pid[0]=b
                accept(*map(ptr,(selected[0],selected[2],preds,prefix,lens,bonus,targets,newlens,order)),width,block)
                expected = rewards(families[b],seqs[b])[chosen[b]]
                assert lens[b] == expected and newlens[b] == prefix[b]+expected
                np.testing.assert_array_equal(targets[b,:expected], seqs[b][:expected])
                assert sorted(order[b]) == list(range(width))
                counts["target_walks"]+=1
            prior = rng.integers(-16,17,(batch,width),dtype=np.int32)
            prior[:,0]=0
            results=np.full_like(prior,-999)
            for b in range(batch):
                pid[0]=b
                update(*map(ptr,(original[0],original[2],chain,targets,lens,prior,results)),width,block)
                oracle=updated_oracle(families[b], prior[b], seqs[b], lens[b])
                np.testing.assert_array_equal(results[b],oracle)
                counts["kernel_score_updates"]+=1
        for b,family in enumerate(families):
            paths = sorted(set.union(*(p for _,p in family)))
            observed_rewards=[]
            for path in paths:
                seq=[*path,*([-222]*(width-len(path)))]
                exact=rewards(family,seq)
                observed_rewards.append(exact)
                assert np.all(np.abs(np.diff(exact))<=1)
                counts["adjacent_full_reward_bounds"]+=width-1
                for observed in range(1,width+1):
                    updated_oracle(family,np.arange(width)//2,seq,observed)
                    counts["censored_oracle_updates"]+=1
            if width==8:
                # Search two synthetic output modes for an explanatory witness only.
                values=np.array(observed_rewards)
                for prob in (.25,.4,.5,.6,.75):
                    exp = prob*values[:,None,:]+(1-prob)*values[None,:,:]
                    base = np.maximum(exp[:,:,0],exp[:,:,-1])
                    ratio=exp.max(axis=2)/base
                    ix=np.unravel_index(np.argmax(ratio),ratio.shape)
                    if best is None or float(ratio[ix]) > best["hybrid_over_best_endpoint"]:
                        best={"width":width,"synthetic":True,"hybrid_over_best_endpoint":float(ratio[ix]),
                              "mixture_probability":prob,"first_path":list(map(int,paths[ix[0]])),
                              "other_path":list(map(int,paths[ix[1]])),"native_tokens":chain[b].tolist(),
                              "base_tree":original[:,b].tolist(),"mean_commits_by_min_spine":exp[ix].tolist()}
    print(json.dumps({**counts,"synthetic_witness":best},indent=2))


if __name__ == "__main__":
    main()
