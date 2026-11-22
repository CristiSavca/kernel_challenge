"""Prespecified paired IID-token replay on V99's binary 0.60/0.40 selector lattice."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from spine_cpu import HARNESS, HERE, array, make_family, ptr


W, K, SEEDS, STEPS = 8, 16, 4096, 64
PROBS = (.60, .70, .80, .94)


def v99_source():
    patch = (HERE.parent / "v99_adaptive_draft_tree/patch.diff").read_text()
    section = next(x for x in re.split(r"(?=^diff --git )", patch, flags=re.M)
                   if x.splitlines() and "ops/speculative/dflash_tree.py " in x.splitlines()[0])
    assert "new file mode" in section
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))


def functions(source):
    return {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def transition_tables(stage):
    file = stage / "python/sglang/kernels/ops/speculative/dflash_tree.py"
    original, current = functions(v99_source()), functions(file.read_text())
    same = ("_edge_probabilities", "_propose_tree", "_accept_tree", "_select_chain_or_tree", "_update_tree_advantage")
    assert all(original[n] == current[n] for n in same)
    worker = (stage / "python/sglang/srt/speculative/dflash_worker_v2.py").read_text()
    assert "original_tree, linear_tokens, out_tokens, commit_lens, policy_advantage" in worker
    names = (*same, "_native_prefix_nodes", "_select_spine_tree", "_update_spine_margins")
    _, (_, propose, accept, old_select, old_update, _, new_select, new_update) = HARNESS["program"](file, names)
    new_select.__globals__["tl"].gather = lambda x, idx, axis: array(np.take(x, idx, axis=axis))
    candidates = np.arange(1, (W - 1)*K+1, dtype=np.int64).reshape(W-1, K)
    logits = np.broadcast_to(np.array([np.log(.6), np.log(.4)] + [-80.]*14, dtype=np.float32), (W-1,K,K))
    tree = np.zeros((4,W), dtype=np.int64)
    anchors = np.full(W,655,dtype=np.int64)
    propose(ptr(candidates),ptr(logits),ptr(anchors),*[ptr(t) for t in tree],W,W,K,W*K)
    chain = np.r_[anchors[0], candidates[:,0]]
    family = make_family(tree, chain)

    def select(fn, state):
        selected = np.full_like(tree, -1)
        fn(*map(ptr, (tree, chain, state, selected)), W,W,W)
        return selected

    for prior in range(-W,W+1):
        np.testing.assert_array_equal(select(old_select, np.array([prior],np.int32)),family[7 if prior>0 else 0][0])
    for depth in range(W):
        state = np.zeros(W, dtype=np.int32)
        state[depth] = 1 if depth else 0
        np.testing.assert_array_equal(select(new_select, state), family[depth][0])

    reward = np.zeros((2**(W-1),W), dtype=np.int16)
    observed = np.zeros_like(reward)
    evidence, using_evidence = {}, []
    for path in range(reward.shape[0]):
        choices = (path >> np.arange(W-1)) & 1
        truth = np.r_[candidates[np.arange(W-1),choices], 777]
        for policy,(topology,_) in enumerate(family):
            commit = np.zeros(1,dtype=np.int32)
            tokens = np.full(W,-8192,dtype=np.int64)
            predictions = truth[topology[1]]
            accept(ptr(topology[0]),ptr(topology[2]),ptr(predictions),ptr(np.array([100],np.int64)),ptr(commit),
                   ptr(np.zeros(1,np.int64)),ptr(tokens),ptr(np.zeros(1,np.int64)),ptr(np.zeros(W,np.int64)),W,W)
            n = int(commit[0])
            np.testing.assert_array_equal(truth[:n], tokens[:n])
            reward[path,policy] = n
            key = tuple(tokens[:n])
            if key not in evidence:
                evidence[key] = len(evidence)
                using_evidence.append(key)
            observed[path,policy] = evidence[key]
    old = np.empty((len(evidence),17), dtype=np.int32)
    new = np.empty((len(evidence),33,W), dtype=np.int32)
    for i,visible in enumerate(using_evidence):
        # Only the chosen proposal's committed prefix and its bonus are observable.
        tokens = np.full(W,-918273645,dtype=np.int64)
        tokens[:len(visible)] = visible
        args = [ptr(t) for t in (tree[0],tree[2],chain,tokens,np.array([len(visible)],np.int32))]
        for prior in range(-8,9):
            result = np.empty(1,dtype=np.int32)
            old_update(*args,ptr(np.array([prior],np.int32)),ptr(result),W,W)
            old[i,prior+8] = result[0]
        for prior in range(-16,17):
            # The deployed update reads each lane's own prior; no prior is gathered across lanes.
            new_update(*args,ptr(np.full(W,prior,dtype=np.int32)),ptr(new[i,prior+16]),W,W)
        for shift in range(33):
            mixed=((np.arange(W)*5+shift)%33-16).astype(np.int32)
            next_mixed=np.zeros_like(mixed)
            new_update(*args,ptr(mixed),ptr(next_mixed),W,W)
            np.testing.assert_array_equal(next_mixed,new[i,mixed+16,np.arange(W)])
            policy=int(np.argmax(np.cumsum(mixed)))
            np.testing.assert_array_equal(select(new_select,mixed),family[policy][0])
    assert np.max(np.abs(old))<=8 and np.max(np.abs(new))<=16
    return reward, observed, old, new, hashlib.sha256(file.read_bytes()).hexdigest()


def expectation(reward, p):
    fallback = np.array([x.bit_count() for x in range(len(reward))])
    return (p**(7-fallback) * (1-p)**fallback) @ reward


def replay(draws, probs, reward, evidence, old, new):
    ids = (((draws >= probs[...,None]).astype(np.int16)) << np.arange(W-1)).sum(axis=2)
    old_prior = np.zeros(SEEDS,dtype=np.int32)
    new_prior = np.zeros((SEEDS,W),dtype=np.int32)
    committed = np.zeros((SEEDS,STEPS,4),dtype=np.int16)
    depths = np.zeros((SEEDS,STEPS),dtype=np.int16)
    for i in range(STEPS):
        path = ids[:,i]
        old_policy = np.where(old_prior>0, W-1, 0)
        # The submitted selector chooses the leftmost maximum of the prefix sums.
        choice = np.argmax(np.cumsum(new_prior,axis=1),axis=1)
        committed[:,i,0] = reward[path,0]
        committed[:,i,1] = reward[path,-1]
        committed[:,i,2] = reward[path,old_policy]
        committed[:,i,3] = reward[path,choice]
        depths[:,i] = choice
        old_prior = old[evidence[path,old_policy],old_prior+8]
        new_prior = new[evidence[path,choice,None],new_prior+16,np.arange(W)]
    results = {}
    for horizon in (8,64):
        rates = committed[:,:horizon,:].mean(axis=1)
        base, chain, v99, v101 = rates.mean(axis=0)
        per_request = rates[:,3]-rates[:,2]
        delta = per_request.mean()
        radius = 1.96*per_request.std(ddof=1)/np.sqrt(SEEDS)
        results[str(horizon)] = {
            "fixed_tree": float(base), "fixed_chain": float(chain), "v99": float(v99), "v101": float(v101),
            "delta_rows": float(delta), "relative_rows": float(v101/v99),
            "paired_delta_95pct_normal_ci": [float(delta-radius),float(delta+radius)],
            "selected_depth_counts_v101": np.bincount(depths[:,:horizon].ravel(),minlength=W).tolist(),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    args=parser.parse_args()
    reward, evidence, old, new, source_sha = transition_tables(args.stage)
    draws=np.stack([np.random.Generator(np.random.PCG64(i)).random((STEPS,W-1)) for i in range(SEEDS)])
    regimes={f"stationary_{p:.2f}":np.full((1,1),p) for p in PROBS}
    for period in (1,8):
        phase=(np.arange(SEEDS)[:,None]%2+np.arange(STEPS)[None,:]//period)%2
        regimes[f"alternate_{period}_calls"]=np.where(phase,.94,.60)
    records={}
    for name,probs in regimes.items():
        distinct,counts=np.unique(probs,return_counts=True)
        means=np.stack([expectation(reward,p) for p in distinct])
        mixture=counts/counts.sum()
        fixed=mixture@means
        records[name]={
            "oracle_best_fixed_overall":float(fixed.max()), "oracle_best_fixed_per_probability":float(mixture@means.max(1)),
            "mean_by_fixed_spine":fixed.tolist(), "results":replay(draws,probs,reward,evidence,old,new),
        }
    print(json.dumps({"synthetic":True,"source_sha256":source_sha,"width":W,"lattice":[.6,.4],
                        "remaining_14_logits":-80,"seeds":[0,SEEDS-1],"generator":"NumPy PCG64",
                        "paired_unit":"same request, verification call, and upcoming token-depth draws",
                        "fixed_initial_state":0,"verified_paths_and_policies":list(reward.shape),
                        "distinct_censored_prefixes":len(old),"results":records},indent=2))


if __name__=="__main__":
    main()
