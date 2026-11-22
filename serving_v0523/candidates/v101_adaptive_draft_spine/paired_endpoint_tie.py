"""Counterfactual controller. Override its V99 endpoint only for a strict interior win."""

import json
from pathlib import Path
import sys

import numpy as np
from paired_binary_replay import HERE, PROBS, SEEDS, STEPS, W, transition_tables


def stats(values):
    mean = values.mean()
    radius = 1.96*values.std(ddof=1)/np.sqrt(SEEDS)
    return {"mean": float(mean), "paired_95pct_delta_ci": [float(mean-radius),float(mean+radius)]}


def simulate(draws, probs, reward, evidence, old, new):
    ids = (((draws >= probs[...,None]).astype(np.int16)) << np.arange(W-1)).sum(axis=2)
    original_end = np.zeros(SEEDS,dtype=np.int32)
    original_spine = np.zeros((SEEDS,W),dtype=np.int32)
    guarded_end = np.zeros_like(original_end)
    guarded_spine = np.zeros_like(original_spine)
    result = np.zeros((SEEDS,STEPS,4),dtype=np.int16)
    for i in range(STEPS):
        path = ids[:,i]
        endpoint = np.where(original_end>0, W-1, 0)
        spine = np.argmax(np.cumsum(original_spine,axis=1),axis=1)
        utilities = np.cumsum(guarded_spine,axis=1)
        inner = 1+np.argmax(utilities[:,1:-1],axis=1)
        inner_value = np.take_along_axis(utilities,inner[:,None],axis=1)[:,0]
        strict = inner_value > np.maximum(utilities[:,0],utilities[:,-1])
        selected = np.where(strict, inner, np.where(guarded_end>0,W-1,0))
        result[:,i,:]=np.column_stack((reward[path,endpoint],reward[path,spine],reward[path,selected],strict))
        original_end = old[evidence[path,endpoint],original_end+8]
        original_spine = new[evidence[path,spine,None],original_spine+16,np.arange(W)]
        # Both counterfactual states use only the evidence exposed by its actual choice.
        chosen_evidence = evidence[path,selected]
        guarded_end = old[chosen_evidence,guarded_end+8]
        guarded_spine = new[chosen_evidence[:,None],guarded_spine+16,np.arange(W)]
    answer = {}
    for horizon in (8,64):
        values = result[:,:horizon].mean(axis=1)
        v99, v101, guarded, override = values.mean(axis=0)
        answer[str(horizon)]={
            "v99":float(v99),"v101":float(v101),"strict_interior":float(guarded),
            "strict_over_v99":float(guarded/v99),"strict_over_v101":float(guarded/v101),
            "fraction_interior_overrides":float(override),
            "delta_strict_minus_v99":stats(values[:,2]-values[:,0]),
            "delta_strict_minus_v101":stats(values[:,2]-values[:,1]),
        }
    return answer


def main():
    reward, evidence, old, new, source_sha = transition_tables(Path(sys.argv[1]))
    draws=np.stack([np.random.Generator(np.random.PCG64(i)).random((STEPS,W-1)) for i in range(SEEDS)])
    regimes={f"stationary_{p:.2f}":np.full((1,1),p) for p in PROBS}
    for period in (1,8):
        phase=(np.arange(SEEDS)[:,None]%2+np.arange(STEPS)[None,:]//period)%2
        regimes[f"alternate_{period}_calls"]=np.where(phase,.94,.60)
    original=json.loads((HERE/"paired_binary_replay.json").read_text())
    records={name:simulate(draws,probs,reward,evidence,old,new) for name,probs in regimes.items()}
    for name, result in records.items():
        for horizon, values in result.items():
            for field in ("v99","v101"):
                assert original["results"][name]["results"][horizon][field] == values[field]
    print(json.dumps({"synthetic":True,"unimplemented_policy":True,"source_sha256":source_sha,
                      "seeds":[0,SEEDS-1],"matched_v99_v101_means_exactly":True,
                      "controller":"Use own V99 endpoint unless an interior utility strictly exceeds both endpoints.",
                      "results":records},indent=2))


if __name__=="__main__":
    main()
