"""Run request-local feedback on a fixed synthetic distribution using real Triton bodies on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np

import sys

HERE=Path(__file__).resolve().parent.parent/"v101_adaptive_draft_spine"
sys.path.insert(0,str(HERE))
from spine_cpu import HARNESS, array, make_family, ptr, rewards
from guard_cpu import choice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    witness=json.loads((HERE/"spine_cpu.json").read_text())["synthetic_witness"]
    original=np.array(witness["base_tree"],dtype=np.int64)
    chain=np.array(witness["native_tokens"],dtype=np.int64)
    width=len(chain)
    family=make_family(original,chain)
    functions=("_native_prefix_nodes","_select_spine_tree","_update_spine_margins","_update_tree_advantage")
    _,(_,select,update,old_update)=HARNESS["program"](args.source,functions)
    select.__globals__["tl"].gather=lambda x, idx, axis: array(np.take(x,idx,axis=axis))
    per_iteration=np.zeros((64,4),dtype=np.int64)
    depth_counts=np.zeros(width,dtype=np.int64)
    for seed in range(32):
        rng=np.random.default_rng(seed)
        margin=np.zeros(width,dtype=np.int32)
        old_margin=np.zeros(1,dtype=np.int32)
        for step in range(len(per_iteration)):
            first = rng.random() < witness["mixture_probability"]
            prefix=witness["first_path" if first else "other_path"]
            seq=np.array(prefix+[-999]*(width-len(prefix)),dtype=np.int64)
            actual=rewards(family,seq)
            old_policy=width-1 if old_margin[0]>0 else 0
            selected=np.full_like(original,-1)
            select(*map(ptr,(original,chain,margin,selected)),width,width,width)
            chosen=choice(margin)
            np.testing.assert_array_equal(selected,family[chosen][0])
            depth_counts[chosen]+=1
            per_iteration[step]+=actual[[0,-1,old_policy,chosen]]
            # Both policies see exactly their verified prefix including one bonus.
            for kernel, length, previous in ((update,actual[chosen],margin),
                                             (old_update,actual[old_policy],old_margin)):
                committed=np.full(width, 92460851,dtype=np.int64)
                committed[:length]=seq[:length]
                new=np.zeros_like(previous)
                kernel(*map(ptr,(original[0],original[2],chain,committed,
                                 np.array([length],dtype=np.int32),previous,new)),width,width)
                previous[:]=new
    names=["fixed_tree","fixed_chain","v99_adaptive","v101b_guarded_spine"]
    avgs=dict(zip(names,(per_iteration.sum(axis=0)/(32*64)).tolist()))
    warm=dict(zip(names,(per_iteration[8:].sum(axis=0)/(32*56)).tolist()))
    print(json.dumps({"synthetic":True,"independent_sequences":32,"decode_calls_per_request":64,
                      "mean_committed":avgs,"after_first_8_calls":warm,
                      "chosen_depth_counts":depth_counts.tolist(),
                      "realized_commits_relative_to_v99":avgs[names[-1]]/avgs[names[-2]]},indent=2))


if __name__ == "__main__":
    main()
