"""A deterministic width-eight conditional lattice with contrasting greedy continuations."""

import json
from pathlib import Path
import runpy
import sys

import numpy as np


def run(source):
    helper = runpy.run_path(str(Path(__file__).with_name("cpu_check.py")))
    pid, kernels = helper["make_program"](source)
    ptr = helper["ptr"]
    width, k = 8, 16
    scores = np.full((width-1,k,k), -np.inf, np.float32)
    scores[:, :, :2] = np.log([.6,.4])
    scores[0,0,:2] = np.log([.8,.2])
    ids = np.broadcast_to(np.arange(k, dtype=np.int64), (width-1,k)).copy()
    incumbent = np.empty((4,width), np.int64)
    alts = np.empty((2,4,width), np.int64)
    pid[0] = 0
    for variant in range(3):
        pid[1] = variant
        kernels["_propose_tree"](ptr(ids),ptr(scores),ptr(np.array([42],np.int64)),
            *map(ptr, incumbent),1,width,k,width*k,ptr(alts),width)
    options = [incumbent, *alts]
    sets = [helper["paths"](t) for t in options]
    by_case = {}
    for seq in [(1,0,1,1,1,1,1,1),(0,0,0,0,0,0,0,0),(0,1,0,1,1,1,1,1)]:
        by_case["".join(map(str,seq))] = [helper["reward"](tree,list(seq)) for tree in options]
    rewards = list(by_case.values())
    assert rewards[0][1] > rewards[0][0]
    assert rewards[1][2] > rewards[1][0]
    assert rewards[2][0] >= max(rewards[2][1:])
    return {"width":width,"conditional_candidates":k,"root_probabilities":[0.8,0.2],
        "all_later_probabilities":[0.6,0.4],"remaining_logits":"-infinity","beta_order":[1.0,.5,2.0],
        "proposal_paths_by_beta":sets,"committed_per_target_in_beta_order":by_case,
        "conclusion":"Synthetic deterministic target continuations benefit different proposals; these do not estimate real acceptance."}


if __name__ == "__main__":
    source = Path(sys.argv[1]) / "python/sglang/kernels/ops/speculative/dflash_tree.py"
    print(json.dumps(run(source),indent=2))
