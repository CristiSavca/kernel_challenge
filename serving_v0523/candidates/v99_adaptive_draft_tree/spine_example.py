"""Enumerate a synthetic eight-row DFLASH lattice against two target populations."""
import itertools
import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np

width, k = 8, 16
source = Path(sys.argv[1]) / "python/sglang/kernels/ops/speculative/dflash_tree.py"
emulator = runpy.run_path(str(Path(__file__).resolve().parents[1] / "v70_dflash_selector_tree/cpu_check.py"))
pointer = emulator["Pointer"]
_, (_, propose, accept) = emulator["program"](source, ("_edge_probabilities", "_propose_tree", "_accept_tree"))
candidate_ids = np.arange(1, (width - 1) * k + 1, dtype=np.int64).reshape(1, width - 1, k)
scores = np.broadcast_to(np.array([math.log(.60), math.log(.40)] + [-80.] * 14, dtype=np.float32),
                         (1, width - 1, k, k)).copy()
tree = np.zeros((4, 1, width), dtype=np.int64)
anchor = np.full((1,width), 655, dtype=np.int64)
propose(pointer(candidate_ids),pointer(scores),pointer(anchor),*[pointer(x) for x in tree],width,width,k,width*k)
chain = tree.copy()
nodes = np.arange(width)
chain[0,0,1:] = candidate_ids[0,:,0]
chain[1,0] = nodes
chain[2,0] = nodes-1
chain[3,0] = (1 << (nodes+1))-1


def expectation(policy, probability):
    mean = 0.
    for path in itertools.product((0,1), repeat=width-1):
        weight = math.prod(probability if r==0 else 1-probability for r in path)
        prediction = np.array([candidate_ids[0,d,path[d]] if d<width-1 else 777 for d in policy[1,0]],dtype=np.int64)
        length = np.zeros(1,np.int32)
        out = np.zeros(width,np.int64)
        accept(pointer(policy[0]),pointer(policy[2]),pointer(prediction),pointer(np.array([100])),pointer(length),
               pointer(np.zeros(1,np.int64)),pointer(out),pointer(np.zeros(1,np.int64)),pointer(np.zeros(width,np.int64)),width,width)
        n = min(int(length[0]),width-1)
        np.testing.assert_equal(out[:n],[candidate_ids[0,d,path[d]] for d in range(n)])
        mean += weight*int(length[0])
    return round(mean,6)


print(json.dumps(dict(synthetic=True, measured_judge_data=False,target_rows=width, candidate_ids_per_position=k,
    selector_top_two=[.60,.40],remaining_logits=-80,cases_per_population=2**(width-1),
    incumbent_tree_depths=tree[1,0].tolist(),native_chain_depths=chain[1,0].tolist(),
    expected_commits=[dict(target_top_choice_agreement=p,tree=expectation(tree,p),chain=expectation(chain,p)) for p in (.60,.94)]),indent=2))
