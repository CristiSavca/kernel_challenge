"""Run V68's real route planner and finalizer on either side of the emulated Hopper cluster."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

CANDIDATES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATES / "v122_clustered_expert_mlp"))
import independent_cpu as cluster


def main(v125, v122):
    v66_fixture = CANDIDATES / "v66_humming_grouped_decode/cpu_check.py"
    sys.argv[1:] = [str(v125)]
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        fix = runpy.run_path(str(v66_fixture))
    scope, tensor, bf16 = fix["scope"], fix["tensor"], fix["bf16"]
    scope["torch"].uint8 = np.uint8
    rng = np.random.default_rng(125)
    experts, width, n, k = 32, 512, 128, 8
    gate = cluster.weight_bank(rng, experts, 2*width, width)
    down = cluster.weight_bank(rng, experts, width, width)
    v86 = cluster.v86
    global13 = tensor(v86.floats(v86.bf16(rng.choice(np.array([.25,.5,1,1.5],np.float32),experts))))
    global2 = tensor(v86.floats(v86.bf16(rng.choice(np.array([.25,.5,1,1.5],np.float32),experts))))
    x = tensor(bf16(rng.integers(-5,6,(n,width)).astype(np.float32)/8))
    weights = tensor(rng.uniform(0.1,1,(n,k)).astype(np.float32))
    norm = tensor(bf16(rng.uniform(.75,1.25,width).astype(np.float32)))
    records = []
    with tempfile.TemporaryDirectory(prefix="v125-integration-") as scratch:
        dll, _ = cluster.compile_actual(v122, Path(scratch))
        def fused(rows,w13,s13,w2,s2,global_scale,offsets):
            bits = v86.bf16(np.asarray(rows))
            poison = bits[:offsets[0]].copy()
            jobs = sum((int(b)-int(a)+15)//16 for a,b in zip(offsets,offsets[1:]))+experts
            tiles = np.empty((jobs,3),np.uint32)
            arrays = [bits,w13,s13,w2,s2,v86.bf16(global_scale),offsets]
            dll.simulate(*(v.ctypes.data for v in arrays),width,width,experts,jobs,tiles.ctypes.data)
            np.testing.assert_array_equal(bits[:offsets[0]],poison)
            rows[:] = v86.floats(bits)
        scope["_cluster_mlp_module"] = lambda: NS(run=fused)
        for pattern in ("remote","mixed","skew","duplicates"):
            ids = tensor(np.full((n,k),250,np.int32))
            # Unsorted local ownership, interleaved peers, repeated slots and a long tile tail.
            mapping = tensor(np.full(256,-1,np.int32))
            mapping[[1,47,93,171]] = [31,0,9,7]
            if pattern=="mixed": ids[::5,2] = rng.choice([1,47,93,171],len(ids[::5,2]))
            if pattern=="skew": ids[:35,0] = 47
            if pattern=="duplicates": ids[:18,2:4] = 93
            layer = NS(humming_cluster_mlp=True,num_local_experts=experts,global_num_experts=256,
                       intermediate_size_per_partition=width,_expert_map=mapping,
                       w13_weight=tensor(gate[2]),w13_weight_scale=tensor(gate[3]),w13_global_scale=global13,
                       w2_weight=tensor(down[2]),w2_weight_scale=tensor(down[3]))
            output, indices, local = scope["humming_grouped_decode_outputs"](layer,x,ids)
            wanted = mapping[ids]
            np.testing.assert_array_equal(local,wanted)
            direct = tensor(np.full((n,k,width),np.nan,np.float32))
            for e in np.unique(wanted[wanted>=0]):
                rows, slots = np.where(wanted==e)
                gu = cluster.reference(np.asarray(x[rows]),gate[0][e],gate[1][e])
                left,right = np.split(v86.floats(v86.bf16(v86.floats(gu)*global13[e])),2,axis=1)
                with np.errstate(over="ignore"):
                    activation = v86.floats(v86.bf16((left/(1+np.exp(-left)))*right))
                raw = v86.floats(cluster.reference(activation,down[0][e],down[1][e]))
                np.testing.assert_array_equal(output[indices[rows,slots]].view(np.uint32),raw.view(np.uint32))
                direct[rows,slots] = bf16(raw*global2[e])
            for weight_first in (True,False):
                actual = scope["fused_post_expert_norm_weighted_sum"](
                    output,weights,norm,weight_first=weight_first,slot_expert_ids=local,
                    slot_row_indices=indices,out=x.new_empty(x.shape),expert_output_scales=global2)
                expected = scope["fused_post_expert_norm_weighted_sum"](
                    direct,weights,norm,weight_first=weight_first,slot_expert_ids=wanted)
                np.testing.assert_array_equal(np.asarray(actual).view(np.uint32),np.asarray(expected).view(np.uint32))
            records.append(dict(pattern=pattern,live_routes=int((wanted>=0).sum()),d_model=width,
                                combined_bf16_values=2*n*width))
    print(json.dumps(dict(cases=records,real_sm90=False,full_graph=False),indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]))
