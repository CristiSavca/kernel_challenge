"""Independent owner/coverage checks against the V89 CUDA computation body."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re


HDR="python/sglang/kernels/jit/csrc/muse_spark/expert_skinny_gate_up.cuh"


def body(text, function):
    begin=text.index("{",text.index("void "+function+"("))
    depth=1
    for i in range(begin+1,len(text)):
        depth+=(text[i]=="{")-(text[i]=="}")
        if depth==0:
            return text[begin+1:i]
    raise ValueError(function)


def main(stage):
    root=Path(__file__).resolve().parent.parent
    patch=(root/"v89_paired_expert_dequant/patch.diff").read_text()
    section=patch.split("diff --git a/"+HDR+" b/"+HDR+"\n")[1].split("diff --git",1)[0]
    assert "new file mode" in section
    old="\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))+"\n"
    new=(stage/HDR).read_text()
    old_math=body(old,"run_kernel")
    for line in ("  const uint32_t expert = blockIdx.y;\n","  const uint32_t column = blockIdx.x * kWidth;\n"):
        assert old_math.count(line)==1
        old_math=old_math.replace(line,"")
    assert old_math==body(new,"project_tile")
    assert old_math.count("__syncthreads();")==2
    assert old_math.rstrip().endswith("__syncthreads();\n  }")
    new_sched=body(new,"run_active_kernel")
    for term in ("lane < p.experts && p.offsets[lane] != p.offsets[lane + 1]",
                 "__ballot_sync(0xffffffffu, active)", "__popc(mask & ((1u << lane) - 1u))",
                 "ordinal += gridDim.y", "__ballot_sync(0xffffffffu, active && rank == ordinal)",
                 "__ffs(owner) - 1", "project_tile(p, expert, blockIdx.x * kWidth)"):
        assert term in new_sched
    rng=random.Random(67824)
    counts={"masks":0,"active_grid_and_offset_layouts":0,"multi_iteration_layouts":0,"edge31_layouts":0,"all_idle_layouts":0}
    for e in (1,2,4,8,12,16,31,32):
        masks=range(1<<e) if e<=12 else [0,(1<<e)-1,1<<(e-1),1,0xaaaaaaaa&((1<<e)-1)]+[rng.randrange(1<<e) for _ in range(768)]
        for mask in masks:
            counts["masks"]+=1
            alive=[lane for lane in range(e) if mask&(1<<lane)]
            active_to_lane={}
            for lane in alive:
                rank=(mask&((1<<lane)-1)).bit_count()
                active_to_lane[rank]=lane
            assert [active_to_lane[i] for i in range(len(alive))]==alive
            for expected in range(1,(e-1)//2+1):
                grid=2*expected
                source=[]
                for y in range(grid):
                    work=list(range(y,len(alive),grid))
                    for ordinal in work:
                        # __ffs returns the unique lane because rank is exclusive prefix popcount.
                        lane=active_to_lane[ordinal]
                        source.append(lane)
                        assert lane==alive[ordinal]
                assert sorted(source)==alive and len(set(source))==len(alive)
                counts["active_grid_and_offset_layouts"]+=1
                counts["multi_iteration_layouts"]+=len(alive)>grid
                counts["edge31_layouts"]+=31 in source
                counts["all_idle_layouts"]+=not alive
    # Production sorted offsets make the row intervals a partition, including repeated empties.
    for routes in ([0]*32,[0]*31+[63],[3,0,0,1]*8,[1]*32,[8,9,15,16]*8):
        start=0; rows=[]
        for count in routes:
            for tile in range(start,start+count,8):
                rows.extend(range(tile,min(start+count,tile+8)))
            start+=count
        assert rows==list(range(sum(routes)))
    return {"header_sha256":hashlib.sha256(new.encode()).hexdigest(),"v89_math_body_byte_identical":True,
            "barriers_per_row_tile":2,"counts":counts,"native":{"global_experts":256,"local_experts":32,
            "intermediate":4096,"n_bands":64,"ctas_one_user_verify_before":2048,
            "ctas_one_user_verify_after":1024,"two_user_expected_local_routes":16,"active_threshold":15},
            "hardware_run":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    print(json.dumps(main(parser.parse_args().stage),indent=2))
