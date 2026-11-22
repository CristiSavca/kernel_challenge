"""Check stock endpoint equivalence, strict interior choices and new score storage."""
import argparse
import json
import runpy
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent / "v101_adaptive_draft_spine"
ref = runpy.run_path(str(BASE / "spine_cpu.py"))
HARNESS, ptr, array = ref["HARNESS"], ref["ptr"], ref["array"]
family_of, rewards, adjacent = ref["make_family"], ref["rewards"], ref["updated_oracle"]


def choice(margins):
    width=len(margins)
    utility=np.concatenate(([0],np.cumsum(margins[1:])))
    endpoints=max(utility[0],utility[-1])
    if width>2 and max(utility[1:-1])>endpoints:
        return 1+int(np.argmax(utility[1:-1]))
    return width-1 if margins[0]>0 else 0


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('source',type=Path)
    args=cli.parse_args()
    names=("_native_prefix_nodes","_select_spine_tree","_update_spine_margins","_update_tree_advantage")
    _,(_,select,update,incumbent)=HARNESS["program"](args.source,names)
    select.__globals__["tl"].gather=lambda x,index,axis:array(np.take(x,index,axis=axis))
    rng=np.random.default_rng(91746)
    counts={"endpoint_selections":0,"strict_interior_selections":0,"scalar_update_matches":0,
            "adjacent_update_matches":0,"width_two_checks":0}
    for width in range(2,9):
        block=1<<(width-1).bit_length()
        for case in range(18):
            k=(2,4,16)[case%3]
            ids=np.vstack([rng.permutation(k)+(1<<33) for _ in range(width-1)])
            scores=rng.normal(size=(width-1,k,k)).astype(np.float32)
            original=HARNESS["proposal_oracle"](ids,scores,17)
            rank=0
            chain=[17]
            for i in range(width-1):
                rank=int(np.argmax(scores[i,rank]))
                chain.append(ids[i,rank])
            chain=np.array(chain,dtype=np.int64)
            family=family_of(original,chain)
            paths=sorted(set.union(*(x for _,x in family)))
            for test in range(13):
                state=rng.integers(-16,17,width,dtype=np.int32)
                state[0]=rng.integers(-width,width+1)
                if test<3: # All adjacent utilities tie, including both endpoints.
                    state[1:]=0
                    state[0]=(-width,0,width)[test]
                elif test<6: # Interior equals or crosses a single endpoint.
                    state[1:]=0
                    state[1]=6
                    state[-1]=(-1,0,1)[test-3]
                chosen=choice(state)
                tree=np.empty_like(original)
                select(*map(ptr,(original,chain,state,tree)),width,width,block)
                np.testing.assert_array_equal(tree,family[chosen][0])
                counts["strict_interior_selections" if 0<chosen<width-1 else "endpoint_selections"]+=1
                counts["width_two_checks"]+=int(width==2)
                path=list(paths[rng.integers(len(paths))])
                full=path+[-981]*(width-len(path))
                obs=int(rewards(family,full)[chosen])
                committed=np.full(width,-814757,dtype=np.int64)
                committed[:obs]=full[:obs]
                lengths=np.array([obs],dtype=np.int32)
                new=np.empty_like(state)
                scalar=np.zeros(1,dtype=np.int32)
                args_=(original[0],original[2],chain,committed,lengths)
                update(*map(ptr,(*args_,state,new)),width,block)
                incumbent(*map(ptr,(*args_,state[:1],scalar)),width,block)
                assert new[0]==scalar[0]
                np.testing.assert_array_equal(new[1:],adjacent(family,state,full,obs)[1:])
                counts["scalar_update_matches"]+=1
                counts["adjacent_update_matches"]+=width-1
    print(json.dumps(counts,indent=2))


if __name__=="__main__":
    main()
