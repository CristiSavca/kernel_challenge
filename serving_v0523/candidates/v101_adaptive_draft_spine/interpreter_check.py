"""Execute staged Triton kernels with TRITON_INTERPRET=1 and writable NumPy storage."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


def main():
    os.environ["TRITON_INTERPRET"] = "1"
    import numpy as np
    import triton

    parser = argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    args=parser.parse_args()
    module=ast.parse(args.source.read_text())
    names={"_native_prefix_nodes","_select_spine_tree","_update_spine_margins"}
    module.body=[item for item in module.body if getattr(item,"name",None) in names]
    assert len(module.body)==3
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs"))
    from interpret_decode_kernels import buf
    from triton import language as tl

    def view(x):
        dtype = {"int64": tl.int64, "int32": tl.int32}[x.dtype.name]
        return buf(x, dtype)

    rng=np.random.default_rng(20261118)
    runs=0
    with tempfile.TemporaryDirectory() as temp:
        file=Path(temp)/"spine_kernels.py"
        file.write_text("import triton\nimport triton.language as tl\n"+ast.unparse(module))
        spec=importlib.util.spec_from_file_location("spine_kernels",file)
        kernels=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernels)
        for width in range(2,9):
            batch=3
            block=triton.next_power_of_2(width)
            for case in range(12):
                original=np.zeros((4,batch,width),dtype=np.int64)
                chain=np.zeros((batch,width),dtype=np.int64)
                prior=rng.integers(-15,16,(batch,width),dtype=np.int32)
                prior[:,0]=0
                actual=np.zeros_like(chain)
                lengths=rng.integers(1,width+1,batch,dtype=np.int32)
                expected=np.zeros_like(original)
                next_scores=np.zeros_like(prior)
                for b in range(batch):
                    # Unique children under each parent. Paths may reuse an ID with another parent.
                    paths=[()]
                    for node in range(1,width):
                        p=int(rng.integers(node))
                        tok=int(rng.integers(2,9))
                        while (*paths[p],tok) in paths:
                            tok+=1
                        paths.append((*paths[p],tok))
                    chain[b,0]=13
                    spine=list(paths[rng.integers(width)])
                    spine+=(rng.integers(2,9,width-1-len(spine))).tolist()
                    chain[b,1:]=spine
                    native=[tuple(spine[:n]) for n in range(width)]
                    def planes(order):
                        slab=np.zeros((4,width),dtype=np.int64)
                        for n,path in enumerate(order):
                            parent=order.index(path[:-1]) if path else -1
                            mask=(slab[3,parent] if parent>=0 else 0)|(1<<n)
                            slab[:,n]=(path[-1] if path else 13,len(path),parent,mask)
                        return slab
                    original[:,b]=planes(paths)
                    policies=[]
                    for s in range(width):
                        forced=native[:s+1]
                        order=(list(paths) if all(p in paths for p in forced) else
                               forced+[p for p in paths if p not in forced][:width-len(forced)])
                        policies.append(order)
                    selected=int(np.argmax(np.cumsum(prior[b])))
                    expected[:,b]=planes(policies[selected])
                    observed_paths=sorted(set(paths+native))
                    current=list(observed_paths[rng.integers(len(observed_paths))])
                    current+=[-111]*(width-len(current))
                    obs=int(lengths[b])
                    actual[b,:obs]=current[:obs]
                    actual[b,obs:]=171771
                    rewards=[min(width,1+sum(tuple(current[:d]) in tree for d in range(1,min(width-1,obs)+1)))
                             for tree in policies]
                    for s in range(1,width):
                        old=int(prior[b,s])
                        censored=(set(policies[s])!=set(policies[s-1])
                                  and obs+1<width and rewards[s]==rewards[s-1]==obs+1)
                        next_scores[b,s]=(old if censored else
                                          int(np.sign(old))*(3*abs(old)//4)+4*(rewards[s]-rewards[s-1]))
                sel=np.full_like(expected,-101)
                out=np.full_like(prior,-108)
                kernels._select_spine_tree[(batch,)](view(original.reshape(-1)),view(chain),view(prior),view(sel),
                                                      batch*width,width,block,num_warps=1)
                kernels._update_spine_margins[(batch,)](view(original[0]),view(original[2]),view(chain),
                                                         view(actual),view(lengths),view(prior),view(out),
                                                         width,block,num_warps=1)
                np.testing.assert_array_equal(sel,expected)
                np.testing.assert_array_equal(out,next_scores)
                runs+=batch
    print(json.dumps({"actual_triton_interpreter":triton.__version__,"widths":list(range(2,9)),
                      "source_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      "tree_selection_rows":runs,"feedback_rows":runs,"hardware_execution":False},indent=2))


if __name__=="__main__":
    main()
