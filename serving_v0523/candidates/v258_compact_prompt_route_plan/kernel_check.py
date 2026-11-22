"""Compare the submitted stable tile planner to a stable full sort and compile SM90 code."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import triton


def load(source, scratch):
    methods = {"_count_route_tiles", "_scan_route_tiles", "_place_routes"}
    body = source.read_text().splitlines(keepends=True)
    nodes = [n for n in ast.parse(source.read_bytes()).body if getattr(n, "name", None) in methods]
    path = scratch / "testing_routes.py"
    path.write_text("import triton\nimport triton.language as tl\n" + "\n".join(
        "".join(body[n.decorator_list[0].lineno - 1:n.end_lineno]) for n in nodes))
    spec = importlib.util.spec_from_file_location("testing_routes", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return [getattr(module, n) for n in ("_count_route_tiles", "_scan_route_tiles", "_place_routes")]


def compile_sm90(kernels):
    import re
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    results = []
    assembler = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
    for experts in (1, 3, 17, 32, 64):
        bins = triton.next_power_of_2(experts + 1)
        specs = [(0,4,2,dict(GLOBAL=experts*8,MAP=mapping,EXPERTS=experts,BINS=bins,TILE=256))
                 for mapping in (False,True)]
        specs.append((2,6,2,dict(EXPERTS=experts,BINS=bins,TILE=256)))
        if experts == 32:
            specs.extend((1,2,1,dict(SPAN=s)) for s in (32,64,128,256,512,1024,2048))
        for idx,pointers,scalar,constants in specs:
            kernel=kernels[idx]
            sig=dict(zip(kernel.arg_names, ["*i32"]*pointers + ["i32"]*scalar + ["constexpr"]*len(constants)))
            binary=triton.compile(ASTSource(kernel, sig, constexprs=constants),target=GPUTarget("cuda",90,32))
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/"route.ptx";p.write_text(binary.asm["ptx"])
                log=subprocess.run([assembler, "-v", "-arch=sm_90a", p, "-o", p.with_suffix(".cubin")],
                                   check=True,text=True,capture_output=True).stderr
            spill=re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",log)
            results.append(dict(kernel=kernel.__name__,constants=constants,registers=int(re.search(r"Used (\d+) registers",log)[1]),
                                shared_bytes=binary.metadata.shared,spill_bytes=list(map(int,spill.groups())),
                                cubin_sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest()))
    return dict(architecture="sm90",cases=results)


def interpret(kernels):
    import triton.language as tl
    from interpret_decode_kernels import buf as buffer
    def buf(value):return buffer(value, tl.int32)
    import numpy as np

    count, scan, place = kernels
    rng = np.random.default_rng(258)
    comparisons=[]
    for slots,experts,mode in ((1,1,"empty"),(255,3,"hot"),(256,32,"single"),(257,64,"uniform"),
                               (4099,17,"repeat"),(8193,1,"empty"),(65541,32,"hot"),
                               (131072,32,"tp8"),(131071,32,"single"),(131072,32,"empty")):
        ids=rng.integers(-1,experts,slots,dtype=np.int32)
        if mode=="hot": ids[::3]=0
        if mode=="repeat":ids[:]=(np.arange(slots)//13%4-1).astype("i4")
        if mode=="empty":ids.fill(-1)
        if mode=="single":ids.fill(experts-1)
        # Exercise both the native contiguous EP map and a permuted expert layout.
        global_ids=ids.copy()
        mapping=np.arange(experts,dtype="i4")
        use_map=mode in ("uniform","repeat","tp8","hot")
        if use_map:
            global_ids=rng.integers(-1,experts*8,slots,dtype="i4")
            if mode=="repeat":global_ids[1::2]=global_ids[::2][:len(global_ids[1::2])]
            if mode=="hot":global_ids[:slots//2]=3
            mapping=np.full(experts*8,-1,dtype="i4")
            mapping[3::8]=np.arange(experts,dtype="i4")[::-1]
            if mode=="tp8":
                # Each token has exactly eight different global experts. This rank owns 96..127.
                global_ids=np.array([rng.choice(256,8,replace=False) for _ in range(slots//8)],dtype="i4").reshape(-1)
                mapping.fill(-1)
                mapping[96:128]=np.arange(32,dtype="i4")
            ids=np.where(global_ids<0,-1,mapping[global_ids.clip(min=0)]).astype("i4")
        clean=np.flatnonzero(ids>=0)
        stable=clean[np.argsort(ids[clean],kind="stable")]
        expected=np.full(slots,-1,dtype="i4")
        expected[stable]=np.arange(len(stable),dtype="i4")
        sizes=np.bincount(ids[clean],minlength=experts)
        tiles=triton.cdiv(slots,256)
        def arena(n):
            raw=np.full(n+2,-987654321,dtype="i4")
            return raw,raw[1:-1]
        p,prefix=arena((experts+1)*tiles);t,totals=arena(experts+1)
        z,sorted_ids=arena(len(stable));m,mapped=arena(slots);o,offsets=arena(experts+1)
        bins=triton.next_power_of_2(experts+1)
        l,local=arena(slots)
        if not use_map:local[:]=global_ids
        count[(tiles,)](buf(global_ids),buf(mapping),buf(local),buf(prefix),slots,tiles,len(mapping),
                        use_map,experts,bins,256)
        np.testing.assert_array_equal(local,ids)
        scan[(experts+1,)](buf(prefix),buf(totals),tiles,triton.next_power_of_2(tiles))
        np.testing.assert_array_equal(totals,np.r_[sizes,len(stable)])
        place[(tiles,)](buf(ids),buf(prefix),buf(totals),buf(sorted_ids),buf(mapped),buf(offsets),
                        slots,tiles,experts,bins,256)
        np.testing.assert_array_equal(mapped,expected)
        np.testing.assert_array_equal(sorted_ids,ids[stable])
        np.testing.assert_array_equal(offsets,np.r_[0,sizes.cumsum()])
        for arr in (p,t,z,m,o,l): assert np.all(arr[[0,-1]]==-987654321)
        comparisons.append(dict(slots=slots,experts=experts,distribution=mode,mapped=use_map,live=len(stable),
                                full_inverse_ids=slots,redzones=12))
    return dict(cases=comparisons,checked_inverse_ids=sum(r['slots'] for r in comparisons),
                checked_stable_expert_rows=sum(r['live'] for r in comparisons))


if __name__=="__main__":
    source=Path(sys.argv[2])
    with tempfile.TemporaryDirectory() as td:
        kernels=load(source,Path(td))
        receipt=compile_sm90(kernels) if sys.argv[1]=="aot" else interpret(kernels)
    print(json.dumps(dict(triton=triton.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          physical_execution=False,**receipt),indent=2))
