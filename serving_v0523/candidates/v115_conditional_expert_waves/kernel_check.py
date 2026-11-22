"""Verify the production wave metadata kernel in the Triton interpreter and Hopper assembler."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs"))
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf


def interpret(kernel):
    totals = dict(layouts=0, iterations=0, checked_offsets=0, covered_tokens=0, live_routes=0)
    rng = np.random.default_rng(115)
    for tokens, k, e, cap in ((1,1,1,1),(64,8,7,33),(8192,8,32,8192),(16384,8,32,16377)):
        waves = (tokens*k+cap-1)//cap
        rows = min(tokens*k,cap+k-1)
        for kind in ("random","remote","local","exact","tail","first","last"):
            ids = rng.integers(-7*e,e,(tokens,k),dtype=np.int32)
            if kind == "remote":
                ids.fill(-1)
            elif kind == "local":
                ids %= e
            elif kind == "exact":
                ids.fill(-1)
                ids.flat[:cap] = e-1
            elif kind == "tail":
                ids[tokens//2:] = -1
            elif kind == "first":
                ids[1:] = -1
            elif kind == "last":
                ids[:-1] = -1
            counts = (ids>=0).sum(1)
            starts = np.cumsum(counts)-counts
            assignment = np.minimum(starts//cap, waves-1)
            bins = (assignment[:,None]*e+ids)[ids>=0]
            edges = np.r_[0,np.bincount(bins,minlength=waves*e).cumsum()].astype("i4")
            token_bounds = np.r_[np.searchsorted(starts,np.arange(waves)*cap),tokens].astype("i4")
            state,bounds,offsets,interval = (np.full(n,-99,"i4") for n in (1,e+1,e+1,2))
            live = np.array([False])
            args = [buf(a,tl.int32) for a in (edges,token_bounds,state,bounds,offsets,interval)]
            args.append(buf(live,tl.int1))
            next_token = matched_routes = 0
            for wave in range(waves+1):
                kernel[(1,)](*args,TOKENS=tokens,ROWS=rows,EXPERTS=e,WAVES=waves,
                    BLOCK=triton.next_power_of_2(e+1),FIRST=wave==0,num_warps=1)
                i = min(wave,waves-1)
                expected = edges[i*e:(i+1)*e+1]
                assert state[0] == wave
                assert np.array_equal(bounds,expected) and np.array_equal(offsets,expected+(rows-expected[-1]))
                assert np.array_equal(interval,token_bounds[i:i+2])
                active = wave < waves and next_token < tokens
                assert live[0] == active
                if not active:
                    break
                assert interval[0] == next_token and interval[1] >= interval[0]
                next_token = interval[1]
                matched_routes += int(bounds[-1]-bounds[0])
                totals["iterations"] += 1
                totals["checked_offsets"] += e+1
            assert next_token == tokens and matched_routes == (ids>=0).sum()
            totals["layouts"] += 1
            totals["covered_tokens"] += tokens
            totals["live_routes"] += matched_routes
    return totals


def assemble(kernel,work):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.backends.nvidia.compiler import get_ptxas
    result = []
    for tokens,rows,experts,waves in ((64,40,3,8),(8192,8199,32,8),(16384,16384,32,9),(16384,16384,64,9)):
        for first in (False,True):
            const = dict(TOKENS=tokens,ROWS=rows,EXPERTS=experts,WAVES=waves,
                         BLOCK=triton.next_power_of_2(experts+1),FIRST=first)
            sig = dict(zip(kernel.arg_names[:7],["*i32"]*6+["*i1"],strict=True))
            sig.update({name:"constexpr" for name in const})
            binary = triton.compile(ASTSource(kernel,sig,constexprs=const),
                target=GPUTarget("cuda",90,32),options={"num_warps":1})
            ptx=work/"kernel.ptx"
            ptx.write_text(binary.asm["ptx"])
            proc=subprocess.run([get_ptxas(90).path,"-arch=sm_90a","-v",str(ptx),"-o",str(work/"k.cubin")],
                                capture_output=True,text=True,check=True)
            assert "0 bytes spill stores, 0 bytes spill loads" in proc.stderr
            match=re.search(r"Used (\d+) registers",proc.stderr)
            result.append(dict(constants=const,sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest(),
                               registers=int(match.group(1)),spill_bytes=0))
    return {"specializations":result}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("--interpreter",action="store_true")
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        kernel=load_jit_module(args.source,Path(tmp))._load_wave
        result=interpret(kernel) if args.interpreter else assemble(kernel,Path(tmp))
    result.update(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),triton=triton.__version__,cuda_execution=False)
    print(json.dumps(result,indent=2))
