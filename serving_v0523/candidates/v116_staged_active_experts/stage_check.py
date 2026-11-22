"""Compile or interpret the production normalized route stager and its optional expert list."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs"))
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf


def shape_constants(tokens, k, hidden, experts, mapped, write):
    slots = tokens * k
    return dict(SLOTS=slots, GLOBAL=experts * (8 if mapped else 1), EXPERTS=experts,
                HAS_MAP=mapped, TOPK=k, K_TILE=triton.next_power_of_2(k), HIDDEN=hidden,
                SPAN=triton.next_power_of_2(slots), BUCKETS=triton.next_power_of_2(experts + 1),
                WIDTH=triton.next_power_of_2(hidden), WRITE_ACTIVE=write)


def assemble(kernel, work):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.compiler import ASTSource
    cases = []
    for m,k,h,e,mapped,idx,gamma in [
        (8,8,4096,32,True,"*i32","*fp32"), (16,8,4096,32,True,"*i32","*fp32"),
        (32,8,4096,32,True,"*i32","*fp32"), (32,8,4096,32,True,"*i64","*bf16"),
        (23,3,4096,17,False,"*i64","*bf16"), (31,8,8192,64,True,"*i64","*fp32"),
    ]:
        for write in (True, False):
            constants = shape_constants(m,k,h,e,mapped,write)
            sig = dict(zip(kernel.arg_names[:10], [idx,"*i32","*bf16",gamma,*["*i32"]*4,"*bf16","fp32"], strict=True))
            sig.update({name:"constexpr" for name in constants})
            binary = triton.compile(ASTSource(kernel,sig,constexprs=constants),
                                   target=GPUTarget("cuda",90,32),options={"num_warps":8})
            ptx=work/"kernel.ptx"
            ptx.write_text(binary.asm["ptx"])
            run=subprocess.run([get_ptxas(90).path,"-arch=sm_90a","-v",str(ptx),"-o",str(work/"k.cubin")],
                               capture_output=True,text=True,check=True)
            assert "0 bytes spill stores, 0 bytes spill loads" in run.stderr
            registers=int(re.search(r"Used (\d+) registers",run.stderr).group(1))
            cases.append(dict(tokens=m,topk=k,hidden=h,experts=e,mapped=mapped,ids=idx,
                              gamma=gamma,emit=write,registers=registers,spill_bytes=0,
                              cubin_sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest()))
    return {"kernels":cases}


def bf16_values(size,rng):
    values=rng.integers(-8,9,size=size).astype("f4")
    return (values.view("u4")>>16).astype("u2")


def one_case(kernel,shape,pattern,rng):
    tokens,k,hidden,e,mapped=shape
    c=shape_constants(*shape,True)
    g=c["GLOBAL"]
    ownership=np.full(g,-1,"i4")
    owners=rng.choice(g,e,replace=False) if mapped else np.arange(e)
    ownership[owners]=np.arange(e,dtype="i4")
    ids=rng.integers(-2,g+2,size=tokens*k,dtype="i4")
    if pattern=="remote": ids.fill(-5)
    elif pattern=="one": ids.fill(owners[-1])
    elif pattern=="all": ids[:]=np.resize(owners,len(ids))
    elif pattern=="skew":
        ids[::2]=owners[-1]
        ids[1::4]=owners[0]
    x=bf16_values((tokens,hidden),rng)
    gamma=rng.integers(-3,4,size=hidden).astype("f4")
    valid=(ids>=0)&(ids<g)
    local=np.full(ids.size,-1,"i4")
    local[valid]=ownership[ids[valid]]
    live=np.flatnonzero(local>=0)
    ordered=live[np.argsort(local[live],kind="stable")]
    rem=ids.size-live.size
    counts=np.bincount(local[live],minlength=e)
    expected_offsets=rem+np.r_[0,counts.cumsum()]
    expected_plan=np.flatnonzero(counts)
    expected_rows=np.full(ids.size,-1,"i4")
    expected_rows[ordered]=np.arange(rem,ids.size)
    def run(emit):
        ids_out,keys,rows=[np.full(ids.size,-777,"i4") for _ in range(3)]
        layout=np.full(2*(e+1)+2,-777,"i4")
        acts=np.full((ids.size,hidden),0x7fc0,"u2")
        args=[buf(ids,tl.int32),buf(ownership,tl.int32),buf(x,tl.bfloat16),buf(gamma,tl.float32),
              buf(ids_out,tl.int32),buf(layout,tl.int32),buf(keys,tl.int32),buf(rows,tl.int32),
              buf(acts,tl.bfloat16)]
        kernel[(tokens,)](*args,1.e-6,**(c|dict(WRITE_ACTIVE=emit)),num_warps=8)
        np.testing.assert_array_equal(ids_out,local)
        np.testing.assert_array_equal(rows,expected_rows)
        np.testing.assert_array_equal(layout[:e+1],expected_offsets)
        np.testing.assert_array_equal(keys[rem:],(local[ordered]+1)*c["SPAN"]+ordered)
        np.testing.assert_array_equal(keys[:rem],-777)
        np.testing.assert_array_equal(acts[:rem],0x7fc0)
        shift=e+1
        if emit:
            assert layout[shift]==len(expected_plan)
            np.testing.assert_array_equal(layout[shift+1:shift+1+len(expected_plan)],expected_plan)
            shift+=1+len(expected_plan)
        np.testing.assert_array_equal(layout[shift:],-777)
        return acts
    np.testing.assert_array_equal(run(True),run(False))
    return len(live)*hidden


def interpret(kernel):
    rng=np.random.default_rng(116)
    shapes=[(1,1,17,1,False),(5,3,43,7,True),(12,8,64,17,False),
            (8,8,4096,32,True),(16,8,4096,32,True),(32,8,4096,32,True),
            (17,8,256,33,True),(12,4,239,64,True)]
    patterns=("random","remote","one","all","skew")
    checked=sum(one_case(kernel,s,p,rng) for s in shapes for p in patterns)
    return {"paired_fixtures":len(shapes)*len(patterns),"interpreter_launches":2*len(shapes)*len(patterns),
            "live_bf16_outputs_compared":checked,"patterns":patterns}


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("source",type=Path);ap.add_argument("--interpreter",action="store_true")
    args=ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        kernel=load_jit_module(args.source,Path(tmp))._stage_token_routes
        result=interpret(kernel) if args.interpreter else assemble(kernel,Path(tmp))
    result.update(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),triton=triton.__version__)
    print(json.dumps(result,indent=2))
