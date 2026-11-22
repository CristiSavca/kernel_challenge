"""Run exact staged-route kernels in the CPU interpreter or assemble the SM90 binaries."""
import argparse
import collections
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


def constants(m, k, h, e, mapped, tile):
    scope=dict(SLOTS=m*k,GLOBAL=e*(8 if mapped else 1),EXPERTS=e,HAS_MAP=mapped,
               BUCKETS=triton.next_power_of_2(e+1))
    return scope|dict(TILE=tile), scope|dict(TOPK=k,K_TILE=triton.next_power_of_2(k),
        HIDDEN=h,SPAN=triton.next_power_of_2(m*k),WIDTH=triton.next_power_of_2(h),
        WRITE_ACTIVE=False,ROUTE_TILE=tile)


def assemble(module, work):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.compiler import ASTSource
    cases=[]
    for m,k,h,e,mapped,ids,gamma in [
        (128,8,4096,32,True,"*i32","*fp32"),(256,8,4096,32,True,"*i32","*fp32"),
        (171,7,4096,17,True,"*i64","*bf16"),(145,8,8192,64,False,"*i64","*fp32"),
    ]:
        p,c=constants(m,k,h,e,mapped,128)
        pointer_stager=[ids,"*i32","*bf16",gamma,*["*i32"]*4,"*bf16","fp32"]
        jobs=[
            ("planner",module._plan_route_tiles,[ids,*["*i32"]*4],p,4),
            ("tiled",module._stage_token_routes,pointer_stager,c,8),
            ("original",module._stage_token_routes,pointer_stager,c|dict(ROUTE_TILE=0),8),
        ]
        for name,fn,ptrs,config,warps in jobs:
            sig=dict(zip(fn.arg_names[:len(ptrs)],ptrs,strict=True))|{a:"constexpr" for a in config}
            binary=triton.compile(ASTSource(fn,sig,constexprs=config),target=GPUTarget("cuda",90,32),options={"num_warps":warps})
            ptx=work/"kernel.ptx";ptx.write_text(binary.asm["ptx"])
            proc=subprocess.run([get_ptxas(90).path,"-arch=sm_90a","-v",str(ptx),"-o",str(work/"kernel.cubin")],
                                capture_output=True,text=True,check=True)
            size=re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",proc.stderr)
            regs=int(re.search(r"Used (\d+) registers",proc.stderr).group(1))
            cubin=work/"triton.cubin"
            cubin.write_bytes(binary.asm["cubin"])
            disassembler=Path(get_ptxas(90).path).with_name("cuobjdump")
            sass=subprocess.run([str(disassembler),"--dump-sass",str(cubin)],check=True,
                                capture_output=True,text=True).stdout
            ops=collections.Counter(re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P(?:\d+|T)\s+)?([A-Z0-9_.]+)\b",sass))
            assert ops
            layouts=[]
            gir=binary.asm["ttgir"]
            for line in gir.splitlines():
                if "}) : (tensor<" in line and f"{h}xf32" in line and "-> f32" in line:
                    ty=re.search(rf"tensor<{h}xf32, (#[^>]+)>",line)
                    if ty:
                        layout=ty.group(1)
                        layouts.extend(x.split(" = ",1)[1] for x in gir.splitlines() if x.startswith(layout+" = "))
            assert layouts or name == "planner"
            cases.append(dict(kernel=name,m=m,k=k,h=h,e=e,mapped=mapped,regs=regs,
                              static_sass_sites=sum(ops.values()),
                              shared_atomic_sites=sum(v for op,v in ops.items() if "ATOMS" in op),
                              warp_vote_sites=sum(v for op,v in ops.items() if op.startswith("VOTE")),
                              rms_reduction_layouts=layouts,
                              shared_bytes=binary.metadata.shared,spill_stores=int(size.group(1)),spill_loads=int(size.group(2)),
                              cubin_sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest()))
    return {"kernels":cases}


def single(module, shape, pattern, rng):
    m,k,h,e,mapped=shape
    p,c=constants(*shape,128)
    slots=m*k;g=c["GLOBAL"];nt=triton.cdiv(slots,128)
    mapping=np.full(g,-1,"i4")
    owned=rng.choice(g,e,replace=False) if mapped else np.arange(e)
    mapping[owned]=np.arange(e,dtype="i4")
    ids=rng.integers(-2,g+2,size=slots,dtype="i8")
    if pattern=="all":
        ids[:]=np.resize(owned,slots)
    elif pattern=="skew":
        ids[::2]=owned[-1];ids[1::4]=owned[0];ids[5::16]=2**40
    elif pattern=="remote": ids.fill(2**40)
    elif pattern=="duplicate":
        ids[:]=np.repeat(rng.choice(owned,size=m),k)
    elif pattern=="tail":
        ids.fill(-1);ids[-min(129,slots):]=np.resize(owned,min(129,slots))
    valid=(ids>=0)&(ids<g);local=np.full(slots,-1,"i4");local[valid]=mapping[ids[valid]]
    live=np.flatnonzero(local>=0);order=live[np.argsort(local[live],kind="stable")];remote=slots-len(live)
    offsets=remote+np.r_[0,np.bincount(local[live],minlength=e).cumsum()]
    rows=np.full(slots,-1,"i4");rows[order]=np.arange(remote,slots)
    keys=(local[order]+1)*c["SPAN"]+order
    x=(rng.integers(-12,12,(m,h)).astype("f4").view("u4")>>16).astype("u2")
    gamma=rng.integers(-3,3,h).astype("f4"); gamma[::17]=np.float32(-0.)
    def run(tiled):
        outs=[np.full(n,-777,"i4") for n in (slots,e+1,slots,slots)]
        loc,off,rowkeys,rowmap=outs
        ends=np.full((nt,e+1),-777,"i4")
        acts=np.full((slots,h),0x7fc0,"u2")
        if tiled:
            module._plan_route_tiles[(nt,)](buf(ids,tl.int64),buf(mapping,tl.int32),
                buf(loc,tl.int32),buf(rowmap,tl.int32),buf(ends,tl.int32),**p,num_warps=4)
            for ti in range(nt):
                sl=local[ti*128:(ti+1)*128]
                expected=np.r_[sum(sl<0),np.bincount(sl[sl>=0],minlength=e)].cumsum()
                np.testing.assert_array_equal(ends[ti],expected)
        module._stage_token_routes[(m,)](buf(ids,tl.int64),buf(ends if tiled else mapping,tl.int32),
            buf(x,tl.bfloat16),buf(gamma,tl.float32),*(buf(arr,tl.int32) for arr in outs),
            buf(acts,tl.bfloat16),1.e-6,**(c|dict(ROUTE_TILE=128 if tiled else 0)),num_warps=8)
        np.testing.assert_array_equal(loc,local);np.testing.assert_array_equal(off,offsets)
        np.testing.assert_array_equal(rowmap,rows);np.testing.assert_array_equal(rowkeys[remote:],keys)
        np.testing.assert_array_equal(rowkeys[:remote],-777);np.testing.assert_array_equal(acts[:remote],0x7fc0)
        return acts
    np.testing.assert_array_equal(run(True),run(False))
    return int(len(live)*h),nt


def interpret(module):
    rng=np.random.default_rng(11717);n=values=tiles=0
    for shape in [(1,1,13,1,False),(19,7,65,13,True),(43,3,127,32,False),
                  (55,7,97,7,True),(128,8,4096,32,True),(256,8,4096,32,True),
                  (171,7,143,17,True),(145,8,157,64,False)]:
        for pattern in ("random","remote","all","skew","duplicate","tail"):
            count,t=single(module,shape,pattern,rng);n+=1;values+=count;tiles+=t
    return dict(paired_fixtures=n,program_launches=3*n,live_bf16_values=values,checked_prefix_tiles=tiles)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("source",type=Path);ap.add_argument("--interpreter",action="store_true");args=ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        mod=load_jit_module(args.source,Path(tmp));result=interpret(mod) if args.interpreter else assemble(mod,Path(tmp))
    result.update(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),triton=triton.__version__)
    print(json.dumps(result,indent=2))
