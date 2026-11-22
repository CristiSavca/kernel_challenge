"""Assemble or interpret the actual two-factor router kernel on Linux with Triton 3.8."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import ml_dtypes
import numpy as np
import triton
import triton.language as tl
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf


def assemble(kernel, scratch):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    records = []
    for n in (64,256):
        for bn,bk,stages in ((32,64,3),(32,64,2),(32,32,3),(64,64,3)):
            constants=dict(K=8192,N=n,BM=64,BN=bn,BK=bk)
            sig=dict(X="*bf16",High="*bf16",Low="*bf16",Out="*fp32",rows="i32")
            aligned={(kernel.arg_names.index(name),):[["tt.divisibility",16]] for name in ("X","High","Low","Out")}
            sig.update({k:"constexpr" for k in constants})
            result=triton.compile(ASTSource(kernel,sig,constexprs=constants,attrs=aligned),target=GPUTarget("cuda",90,32),
                                  options=dict(num_warps=4,num_stages=stages))
            ptx=result.asm["ptx"]
            (scratch/"probe.ptx").write_text(ptx)
            ptxas=Path(triton.__file__).parent/"backends/nvidia/bin/ptxas"
            report=subprocess.run([ptxas,"-arch=sm_90a","-v",scratch/"probe.ptx","-o",scratch/"probe.cubin"],
                                  check=True,text=True,capture_output=True).stderr
            ops=Counter(re.findall(r"(?:wgmma\.mma_async|mma\.sync)[\w.]+",ptx))
            assert ops and all(".f32.bf16.bf16" in k for k in ops),ops
            assert re.search(r"add(?:\.rn)?\.f32",ptx)
            records.append(dict(constants=constants,num_stages=stages,
                cubin_sha256=hashlib.sha256(result.asm["cubin"]).hexdigest(),smem=result.metadata.shared,
                matrix=ops,registers=int(re.search(r"Used (\d+) registers",report)[1]),
                spills=list(map(int,re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",report).groups()))))
    return dict(offline_sm90a=True,cases=records)


def interpret(kernel):
    from triton.runtime.interpreter import InterpreterBuilder
    native=InterpreterBuilder.create_dot
    # The Linux interpreter stores BF16 as uint16. Both inputs to its dot must
    # be decoded as BF16, never multiplied as unsigned storage bits.
    def dot(self,a,b,acc,*flags):
        assert a.dtype==b.dtype==tl.bfloat16
        left=a.data.view(ml_dtypes.bfloat16).astype("f4")
        right=b.data.view(ml_dtypes.bfloat16).astype("f4")
        return type(acc)(left@right+acc.data,acc.dtype.scalar)
    ret=[]
    rng=np.random.default_rng(252)
    def bf(a):return np.asarray(a,dtype=ml_dtypes.bfloat16)
    def original(x,h,l,bk):
        outs=[np.zeros((len(x),h.shape[1]),"f4") for _ in range(2)]
        for k in range(0,x.shape[1],bk):
            a=x[:,k:k+bk].astype("f4")
            for acc,w in zip(outs,(h,l)):
                acc+=a@w[k:k+bk].astype("f4")
        return outs[0]+outs[1]
    InterpreterBuilder.create_dot=dot
    try:
        for rows,k,n,mode in ((1,64,32,"integer"),(63,256,64,"random"),(64,512,256,"tie"),
                              (65,192,96,"random"),(131,1024,64,"integer"),(8,8192,256,"tie"),
                              (2048,8192,256,"sparse-exact")):
            if mode=="sparse-exact":
                # Keep both full production matrices and all K iterations. Exact integer rows
                # make an independent real-geometry FP32 oracle affordable on a small CPU.
                x=np.zeros((rows,k),"f4");w=np.zeros((k,n),"f4")
                for j in (0,15,63,127,k-1):
                    x[:,j]=rng.integers(-4,5,rows)
                    w[j]=rng.integers(-4,5,n)+rng.integers(-4,5,n)/4096
            else:
                x=rng.integers(-3,4,(rows,k)) if mode=="integer" else rng.normal(size=(rows,k))
                w=rng.integers(-32,33,(k,n))+rng.integers(-4,5,(k,n))/1024 if mode=="integer" else rng.normal(size=(k,n))
            if mode=="tie":
                x[:]=0; x[:,0]=1;w[:]=0
                w[0,:7]=2;w[0,7]=1-2**-11;w[0,8]=1+2**-11
            x=bf(x);w=w.astype("f4")
            high=bf(w);low=bf(w-high.astype("f4"))
            full=np.full((rows+2,n),-714.25,"f4")
            kernel[(triton.cdiv(rows,64),n//32)](
                *[buf(t.view("u2"),tl.bfloat16) for t in (x,high,low)],buf(full[1:-1],tl.float32),
                rows,k,n,64,32,64)
            out=full[1:-1]
            assert np.all(full[[0,-1]]==-714.25)
            expected=original(x,high,low,64)
            # BLAS can choose a different FP32 accumulation on different matrix row/column
            # sizes. The integer and sparse witnesses have exactly representable sums.
            if mode in ("integer","tie","sparse-exact"):np.testing.assert_array_equal(out,expected)
            else:np.testing.assert_allclose(out,expected,rtol=2e-5,atol=8e-5)
            top=lambda a:np.argsort(-a,axis=1,kind="stable")[:,:8]
            np.testing.assert_array_equal(top(out),top(expected))
            if mode=="tie":
                assert np.all(top(out)[:,-1]==8) and np.all(top(x.astype("f4")@high.astype("f4"))[:,-1]==7)
            ret.append(dict(rows=rows,cols=n,width=k,fixture=mode,fp32_values=int(out.size),
                         changed_bits=int(np.sum(out.view("u4")!=expected.view("u4"))),
                         max_abs_error=float(np.max(np.abs(out-expected))),top8_ids=int(rows*8)))
    finally:InterpreterBuilder.create_dot=native
    return dict(cases=ret,checked_fp32=sum(r["fp32_values"] for r in ret),checked_top8=sum(r["top8_ids"] for r in ret),
                scope="The CPU interpreter executes all production loop iterations and BF16 input conversions; CPU BLAS implements FP32 dot tiles. Native CUDA/cuBLAS agreement and serving throughput need the judge.")


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("compile","interpret"));p.add_argument("source",type=Path);args=p.parse_args()
    with tempfile.TemporaryDirectory() as t:
        tmp=Path(t);kernel=load_jit_module(args.source,tmp)._paired_router
        result=assemble(kernel,tmp) if args.mode=="compile" else interpret(kernel)
    print(json.dumps(dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                          triton=triton.__version__,physical_H200=False,**result),indent=2))


if __name__=="__main__":main()
