"""Compile the original and preplanned Hopper partition and merge variants from their source."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from compile_decode_sm90 import load_jit_module


def assemble(fn, sig, const, tmp, label):
    sig = dict(sig)
    attrs = {(fn.arg_names.index(k),): [["tt.divisibility", 16]] for k,t in sig.items() if t.startswith("*")}
    sig.update({k: "constexpr" for k in const})
    assert set(sig) == set(fn.arg_names), (label, set(fn.arg_names)-set(sig))
    binary = triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=GPUTarget("cuda",90,32),
                            options=dict(num_warps=4,num_stages=3))
    ptx = binary.asm["ptx"]
    (tmp/"probe.ptx").write_text(ptx)
    ptxas = Path(triton.__file__).parent/"backends/nvidia/bin/ptxas"
    report = subprocess.run([ptxas,"-arch=sm_90a","-v",tmp/"probe.ptx","-o",tmp/"probe.cubin"],
                            check=True,capture_output=True,text=True).stderr
    def ops(pattern):return dict(Counter(re.findall(pattern,ptx)))
    return dict(label=label,kernel=fn.__name__,cubin_sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest(),
                registers=int(re.search(r"Used (\d+) registers",report)[1]),
                spills=list(map(int,re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",report).groups())),
                shared_bytes=binary.metadata.shared,matrix=ops(r"(?:wgmma\.mma_async|mma\.sync)[\w.]+"),
                tensor_load=ops(r"cp\.async\.bulk\.tensor[\w.:]+"))


def main():
    old_path, new_path = map(Path, sys.argv[1:])
    cases=[]
    with tempfile.TemporaryDirectory() as scratch:
        tmp = Path(scratch)
        old, new = [load_jit_module(p,tmp) for p in (old_path,new_path)]
        for b in (1,4,16,32):
            for window in (-1,2047):
                n,p = (128,132) if window < 0 else (64,66)
                common=dict(B=b,H_KV=2,GQA=8,D=64,WINDOW_LEFT=window,TILE_M=64,TILE_N=n,PARTITIONS=p,BLOCK_B=b)
                rows=min(4,1<<(max(1,b*128//132).bit_length()-1))
                if b>1:
                    c=dict(B=b,PARTITIONS=p,WINDOW_LEFT=window,TILE_N=n,BLOCK_B=b,BLOCK_P=16)
                    cases.append(assemble(new._build_verify_raster,dict(Lengths="*i32",CuQ="*i32",Plan="*i32"),
                                          c,tmp,f"b{b}-w{window}/planner"))
                for direct in (False,True):
                    reference=None
                    for mode,mod in (("parent",old),("planned",new),("fallback",new)):
                        if b==1 and mode=="planned":continue
                        sig=dict(zip(mod._partition_attention.arg_names[:9],["*fp8e4nv"]*3+["*i32"]*3+["*fp32"]*3))
                        const=dict(common,PAGE_ROW_STRIDE=131072,PAGE_SIZE=1,K_PAGE=128,K_TOKEN=128,K_HEAD=64,
                                   V_PAGE=128,V_TOKEN=128,V_HEAD=64,KS_ROW=0,KS_HEAD=0,SCALE_LOG2=math.log2(math.e)/8,
                                   MAX_Q=8,ORIGIN_STRIDE=(1024 if window<0 else 34) if direct else 0,CAPACITY=3420200 if direct else 0)
                        if direct:sig.update(Origins="*i32",KDesc=f"tensordesc<fp8e4nv[{n},64]>",VDesc=f"tensordesc<fp8e4nv[{n},64]>")
                        else:const.update(Origins=None,KDesc=None,VDesc=None)
                        if mode=="planned":sig["Plan"]="*i32"
                        if mode=="fallback":const["Plan"]=None
                        t=assemble(mod._partition_attention,sig,const,tmp,f"b{b}-w{window}-tma{int(direct)}/{mode}")
                        assert t["matrix"] and bool(t["tensor_load"])==direct,t
                        if reference is not None:assert set(t["matrix"])==set(reference["matrix"]),t
                        else:reference=t
                        cases.append(t)
                for mode,mod in (("parent",old),("planned",new),("fallback",new)):
                    if b==1 and mode=="planned":continue
                    sig=dict(Partial="*fp32",LSE="*fp32",Lengths="*i32",CuQ="*i32",VScale="*fp32",Out="*bf16")
                    const=dict(common,VS_ROW=2,VS_HEAD=1,BLOCK_SPLITS=16,ROWS=rows)
                    if mode=="planned":sig["Plan"]="*i32"
                    if mode=="fallback":const["Plan"]=None
                    cases.append(assemble(mod._merge_partitions,sig,const,tmp,f"b{b}-w{window}/merge-{mode}"))
    print(json.dumps(dict(triton=triton.__version__,physical_H200=False,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path,new_path)},cases=cases),indent=2))


if __name__=="__main__":main()
