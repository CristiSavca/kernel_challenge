"""Compile final producer and both partition paths with the pinned Triton 3.8 tools."""
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


def main():
    src, tools = map(Path,sys.argv[1:3])
    sys.path.insert(0,str(tools.resolve()))
    from compile_decode_sm90 import load_jit_module
    arch=GPUTarget("cuda",90,32)
    records=[]
    ptxas=Path(triton.__file__).parent/"backends/nvidia/bin/ptxas"
    def one(func, sig, cs, opts, label, path, work, aligned):
        sig.update({k:"constexpr" for k in cs})
        attrs={(func.arg_names.index(k),):[["tt.divisibility",16]] for k in aligned}
        binary=triton.compile(ASTSource(func,sig,constexprs=cs,attrs=attrs),target=arch,options=opts)
        ptx=binary.asm['ptx'];pt=work/'test.ptx';pt.write_text(ptx)
        stats=subprocess.run([ptxas,"-v","-arch=sm_90a",pt,"-o",work/"test.cubin"],
                             capture_output=True,text=True,check=True).stderr
        records.append(dict(label=label,kernel=func.__name__,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            shared_bytes=binary.metadata.shared,cubin_sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest(),
            cubin_bytes=len(binary.asm['cubin']),tma_ops=ptx.count("cp.async.bulk.tensor"),
            mma_ops=ptx.count("mma.sync.aligned"),wgmma_ops=ptx.count("wgmma.mma_async"),
            spill_bytes=list(map(int,re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',stats).groups())),
            registers=int(re.search(r'Used (\d+) registers',stats).group(1))))
    with tempfile.TemporaryDirectory() as tmp:
        w=Path(tmp)
        for source,mode in [('incumbent.py','V190'),('paged_verify_attention.py','indexed'),('paged_verify_attention.py','direct')]:
            path=src/source;fn=load_jit_module(path,w)._partition_attention
            for b in (1,4,16,32):
                for scale in (True,False):
                    cs=dict(PAGE_ROW_STRIDE=131088,PAGE_SIZE=1,K_PAGE=128,K_TOKEN=128,K_HEAD=64,V_PAGE=128,
                        V_TOKEN=128,V_HEAD=64,KS_ROW=0,KS_HEAD=0,B=b,H_KV=2,GQA=8,D=64,
                        SCALE_LOG2=math.log2(math.e)/8,WINDOW_LEFT=-1,MAX_Q=8,TILE_M=64,TILE_N=128,
                        PARTITIONS=132,BLOCK_B=b)
                    sig=dict(zip(fn.arg_names[:9],['*fp8e4nv']*3+['*i32']*3+['*fp32']*3))
                    if not scale:del sig['KScale'];cs['KScale']=None
                    if mode=='direct':
                        sig.update(Origins='*i32',KDesc='tensordesc<fp8e4nv[128,64]>',VDesc='tensordesc<fp8e4nv[128,64]>')
                        cs.update(ORIGIN_STRIDE=1025,CAPACITY=3420200)
                    elif mode=='indexed':cs.update(Origins=None,KDesc=None,VDesc=None,ORIGIN_STRIDE=0,CAPACITY=0)
                    one(fn,sig,cs,dict(num_warps=4,num_stages=3),f"{mode}/B{b}/scale{int(scale)}",path,w,
                        [k for k,v in sig.items() if v.startswith('*')])
                    assert bool(records[-1]["tma_ops"]) == (mode=="direct")
        path=src/'metadata.py'; fn=load_jit_module(path,w)._fused_decode_metadata_kernel
        for swa in (False,True):
            for it in ("i32","i64"):
                for seq in ("i32","i64"):
                    block=128 if swa else 256
                    cs=dict(PAGE_SIZE=1,SEQ_LEN_DELTA=8,BLOCK_COLS=block,SKIP_PAGE_TABLE=False,ZERO_TAIL=False,
                            WINDOW_LEFT=-1,ORIGIN_STRIDE=1025,full_v2p=None,swa_v2p=None)
                    ptr=dict(seq_lens='*'+seq,req_to_token='*i32',req_pool_indices='*'+it,cache_seqlens_int32='*i32',
                             cu_seqlens_k='*i32',page_table='*i32',origins='*i32')
                    if swa:ptr.update(swa_page_table='*i32',full_to_swa_index_mapping='*i64')
                    else:cs.update(swa_page_table=None,full_to_swa_index_mapping=None)
                    sig={a:ptr.get(a,'i32') for a in fn.arg_names if a not in cs}
                    one(fn,sig,cs,dict(num_warps=block//32,num_stages=3),f'cert/swa{int(swa)}/{it}/{seq}',path,w,ptr)
    print(json.dumps(dict(triton=triton.__version__,target="sm90",physical_h200=False,tests=records),indent=2))
if __name__=='__main__':
    main()
