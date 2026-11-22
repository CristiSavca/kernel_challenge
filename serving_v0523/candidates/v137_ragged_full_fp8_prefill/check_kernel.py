"""Execute the exact submitted gather under Triton's CPU interpreter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"runs"))
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf

def test(kernel,width,klen,dtype,rng):
    batch=len(klen);n=121
    cu=np.array([0,*np.cumsum(klen)],dtype=np.int32)
    cap=int(cu[-1])+batch*256
    K=rng.integers(0,256,(n,width+13),dtype=np.uint8)
    V=rng.integers(0,256,(n,width+6),dtype=np.uint8)
    table=rng.integers(0,n,(batch,max(klen)+19),dtype=dtype)
    out=[np.full((cap+2,width),123,dtype=np.uint8) for _ in range(2)]
    targets=[a.copy() for a in out];new_cu=np.full(batch+3,-77,np.int32)
    expected_cu=cu+np.arange(batch+1,dtype=np.int32)*256
    for b,k in enumerate(klen):
        dst=1+expected_cu[b]
        for source,target in zip((K,V),targets):
            target[dst:dst+k]=source[table[b,:k],:width]
            target[dst+k:dst+k+256]=0
    block=min(32,4096//width)
    kernel[(min(16,triton.cdiv(max(klen)+256,block)),batch)](
        buf(K,tl.uint8),buf(V,tl.uint8),buf(table,tl.int32 if dtype==np.int32 else tl.int64),
        buf(cu,tl.int32),buf(new_cu[1:-1],tl.int32),buf(out[0][1:-1],tl.uint8),buf(out[1][1:-1],tl.uint8),
        K.shape[1],V.shape[1],table.shape[1],batch,ROW_BYTES=width,BLOCK_ROWS=block)
    for x,y in zip(out,targets): np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(new_cu,np.r_[-77,expected_cu,-77])
    return 2*cap*width

def main(path):
    rng=np.random.default_rng(277)
    cases=0;total=0
    with tempfile.TemporaryDirectory() as tmp:
        kernel=load_jit_module(path,Path(tmp))._pack_ragged_prefill_kv
        for width in (16,32,64,128,256,512,1024):
            for dtype in (np.int32,np.int64):
                for rows in ([0,31,65,0],[320,17],[1,2,3,4],[33,0,1501,101]):
                    total+=test(kernel,width,rows,dtype,rng);cases+=1
        total+=test(kernel,128,[74160,8192],np.int32,rng);cases+=1
    print(json.dumps(dict(cases=cases,destination_bytes=total,guard_rows=256,passed=True,
                          device="Triton CPU interpreter",triton=triton.__version__,
                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("source",type=Path)
    main(parser.parse_args().source)
