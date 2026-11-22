"""Run both versions of the real split and merge JIT on host buffers and compare to dense FP64."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import ml_dtypes
import numpy as np
import triton
import triton.language as tl
from triton.runtime.interpreter import InterpreterBuilder
from triton.tools.tensor_descriptor import TensorDescriptor

from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf


def check_case(modules, rng, loads, b, rows, window, parts, page_size, descriptor, step, groups):
    heads, width, dim = 2, 8, 64
    m, tile = triton.next_power_of_2(width * groups), 64 if window >= 0 else 128
    pages = triton.cdiv(rows, page_size)
    row_stride = pages + 5
    per_req = pages + 2
    capacity = (b * per_req + 1) * page_size
    lengths = np.array([max(width, rows - ((r*73 + step*7) % (rows-width+1))) for r in range(b)], 'i4')
    qlen = np.array([max(1, width-(r+step)%4) for r in range(b)], 'i4')
    if b > 1 and step:
        qlen[0] = lengths[0] = 0
    cu = np.r_[0, np.cumsum(qlen)].astype('i4')
    pitch = heads*dim
    table = np.full((b,row_stride),-1234,'i4')
    columns = triton.cdiv(row_stride * page_size, tile)
    origins = np.full((b,columns),-3,'i4')
    k, v = [rng.integers(-2,3,(capacity,pitch)).astype(ml_dtypes.float8_e4m3fn) for _ in range(2)]
    k[:page_size] = v[:page_size] = np.nan
    physical = []
    for r,n in enumerate(lengths):
        pmap = 1 + ((r+step)%b)*per_req + np.arange(pages,dtype='i4')
        certificate_base = max(0,int(n)-width-window)//tile if window>=0 else 0
        first = max(0,int(n)-int(qlen[r])-window) if window>=0 else 0
        pmap[:first//page_size] = 0
        for t in range(triton.cdiv(first,tile), n//tile):
            if (t + r + step)%3 == 1 and page_size == 1:
                i = t*tile
                pmap[i+5],pmap[i+17] = pmap[i+17],pmap[i+5]
        table[r,:pages] = pmap
        ids = pmap[np.arange(rows)//page_size]*page_size + np.arange(rows)%page_size
        physical.append(ids)
        for t in range(certificate_base,triton.cdiv(int(n),tile)):
            lo = t*tile
            a = ids[lo:min(lo+tile,int(n))]
            if lo>=first and len(a)==tile and a[0]>0 and np.array_equal(a,a[0]+np.arange(tile)):
                origins[r,t-certificate_base] = a[0]
    q = rng.integers(-2,3,(b*width,heads*groups,dim)).astype(ml_dtypes.float8_e4m3fn)
    ks = np.linspace(.72,1.13,b*heads,dtype='f4').reshape(b,heads)
    vs = np.linspace(.93,.69,b*heads,dtype='f4').reshape(b,heads)
    kb,vb = [buf(x.view('u1'),tl.float8e4nv) for x in (k,v)]
    direct = {}
    if descriptor:
        assert page_size==1 and window<0
        direct = dict(Origins=buf(origins,tl.int32), ORIGIN_STRIDE=origins.shape[1],CAPACITY=capacity,
                      KDesc=TensorDescriptor(kb,[capacity,pitch],[pitch,1],[tile,dim]),
                      VDesc=TensorDescriptor(vb,[capacity,pitch],[pitch,1],[tile,dim]))
    cap = triton.cdiv(row_stride*page_size,tile)
    if window>=0:
        cap = min(cap,triton.cdiv(window+width+tile-1,tile))
    bound = max(1,b*cap)
    results = []
    reads = []
    for name,mod in modules.items():
        split = min(bound,parts) if name=='candidate' else parts
        num_sms = 132 if step == 0 else 17
        programs = max(1, b * heads * width * groups // num_sms)
        merge_rows = min(4, 1 << (programs.bit_length() - 1)) if name == "candidate" else 1
        merge = dict(ROWS=merge_rows) if name == "candidate" else {}
        acc = np.full((split+b-1,heads,m,dim),-77,'f4')
        lse = np.full(acc.shape[:-1],-91,'f4')
        out = np.full(q.shape,0xCAF1,'u2')
        loads.clear()
        mod._partition_attention[(split,heads)](
            buf(q.view('u1'),tl.float8e4nv),kb,vb,buf(table,tl.int32),buf(lengths,tl.int32),buf(cu,tl.int32),
            buf(ks,tl.float32),buf(acc,tl.float32),buf(lse,tl.float32),row_stride,page_size,
            pitch*page_size,pitch,dim,pitch*page_size,pitch,dim,heads,1,b,heads,groups,dim,
            math.log2(math.e)/8,window,width,m,tile,split,triton.next_power_of_2(b),**direct)
        mod._merge_partitions[(b,heads,triton.cdiv(width*groups,merge_rows))](
            buf(acc,tl.float32),buf(lse,tl.float32),buf(lengths,tl.int32),buf(cu,tl.int32),buf(vs,tl.float32),
            buf(out,tl.bfloat16),heads,1,b,heads,groups,dim,window,m,tile,split,triton.next_power_of_2(b),16,**merge)
        results.append((acc,lse,out))
        reads.append(Counter(loads))
    assert reads[0]==reads[1]
    for parent,child in zip(*results):
        np.testing.assert_array_equal(parent[:len(child)].view('u2' if child.dtype==np.uint16 else 'u4'),
                                      child.view('u2' if child.dtype==np.uint16 else 'u4'))
        if len(parent)>len(child):
            assert (parent[len(child):] == (-77 if parent.ndim==4 else -91)).all()
    assert (out[int(cu[-1]):]==0xCAF1).all()
    worst, dense_words = 0.,0
    for r,count in enumerate(qlen):
        for j in range(count):
            position=int(lengths[r]-count+j)
            ids=physical[r][max(0,position-window) if window>=0 else 0:position+1]
            for h in range(heads):
                hslice=slice(h*dim,(h+1)*dim)
                gslice=slice(h*groups,(h+1)*groups)
                qq=q[cu[r]+j,gslice].astype('f8')
                logits=(qq@k[ids,hslice].astype('f8').T)*(float(ks[r,h])/8)
                prob=np.exp(logits-logits.max(axis=1,keepdims=True))
                ref=prob@v[ids,hslice].astype('f8') / prob.sum(axis=1,keepdims=True) * vs[r,h]
                actual=out[cu[r]+j,gslice].view(ml_dtypes.bfloat16).astype('f8')
                np.testing.assert_allclose(actual,ref,atol=.025,rtol=.04)
                worst=max(worst,float(np.max(np.abs(ref-actual))))
                dense_words+=actual.size
    return dict(batch=b,rows=rows,window=window,page_size=page_size,direct=descriptor,step=step,
                prior_partitions=parts,partitions=split,merge_rows=merge_rows,gqa=groups,max_q=width,
                fp32_words=acc.size+lse.size,bf16_words=out.size,dense_words=dense_words,max_abs_fp64=worst,
                tensor_map_loads=sum(reads[0].values()),empty_requests=int((qlen==0).sum()))


def main(root):
    rng=np.random.default_rng(311)
    original_dot=InterpreterBuilder.create_dot
    original_load=InterpreterBuilder.create_descriptor_load
    loads=[]
    def bf16_dot(self,a,b,acc,*flags):
        if a.dtype==b.dtype==tl.bfloat16:
            left=a.data.view(ml_dtypes.bfloat16).astype('f4')
            right=b.data.view(ml_dtypes.bfloat16).astype('f4')
            return type(acc)(left@right+acc.data,acc.dtype.scalar)
        return original_dot(self,a,b,acc,*flags)
    def logged(self,desc,idx,*flags):
        loads.append(tuple(int(np.asarray(i.data).item()) for i in idx))
        return original_load(self,desc,idx,*flags)
    cases=[(1,4281,2047,66,1),(1,769,-1,132,1),(1,2147,-1,4,1),
           (2,397,127,66,1),(4,2581,2047,66,1),(3,257,127,2,1),
           (2,5001,-1,1,1),(3,477,127,66,16)]
    result=[]
    with tempfile.TemporaryDirectory(dir=root) as temp:
        modules={n:load_jit_module(root/(n+'.py'),Path(temp)) for n in ('parent','candidate')}
        InterpreterBuilder.create_dot=bf16_dot
        InterpreterBuilder.create_descriptor_load=logged
        try:
            for b,rows,w,p,psz in cases:
                for step in range(2):
                    for direct in ((False,True) if psz==1 and w<0 else (False,)):
                        result.append(check_case(modules,rng,loads,b,rows,w,p,psz,direct,step,3 if step else 8))
        finally:
            InterpreterBuilder.create_dot=original_dot
            InterpreterBuilder.create_descriptor_load=original_load
    assert sum(v['tensor_map_loads'] for v in result)>0
    return dict(interpreter=triton.__version__,physical_h200=False,cases=result,
                source_sha256={n:hashlib.sha256((root/(n+'.py')).read_bytes()).hexdigest() for n in modules},
                total={k:sum(r[k] for r in result) for k in ('fp32_words','bf16_words','dense_words','tensor_map_loads','empty_requests')},
                max_abs_fp64=max(r['max_abs_fp64'] for r in result))


if __name__=='__main__':
    print(json.dumps(main(Path(sys.argv[1]).resolve()),indent=2))
