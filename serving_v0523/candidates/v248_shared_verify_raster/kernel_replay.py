"""Execute the V247 and V248 Triton partition/merge pair on identical physical K/V addresses."""
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


def check_case(old,new,case,rng,counts):
    batch, heads, groups, qwidth, max_len, window, parts, direct = case
    dim=64
    tile_n=64 if window >= 0 else 128
    tile_m=triton.next_power_of_2(groups*qwidth)
    block_b=triton.next_power_of_2(batch)
    rows=min(4,1<<(max(1,batch*heads*groups*qwidth//132).bit_length()-1))
    as_int=lambda x:buf(x,tl.int32)
    plan=np.full(2*batch+2+parts+2,-987654321,dtype="i4")
    packed_plan=as_int(plan[1:-1])
    lens=np.zeros(batch,"i4")
    cu=np.zeros(batch+1,"i4")
    cache_rows=batch*(max_len+256)+256
    pitches=[heads*dim+16,heads*dim+32]
    kv=[rng.integers(-2,3,(cache_rows,p)).astype(ml_dtypes.float8_e4m3fn) for p in pitches]
    packed=[buf(x.view("u1"),tl.float8e4nv) for x in kv]
    descriptors=[TensorDescriptor(x,[cache_rows,heads*dim],[p,1],[tile_n,dim]) for x,p in zip(packed,pitches)]
    scale=[rng.uniform(.7,1.1,(batch,heads)).astype("f4") for _ in range(2)]
    table=np.full((batch,max_len+7),-44,"i4")
    origins=np.full((batch,triton.cdiv(max_len,tile_n)+3),-44,"i4")
    for repeat in range(3):
        qlen=np.full(batch,qwidth,dtype="i4")
        if repeat==1:
            qlen[::3]=0
            qlen[-1]=max(1,qwidth-1)
        if repeat==2:
            qlen[:]=0
        lens[:]=rng.integers(qwidth,max_len+1,batch,dtype="i4")
        if not repeat:lens[0]=max_len
        cu[1:]=qlen.cumsum()
        first=np.zeros(batch,"i4") if window<0 else np.maximum(lens-qlen-window,0)//tile_n
        tile_counts=np.where(qlen>0,(lens+tile_n-1)//tile_n-first,0)
        boundaries=np.r_[0,tile_counts.cumsum()].astype("i4")
        width=max(1,(int(boundaries[-1])+parts-1)//parts)
        owner=np.searchsorted(boundaries[1:],np.arange(parts)*width,side="right")
        new._build_verify_raster[(triton.cdiv(parts,16),)](as_int(lens),as_int(cu),packed_plan,batch,parts,
                    window,tile_n,block_b,16)
        np.testing.assert_array_equal(plan[1:-1],np.r_[boundaries,first,width,owner])
        assert np.all(plan[[0,-1]]==-987654321)
        for i,length in enumerate(lens):
            slot=1+((i+repeat)%batch)*(max_len+256)
            address=slot+np.arange(length,dtype="i4")
            base=0 if window<0 else max(int(length)-qwidth-window,0)//tile_n
            for t in range(base,int(length)//tile_n):
                origins[i,t-base]=slot+t*tile_n
                if (t+repeat)%7==4:
                    a=t*tile_n
                    address[[a+3,a+5]]=address[[a+5,a+3]]
                    origins[i,t-base]=-1
            table[i,:length]=address
        query=rng.integers(-2,3,(batch*qwidth,heads*groups,dim)).astype(ml_dtypes.float8_e4m3fn)
        saved=None
        for mode,mod in (("parent",old),("planned",new),("fallback",new)):
            partial=np.full((parts+batch-1,heads,tile_m,dim),-71,"f4")
            lse=np.full(partial.shape[:-1],-91,"f4")
            out=np.full((batch*qwidth+1,heads*groups,dim),0x7f81,"u2")
            td=dict(Origins=as_int(origins),KDesc=descriptors[0],VDesc=descriptors[1],
                    ORIGIN_STRIDE=origins.shape[1],CAPACITY=cache_rows) if direct else {}
            planarg={"Plan":packed_plan} if mode=="planned" else {}
            ks,vs=[buf(x,tl.float32) if repeat!=1 else None for x in scale]
            mod._partition_attention[(parts,heads)](
                buf(query.view("u1"),tl.float8e4nv),*packed,as_int(table),as_int(lens),as_int(cu),ks,
                buf(partial,tl.float32),buf(lse,tl.float32),table.shape[1],1,pitches[0],pitches[0],dim,
                pitches[1],pitches[1],dim,heads,1,batch,heads,groups,dim,math.log2(math.e)/8,
                window,qwidth,tile_m,tile_n,parts,block_b,**td,**planarg)
            mod._merge_partitions[(batch,heads,triton.cdiv(qwidth*groups,rows))](
                buf(partial,tl.float32),buf(lse,tl.float32),as_int(lens),as_int(cu),vs,buf(out,tl.bfloat16),
                heads,1,batch,heads,groups,dim,window,tile_m,tile_n,parts,block_b,16,rows,**planarg)
            assert np.all(out[cu[-1]:]==0x7f81)
            result=(partial.view("u4"),lse.view("u4"),out)
            if saved is None:saved=result
            else:
                for a,b in zip(saved,result):np.testing.assert_array_equal(a,b)
                counts["equal_partition_and_lse_fp32_words"]+=partial.size+lse.size
                counts["equal_live_bf16_words"]+=int(cu[-1])*heads*groups*dim
                counts["comparisons_to_parent"]+=1
            if mode=="planned":
                for i,n in enumerate(qlen):
                    if not n:continue
                    last=int(lens[i])-1
                    hist=table[i,max(0,last-window) if window>=0 else 0:last+1]
                    h=i%heads; sl=slice(h*dim,(h+1)*dim); qrow=cu[i+1]-1
                    logits=kv[0][hist,sl].astype("f8") @ query[qrow,h*groups].astype("f8")
                    logits*=float(scale[0][i,h] if ks is not None else 1)/8
                    probs=np.exp(logits-logits.max()); probs/=probs.sum()
                    ref=(probs@kv[1][hist,sl].astype("f8"))*float(scale[1][i,h] if vs is not None else 1)
                    np.testing.assert_allclose(out[qrow,h*groups].view(ml_dtypes.bfloat16).astype("f4"),ref,rtol=.04,atol=.025)
                    counts["dense_fp64_reference_components"]+=dim
        counts["actual_planner_replays"]+=1


def main():
    paths=list(map(Path,sys.argv[1:]))
    geometry=[(4,2,8,8,677,-1,3,True),(4,2,8,8,601,255,7,True),
              (3,1,4,4,317,-1,17,False),(7,2,4,4,391,127,13,False),
              (16,2,8,8,134,2047,66,False),(32,2,8,8,130,-1,132,False)]
    counts=Counter(); rng=np.random.default_rng(248)
    native=InterpreterBuilder.create_dot
    def dot(self,a,b,acc,*flags):
        if a.dtype==b.dtype==tl.bfloat16:
            left=a.data.view(ml_dtypes.bfloat16).astype("f4");right=b.data.view(ml_dtypes.bfloat16).astype("f4")
            return type(acc)(left@right+acc.data,acc.dtype.scalar)
        return native(self,a,b,acc,*flags)
    with tempfile.TemporaryDirectory() as tmp:
        old,new=[load_jit_module(p,Path(tmp)) for p in paths]
        InterpreterBuilder.create_dot=dot
        try:
            for case in geometry:check_case(old,new,case,rng,counts)
        finally:InterpreterBuilder.create_dot=native
    print(json.dumps(dict(triton=triton.__version__,physical_H200=False,geometries=geometry,counts=counts,
          source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))


if __name__=="__main__":main()
