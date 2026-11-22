"""Compile full-shape CUDA and execute the exact Triton wave kernels on CPU memory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf

def fixtures():
    rows=[]
    def add(name,label,types,constants,warps,aligned):
        rows.append(dict(file="srt/layers/muse_spark/humming_prefill_waves.py",kernel=name,label=label,
                    types=types.split(),constants=constants,num_warps=warps,num_stages=3,aligned=aligned))
    for tokens in (2048,8192,16384):
        h=i=4096;k=8;e=32;s=tokens*k
        limit=(384<<20)//(2*(max(h,i)+max(h,2*i)))
        ordinary=s if s<=limit else min(tokens,limit-k+1)
        n=min(s,ordinary+k-1)
        fused=min(s,n*(max(h,i)+max(h,2*i))//(h+i))
        for plain,count,capacity in ((False,n,ordinary),(True,fused,fused if fused==s else fused-k+1)):
            label=f"{tokens}-{count}-plain{int(plain)}"
            add("_pack_wave",label,"*bf16 *i32 *i64 *i32 *bf16 *i32 *i32",
                dict(ROWS=count,EXPERTS=e,TOPK=k,HIDDEN=h,WIDTH=h,SPAN=triton.next_power_of_2(max(s,count)),PLAIN_IDS=plain),8,
                ["Inputs","SortedBins","SortedSlots","Packed","RowKeys","RowMap"])
            for first in (True,False):
                add("_load_wave",label+f"-first{int(first)}","*i32 *i32 *i32 *i32 *i32 *i32 *i1",
                    dict(TOKENS=tokens,ROWS=count,EXPERTS=e,WAVES=triton.cdiv(s,capacity),BLOCK=64,FIRST=first),1,
                    ["Edges","TokenBounds","Cursor","Bounds","Offsets","Interval","Active"])
    return rows

def one(kernel,case,plain,gen):
    t,k,e,cap,h=[case[key] for key in ("tokens","k","e","cap","width")]
    ids=np.array(case["ids"],np.int32)
    bins=np.array(case["bins"],np.int32);slots=np.array(case["slots"],np.int64)
    edges=np.array(case["edges"],np.int32);tb=np.array(case["bounds"],np.int32)
    waves=len(tb)-1;r=min(t*k,cap+k-1);span=triton.next_power_of_2(max(t*k,r));block=triton.next_power_of_2(e+1)
    # Include every BF16 bit pattern, special values, and repeated expert assignments.
    inputs=gen.integers(0,65536,size=(t,h),dtype=np.uint16);inputs.flat[:8]=[0,32768,32640,65408,32705,65473,1,32769]
    packed=gen.integers(0,65536,(r+2,h),dtype=np.uint16);rp=packed.copy()
    keys=np.full(r+2,-7312,np.int32);rkeys=keys.copy()
    rowmap=np.full(t*k+2,-1,np.int32);rmap=rowmap.copy()
    bounds=np.full(e+1,-1,np.int32);offsets=bounds.copy();interval=np.full(2,-1,np.int32)
    cur=np.array([97],np.int32);active=np.zeros(1,np.bool_)
    cursor=0;groups=[[] for _ in range(waves)]
    for token,row in enumerate(ids):
        wave=min(cursor//cap,waves-1)
        live=[(int(x),token*k+j) for j,x in enumerate(row) if 0<=x<e]
        groups[wave].extend(live);cursor+=len(live)
    checked=0;steps=0;empty=0
    for w in range(waves+1):
        z=min(w,waves-1)
        kernel._load_wave[(1,)](buf(edges,tl.int32),buf(tb,tl.int32),buf(cur,tl.int32),buf(bounds,tl.int32),
                buf(offsets,tl.int32),buf(interval,tl.int32),buf(active,tl.int1),TOKENS=t,ROWS=r,EXPERTS=e,
                WAVES=waves,BLOCK=block,FIRST=w==0,num_warps=1)
        assert cur[0]==w and bool(active[0])==(w<waves and tb[z]<t)
        assert np.array_equal(interval,tb[z:z+2]) and np.array_equal(bounds,edges[z*e:(z+1)*e+1])
        expected=sorted(groups[z],key=lambda p:p[0])
        exact_edges=(r-len(expected))+np.r_[0,np.cumsum(np.bincount([p[0] for p in expected],minlength=e))]
        assert np.array_equal(offsets,exact_edges)
        if w==waves:break
        kernel._pack_wave[(min(r,256),)](buf(inputs,tl.bfloat16),buf(bins,tl.int32),buf(slots,tl.int64),buf(bounds,tl.int32),
             buf(packed[1:-1],tl.bfloat16),buf(keys[1:-1],tl.int32),buf(rowmap[1:-1],tl.int32),
             ROWS=r,EXPERTS=e,TOPK=k,HIDDEN=h,WIDTH=triton.next_power_of_2(h),SPAN=span,PLAIN_IDS=plain,num_warps=8)
        for row,(x,slot) in enumerate(expected,r-len(expected)):
            rp[row+1]=inputs[slot//k];rmap[slot+1]=row;rkeys[row+1]=x if plain else (x+1)*span+slot
        assert np.array_equal(packed,rp) and np.array_equal(keys,rkeys) and np.array_equal(rowmap,rmap)
        checked+=packed.size;steps+=1;empty+=not expected
    assert np.array_equal(rowmap[1:-1]<0,(ids.ravel()<0)|(ids.ravel()>=e))
    return dict(waves=steps,zero_route_waves=empty,verified_payload_words=checked,local_routes=cursor)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("source",type=Path);ap.add_argument("cases",type=Path,nargs="?")
    args=ap.parse_args()
    if args.cases is None:
        print(json.dumps(fixtures()));return
    data=json.loads(args.cases.read_bytes())
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==data["source_sha256"]
    gen=np.random.default_rng(306)
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        module=load_jit_module(args.source,Path(tmp))
        for case in data["cases"]:
            for plain in (True,False):
                out=one(module,case,plain,gen)
                results.append(dict(layout=case["layout"],tokens=case["tokens"],width=case["width"],plain=plain,**out))
    summary={key:sum(c[key] for c in results) for key in ("waves","zero_route_waves","verified_payload_words","local_routes")}
    print(json.dumps(dict(source_sha256=data["source_sha256"],fixture_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              triton=triton.__version__,summary=summary,cases=results,scope="Production plan_waves on CPU torch. Production Triton byte-exact packer and wave metadata including terminal update on Linux CPU interpreter. No physical CUDA execution."),indent=2))

if __name__=="__main__":main()
