"""Execute the production Triton prefix/guard refill against aliased writer storage on CPU."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import triton
import triton.language as tl
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import HostBuffer, TensorWrapper


class ArrayView(HostBuffer):
    """Expose NumPy's full backing allocation and real view offsets to Triton's interpreter."""

    def untyped_storage(self):
        base = self.array
        while isinstance(base.base, np.ndarray):
            base = base.base
        assert base.flags.c_contiguous
        return HostBuffer(base)

    def storage_offset(self):
        return (self.data_ptr() - self.untyped_storage().data_ptr()) // self.array.itemsize

    def new_empty(self, size, device):
        return ArrayView(np.empty(size, self.array.dtype))

    def set_(self, storage, offset, size, stride):
        item = self.array.itemsize
        self.array = np.ndarray(size, self.array.dtype, buffer=storage.array,
                                offset=offset * item, strides=tuple(s * item for s in stride))
        self.shape = size
        self.data = self.array


def buf(array, dtype):
    assert array.flags.c_contiguous
    return TensorWrapper(ArrayView(array), dtype)


def cases():
    # Include no prefix, guard partially covering tail, guard wholly covering tail, and
    # a tail longer than the guard. Wider head geometries exercise nondefault blocks.
    return [(16,4096,0,8),(32,4103,17,248),(64,4097,97,256),(128,16384,0,8),
            (128,16384,1,120),(128,16384,49152,248),(128,8193,4097,480),
            (256,4099,128,16),(512,4096,1025,8),(1024,4096,13,8)]


def check(fn,width,n,prefix,tails,index_type,rng):
    guard=256
    visible=prefix+n; capacity=visible+guard
    rows=max(capacity,visible+tails)
    used=np.array([visible],np.int32)
    cu=np.array([0,n],np.int32); starts=np.array([0,capacity],np.int32)
    slots=617; stride=visible+9; table=np.full((1,stride),-2000000,index_type)
    take=rng.integers(1,slots,size=prefix,dtype=index_type)
    table[0,:prefix]=take
    caches=[rng.integers(0,256,(slots,width+offset),np.uint8) for offset in (3,13)]
    source=[rng.integers(0,256,(n+tails,width),np.uint8) for _ in range(2)]
    block=min(32,4096//width)
    def launch(current,dest,inline):
        params=[buf(a,tl.uint8) for a in [*caches,*current]]
        params.extend([buf(table,tl.int32 if index_type==np.int32 else tl.int64)])
        params.extend(buf(a,tl.int32) for a in [used,cu,starts,used])
        params.extend(buf(t,tl.uint8) for t in dest)
        work=prefix if inline else visible
        fn[(min(64,triton.cdiv(work+guard,block)),1)](*params,caches[0].shape[1],caches[1].shape[1],stride,
                  ROW=width,BLOCK=block,GUARD=guard,INLINE_NEW=inline)
    # These contiguous slices are the real alias. Guard at each end is outside every legal destination.
    banks=[np.full((rows+2,width),0xA7,np.uint8) for _ in range(2)]
    spans=[b[1:-1] for b in banks]
    written=[a[prefix:prefix+n+tails] for a in spans]
    for wk,src in zip(written,source):wk[:]=src
    physical=[wk.copy() for wk in written]
    previous=[np.empty((capacity,width),np.uint8) for _ in range(2)]
    launch(source,previous,False)
    launch([w[:n] for w in written],[a[:capacity] for a in spans],True)
    for i,b in enumerate(banks):
        exp=np.concatenate((caches[i][take,:width],source[i][:n],np.zeros((guard,width),np.uint8)))
        np.testing.assert_array_equal(spans[i][:capacity],exp)
        np.testing.assert_array_equal(previous[i],exp)
        np.testing.assert_array_equal(physical[i],source[i])
        if tails>guard:np.testing.assert_array_equal(spans[i][capacity:],source[i][n+guard:])
        assert np.all(b[[0,-1]]==0xA7)
    return dict(width=width,query=n,prefix=prefix,tail=tails,index=np.dtype(index_type).name,
                compared_dense_bytes=2*capacity*width,verified_physical_write_bytes=2*(n+tails)*width)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["interpret","compile-fixtures"]);parser.add_argument("source",type=Path)
    args=parser.parse_args()
    if args.mode=="compile-fixtures":
        from itertools import product
        items=[]
        for width,idx,inline in product((16,32,64,128,256,512,1024),("i32","i64"),(False,True)):
            items.append(dict(file="kernels/ops/attention/batched_prompt_kv.py",kernel="_pack_prompt_rows",
                label=f"w{width}-{idx}-inline{int(inline)}",types=("*u8 "*4+"*"+idx+" "+"*i32 "*4+"*u8 *u8 i32 i32 i32").split(),
                num_warps=4,num_stages=3,constants=dict(ROW=width,BLOCK=min(32,4096//width),GUARD=256,INLINE_NEW=inline)))
        print(json.dumps(items));return
    rng=np.random.default_rng(246)
    with tempfile.TemporaryDirectory() as tmp:
        fn=load_jit_module(args.source,Path(tmp))._pack_prompt_rows
        result=[check(fn,w,n,p,t,idx,rng) for w,n,p,t in cases() for idx in (np.int32,np.int64)]
    print(json.dumps(dict(cases=result,compared_dense_bytes=sum(c["compared_dense_bytes"] for c in result),
                 physical_write_bytes=sum(c["verified_physical_write_bytes"] for c in result),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                 physical_H200=False,triton=triton.__version__),indent=2))


if __name__=="__main__":main()

