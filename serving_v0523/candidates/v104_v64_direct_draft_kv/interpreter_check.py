"""Run candidate and original norm/RoPE plus real prefix pool writers on host Triton."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    import triton
    import triton.language as tl
    cli=argparse.ArgumentParser()
    cli.add_argument("candidate",type=Path)
    cli.add_argument("stock",type=Path)
    cli.add_argument("pool_writers",type=Path)
    cli.add_argument("--cache-dtype",choices=("both","bf16","fp8"),default="both")
    args=cli.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"runs"))
    from compile_decode_sm90 import load_jit_module
    from interpret_decode_kernels import buf
    def bits(x):
        y=np.asarray(x,dtype=np.float32).view(np.uint32)
        return ((y+0x7fff+((y>>16)&1))>>16).astype(np.uint16)
    rng=np.random.default_rng(20261118)
    counts={"bf16_cache_cases":0,"fp8_cache_cases":0,"real_physical_rows_per_storage":0,
            "poisoned_uncommitted_rows_per_storage":0}
    import hashlib
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for key in ("old","new","pool"):
            (root/key).mkdir()
        old=load_jit_module(args.stock,root/"old")
        new=load_jit_module(args.candidate,root/"new")
        stores=load_jit_module(args.pool_writers,root/"pool")
        for heads,hd,rd,width in [(1,128,128,8),(2,64,32,4),(1,96,64,3)]:
            layers,batch=5,3
            total=batch*width
            size=hd*heads
            physical=total*2+3
            loc=np.array(rng.permutation(physical)[:total],dtype=np.int64)
            lens=np.array([width,0,max(1,width-2)],dtype=np.int32)
            valid=(np.arange(total)%width)<lens[np.arange(total)//width]
            loc[~valid]=-1700000000
            pos=rng.integers(0,65,total,dtype=np.int64)
            kv=bits(rng.normal(size=(total,layers,2*size))*15)
            # NaN, signed zero, infinities, BF16s near FP8 ties, and saturation.
            patterns=np.array([0x8000,0,0x7F80,0xFF80,0x7fc0,0xffc0,0x43e0,0x43e1,0xc3e1,0x3b00,0x3a80,0x3f88,0x3f89],np.uint16)
            kv[0,0,size:size+len(patterns)]=patterns
            kweight=bits(rng.normal(size=(layers,hd)))
            eps=np.full(layers,1.e-5,dtype=np.float32)
            cos=bits(rng.normal(size=(65,rd)))
            shared=[buf(kv,tl.bfloat16),buf(kweight,tl.bfloat16),buf(eps,tl.float32),buf(cos,tl.bfloat16)]
            shape=(layers,total,heads,hd)
            ktmp=np.empty(shape,dtype=np.uint16)
            vtmp=np.empty_like(ktmp)
            cfg=[total,layers,heads,hd,size,rd,rd//2,triton.next_power_of_2(hd)]
            old._fused_norm_rope_kernel_stacked[(total,heads,layers)](
                *shared,buf(pos,tl.int64),buf(ktmp,tl.bfloat16),buf(vtmp,tl.bfloat16),
                layers*2*size,2*size,hd,rd,total*size,size,hd,total*size,size,hd,*cfg)
            safe=pos.copy()
            safe[~valid]=1000000000000
            dtypes = (False,True) if args.cache_dtype=="both" else (args.cache_dtype=="fp8",)
            for fp8 in dtypes:
                typ=tl.float8e4nv if fp8 else tl.bfloat16
                dtype=np.uint8 if fp8 else np.uint16
                slab=(layers,physical,heads,hd)
                expected_k=np.full(slab,173,dtype=dtype)
                expected_v=np.full_like(expected_k,81)
                actual_k=expected_k.copy()
                actual_v=expected_v.copy()
                for layer in range(layers):
                    base=[buf(ktmp[layer],tl.bfloat16),buf(vtmp[layer],tl.bfloat16),
                          buf(expected_k[layer],typ),buf(expected_v[layer],typ),buf(loc,tl.int64),buf(lens,tl.int32)]
                    if fp8:
                        stores.set_kv_buffer_prefix_valid_tiled_fp8[(batch,width,(size+63)//64)](
                            *base,1.,1.,size,size,size,size,width,size,64,448.,False,False)
                    else:
                        byte_width=size*2
                        stores.set_kv_buffer_prefix_valid_tiled[(batch,width,(byte_width+127)//128)](
                            *base,byte_width,byte_width,byte_width,byte_width,width,byte_width,128)
                pts=lambda cache:buf(np.array([a.ctypes.data for a in cache],dtype=np.uint64),tl.uint64)
                new._fused_norm_rope_kernel_stacked[(total,heads,layers)](
                    *shared,buf(safe,tl.int64),pts(actual_k),pts(actual_v),
                    layers*2*size,2*size,hd,rd,0,0,0,0,0,0,*cfg,buf(loc,tl.int64),buf(lens,tl.int32),width,fp8,1.)
                np.testing.assert_array_equal(actual_k,expected_k)
                np.testing.assert_array_equal(actual_v,expected_v)
                counts["fp8_cache_cases" if fp8 else "bf16_cache_cases"]+=1
            counts["real_physical_rows_per_storage"]+=int(valid.sum())*layers*heads
            counts["poisoned_uncommitted_rows_per_storage"]+=int((~valid).sum())*layers*heads
    print(json.dumps({"interpreter":triton.__version__,"candidate_sha256":hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                      "stock_sha256":hashlib.sha256(args.stock.read_bytes()).hexdigest(),**counts},indent=2))


if __name__=="__main__":
    main()
