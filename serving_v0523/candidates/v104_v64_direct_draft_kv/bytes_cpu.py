"""Compare both production writers using a CPU Torch evaluator for their Triton bodies."""
import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch


class Pointer:
    def __init__(self,tensor,offset=0):
        self.tensor=tensor
        self.offset=offset
        self.dtype=NS(element_ty=getattr(tensor,"dtype",torch.uint64))

    def __add__(self,delta):
        return Pointer(self.tensor,self.offset+delta)

    def to(self,dtype):
        assert int(self.offset)==0
        return Pointer(self.tensor.view(dtype))


def load(pointer,mask=True,other=0.,**kw):
    if isinstance(pointer.tensor,list):
        return Pointer(pointer.tensor[int(pointer.offset)])
    off,mask=torch.broadcast_tensors(torch.as_tensor(pointer.offset,dtype=torch.int64),torch.as_tensor(mask,dtype=torch.bool))
    active=off[mask]
    flat=pointer.tensor.flatten()
    assert ((active>=0)&(active<flat.numel())).all()
    out=torch.full(off.shape,other,dtype=flat.dtype)
    out[mask]=flat[active]
    return out


def put(pointer,value,mask=True,**kw):
    off,value,mask=torch.broadcast_tensors(torch.as_tensor(pointer.offset,dtype=torch.int64),value,torch.as_tensor(mask,dtype=torch.bool))
    flat=pointer.tensor.flatten()
    active=off[mask]
    assert ((active>=0)&(active<flat.numel())).all()
    if flat.dtype==torch.float8_e4m3fn:
        flat.view(torch.uint8)[active]=value.to(flat.dtype).contiguous().view(torch.uint8)[mask]
    else:
        flat[active]=value[mask].to(flat.dtype)


def source(path, names, pid):
    module=ast.parse(path.read_text())
    module.body=[n for n in module.body if getattr(n,'name',None) in names]
    assert len(module.body)==len(names)
    src=ast.parse('from __future__ import annotations')
    src.body.extend(module.body)
    tl=NS(program_id=lambda d:pid[d],load=load,store=put,pointer_type=lambda dtype:dtype,
          cast=lambda p,d:p.to(d),arange=torch.arange,float32=torch.float32,int64=torch.int64,
          bfloat16=torch.bfloat16,float8e4nv=torch.float8_e4m3fn,uint8=torch.uint8,where=torch.where,
          rsqrt=torch.rsqrt,sum=torch.sum,div_rn=torch.div,multiple_of=lambda *a:None)
    env={"tl":tl,"triton":NS(jit=lambda f:f)}
    exec(compile(src,str(path),'exec'),env)
    return NS(**env)


def run(candidate,stock,pool_writers):
    pid=[0]*3
    orig=source(stock,{'_fused_norm_rope_kernel_stacked'},pid)
    new=source(candidate,{'_cache_round','_fused_norm_rope_kernel_stacked'},pid)
    pool=source(pool_writers,{'_saturate_to_fp8_range','set_kv_buffer_prefix_valid_tiled_fp8','set_kv_buffer_prefix_valid_tiled'},pid)
    rng=torch.Generator().manual_seed(64523)
    torch.set_num_threads(1)
    cases=0
    row_count=0
    byte_count=0
    for n,h,dim,rot,b,w in [(5,1,128,128,3,8),(5,2,64,32,2,7),(2,1,96,64,3,4),(1,3,24,16,2,3)]:
        size=h*dim
        m=b*w
        slots=m*2+5
        inp=(torch.randn((m,n,2*size),generator=rng)*10).bfloat16()
        # Raw BF16 patterns include both signed zeros, FP8 subnormal ties, saturation and NaNs.
        specials=torch.tensor([0x8000,0,0x7f80,0xff80,0x7fc0,0xffc0,0x43e0,0x43e1,0xc3e1,0x3b00,0x3a80,0x3f88,0x3f89],dtype=torch.int32).to(torch.int16)
        inp[0,0,size:size+len(specials)].view(torch.int16).copy_(specials)
        weight=torch.randn((n,dim),generator=rng).bfloat16()
        eps=torch.linspace(.00001,.00007,n)
        trig=torch.randn((55,rot),generator=rng).bfloat16()
        positions=torch.randint(0,55,(m,),generator=rng)
        tmp_k=torch.empty((n,m,h,dim),dtype=torch.bfloat16)
        tmp_v=torch.empty_like(tmp_k)
        common=list(map(Pointer,(inp,weight,eps,trig)))
        grid=[(r,head,layer) for r in range(m) for head in range(h) for layer in range(n)]
        block=1<<(dim-1).bit_length()
        config=[m,n,h,dim,size,rot,rot//2,block]
        for ix in grid:
            pid[:]=ix
            orig._fused_norm_rope_kernel_stacked(*common,Pointer(positions),Pointer(tmp_k),Pointer(tmp_v),
                  n*2*size,2*size,dim,rot,m*size,size,dim,m*size,size,dim,*config)
        for k in range(4): # includes zero, full, mixed and shorter prefixes.
            lengths=(torch.zeros(b,dtype=torch.int32) if k==0 else torch.full((b,),w,dtype=torch.int32) if k==1
                     else torch.randint(0,w+1,(b,),generator=rng,dtype=torch.int32))
            valid=torch.arange(m)%w < lengths[torch.arange(m)//w]
            loc=torch.randperm(slots,generator=rng)[:m]
            loc[~valid]=-2000000000
            pos=positions.clone()
            pos[~valid]=11000000000
            for is_fp8 in (False,True):
                dtype=torch.float8_e4m3fn if is_fp8 else torch.bfloat16
                key=torch.full((n,slots,h,dim),51,dtype=torch.uint8 if is_fp8 else torch.int16).view(dtype)
                val=torch.full_like(key.view(torch.uint8),174).view(dtype)
                other_k=key.clone()
                other_v=val.clone()
                for layer in range(n):
                    p=[Pointer(t) for t in (tmp_k[layer],tmp_v[layer],key[layer],val[layer],loc,lengths)]
                    for bid in range(b):
                        for row in range(w):
                            unit=size if is_fp8 else size*2
                            for tile in range((unit+63)//64):
                                pid[:]=bid,row,tile
                                if is_fp8:
                                    pool.set_kv_buffer_prefix_valid_tiled_fp8(*p,1.,1.,size,size,size,size,w,size,64,448.,False,False)
                                else:
                                    pool.set_kv_buffer_prefix_valid_tiled(*p,unit,unit,unit,unit,w,unit,64)
                for ix in grid:
                    pid[:]=ix
                    new._fused_norm_rope_kernel_stacked(*common,Pointer(pos),Pointer(list(other_k)),Pointer(list(other_v)),
                          n*2*size,2*size,dim,rot,0,0,0,0,0,0,*config,Pointer(loc),Pointer(lengths),w,is_fp8,1.)
                assert torch.equal(key.view(torch.uint8),other_k.view(torch.uint8))
                if not torch.equal(val.view(torch.uint8),other_v.view(torch.uint8)):
                    a=val.view(torch.uint8);bc=other_v.view(torch.uint8);mask=a!=bc
                    raise AssertionError((is_fp8,k,mask.nonzero()[:8],a[mask][:8],bc[mask][:8]))
                cases+=1
                row_count+=int(valid.sum())*h*n
                byte_count+=key.numel()*key.element_size()*2
    return dict(torch=torch.__version__,paired_bf16_fp8_cases=cases,verified_live_head_rows=row_count,
                compared_cache_bytes_including_untouched_rows=byte_count)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('candidate',type=Path)
    p.add_argument('stock',type=Path)
    p.add_argument('pool_writers',type=Path)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
