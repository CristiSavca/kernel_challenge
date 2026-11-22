"""Check source operand layouts and the complete per-output K schedule for 32- and 64-channel warps."""
from hashlib import sha256
import json,random,struct,sys
from pathlib import Path

root=Path(sys.argv[1])/'include/humming'
sources={name:(root/name).read_text() for name in ['memory/s2r_loader/loader_b.cuh','memory/s2r_loader/loader_bs.cuh',
    'epilogue/smem_writer.cuh','epilogue/smem_reducer.cuh','kernel/humming_ws.cuh','mma/wgmma.cuh','datatype/dequant.cuh']}
for f,needle in [('memory/s2r_loader/loader_b.cuh','warp_id % 2 * kLoadIters + j'),
        ('memory/s2r_loader/loader_bs.cuh','n_warp_id % kNumWarpsPerPackBlock + pack_block_id * kPackBlockSize'),
        ('epilogue/smem_writer.cuh','(n_warp_id / count) * (BlockShape::M / kNumWriteSplits * 64 / 2)'),
        ('datatype/dequant.cuh','uint32_t qb_index = kResultPackedNumSourceBits * index / 32;'),
        ('kernel/humming_ws.cuh','slice_iters--'),('epilogue/smem_reducer.cuh','BlockShape::K / WarpShape::K')]:
    assert needle in sources[f]

def writer(w,n,lane,row,band,m,swizzle):
    sub=(lane%4)*2+lane%8//4
    col1=(((w%(64//n))*(n//8)+band)^((sub+swizzle)%8))*4+lane//8
    return (8*row+sub)*32 + col1 + (w//(64//n))*(m*32)

words=scales=writes=0
for n,k in ((128,256),(256,128)):
    nw=n//32
    for kg in range(k//64):
     for it in range(4):
      off=(4*kg+it)
      for w in range(nw):
       for lane in range(32):
        for b in range(4):
            old_cta,old=divmod(2*w+b//2,nw)
            native_word=(off*2*n+((old//2*32+lane)*2+old%2)*2+b%2)
            joined_word=off*4*n + old_cta*2*n + native_word%(2*n)
            wide_word=off*4*n+(w*32+lane)*4+b
            assert joined_word==wide_word
            words+=1
        for b in range(8):
            old_cta,old=divmod(2*w+b//4,nw)
            native=off*n+((old//2)*16+(lane//4)*2+old%2)*4+b%4
            wide=off*2*n+(w*8+lane//4)*8+b
            assert off*2*n+old_cta*n+native%n==wide
            scales+=1
    for m in (8,16,24):
     for sw in range(8):
      owners=set()
      for w in range(nw):
       for lane in range(32):
        for row in range(m//8):
         for band in range(8):
            cta,old=divmod(2*w+band//4,nw)
            ref=cta*m*n//2+writer(old,32,lane,row,band%4,m,sw)
            got=writer(w,64,lane,row,band,m,sw)
            assert ref==got and got not in owners
            owners.add(got);writes+=1
      assert len(owners)==m*n

f=struct.Struct('f')
def f32(x):return f.unpack(f.pack(x))[0]
def ordered(x,y,bk,stages):
    partial=[0.0]*(bk//64)
    visited=[]
    for ring in range(0,4096//bk,stages):
     for stage in range(stages):
      block=ring+stage
      if block>=4096//bk:break
      for g in range(bk//64):
       for it in range(4):
        for k in range(block*bk+g*64+16*it,block*bk+g*64+16*(it+1)):
         # The BF16 product is exact in FP32. Generated operands keep its addition exact in FP64.
         partial[g]=f32(partial[g]+x[k]*y[k]);visited.append((k,g))
    assert len(visited)==4096 and len(set(visited))==4096
    if len(partial)==2:return f32(partial[0]+partial[1]),visited
    return f32(f32(partial[0]+partial[2])+f32(partial[1]+partial[3])),visited

rng=random.Random(275)
same=neg=0
for n,bk,new_stages in ((128,256,3),(256,128,4)):
 for case in range(96):
    x=[rng.choice([-1,1])*(1+rng.randrange(128)/128)*2**rng.randrange(-5,6) for _ in range(4096)]
    scale=[(1+rng.randrange(8)/8)*2**rng.randrange(-5,6) for _ in range(256)]
    y=[rng.choice([-1,1])*rng.choice((0,.5,1,1.5,2,3,4,6))*scale[i//16] for i in range(4096)]
    before,order=ordered(x,y,bk,4);after,neworder=ordered(x,y,bk,new_stages)
    assert order==neworder and f.pack(before)==f.pack(after);same+=1
    linear=0.
    for a,b in zip(x,y):linear=f32(linear+a*b)
    neg+=f.pack(linear)!=f.pack(before)
assert neg>0
# BF16 round-to-nearest even, finite positive counterexample.
def bf16(x):
    bits=struct.unpack('I',f.pack(x))[0]
    return f.unpack(struct.pack('I',(bits+0x7fff+((bits>>16)&1))&0xffff0000))[0]
dot=1+2**-8
post=bf16(bf16(dot)*1.5);folded=bf16(1.5+1.5*2**-8)
assert post==1.5 and folded==1.5078125
print(json.dumps(dict(packed_word_coordinates=words,fp4_coordinates=8*words,scale_byte_coordinates=scales,
    native_and_wide_output_half2_coordinates=writes,full_K4096_dot_cases=same,linear_order_counterexamples=neg,prepacked_global_scale_counterexample=dict(activation=[1,2**-8],FP4_weights=[1,1],block_scale=1,global_scale=1.5,native=post,weight_folded=folded),
    actual_vendor_headers={k:sha256(v.encode()).hexdigest() for k,v in sources.items()},
    scope='Lane-local packed-word, scale, and post-transpose writer addresses in native Humming; host FP32 schedule model, not physical GPU arithmetic.'),indent=2))
