"""Carry coordinate-labelled WGMMA accumulator elements through Humming's native global epilogue.

Usage: python independent_scatter.py extracted/humming/include/humming
"""
import hashlib
import json
import sys
from pathlib import Path

here=Path(__file__).resolve().parent
vendor=Path(sys.argv[1])
sources={name:(vendor/name).read_text() for name in ("epilogue/gmem_writer.cuh","epilogue/smem_writer.cuh","mma/wgmma.cuh")}
for filename,needle in (("epilogue/gmem_writer.cuh","gmem_row < (kIsIndexedGemm ? output_shape_m : block_output_shape_m)"),("epilogue/smem_writer.cuh","__shfl_xor_sync(0xffffffff, val, 4)"),
                        ("mma/wgmma.cuh","CRegistersType[WarpShape::N * 4 / MmaShape::N]")):
    assert needle in sources[filename],needle


def pack(m,bn,wn,sw):
    scratch=[None]*(m*bn)
    for warp in range(bn//wn):
        for j in range(wn//16):
            for mb in range(m//8):
                for half in range(2):
                    # One m64n{16,24} instruction emits two token columns per lane.
                    # The RF operand carries logical output channels; j advances
                    # 16 channels for this warp on the next source fragment.
                    raw=[]
                    for lane in range(32):
                        channel=warp*wn+j*16+half*8+lane//4
                        token=mb*8+(lane%4)*2
                        raw.append((token*bn+channel,(token+1)*bn+channel))
                    for lane,local in enumerate(raw):
                        other=raw[lane^4]
                        assembled=(local[0],other[0]) if lane//4%2==0 else (other[1],local[1])
                        sub=(lane%4)*2+(lane%8)//4
                        count=64//wn
                        c1=(((warp%count)*(8//count)+j*2+half)^((sub+sw)%8))*4+lane//8
                        c2=(warp//count)*m*32
                        dst=((8*mb+sub)*32+c1+c2)*2
                        assert scratch[dst:dst+2]==[None,None]
                        scratch[dst:dst+2]=assembled
    assert sorted(scratch)==list(range(m*bn))
    return scratch


def export(smem,m,bn,n,threads,sw,live,tile,absolute):
    stored={}
    for th in range(threads):
        for i in range(th,m*bn//8,threads):
            row,col=divmod(i,8)
            reg=(row*8+(col^((row+sw)%8)))*8
            token=row%m
            if token>=live:continue
            band=row//m*64+col*8
            for b,x in enumerate(smem[reg:reg+8]):
                dst=(absolute+token)*n+tile*bn+band+b
                assert dst not in stored and x==token*bn+band+b
                stored[dst]=x
    assert len(stored)==live*bn
    return len(stored)


cfg=json.loads((here/'configs.json').read_text())
count=0; by={}
for name in ('w2','w13'):
    n=4096 if name=='w2' else 8192
    for m in (16,24):
        for kind in ('native','wide3' if name=='w2' else 'wide'):
            case=cfg[f'{name}_{m}_{kind}']
            bn=case['block'][1];wn=case['warp'][1]
            checked=0
            for sw in range(8):
                scratch=pack(m,bn,wn,sw)
                for live in (0,1,m-1,m):
                    for offset,tile in ((0,0),(37,n//bn-1)):
                        checked+=export(scratch,m,bn,n,case['math_threads'],sw,live,tile,offset)
            by[f'{name}_{m}_{kind}']=checked
            count+=checked
print(json.dumps(dict(global_bf16_coordinates=count,cases=by,source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},
                      scope="Host coordinate replay from WGMMA fragments through the XOR shuffle, shared swizzle and grouped non-TMA-C global writer; no GPU arithmetic."),indent=2))
