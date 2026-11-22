"""Triton CPU interpreter and offline Hopper fixtures for the actual SWA span kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"runs"))
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf

FILE = "srt/layers/attention/prefill_kv_span.py"
BLOCK = 256


def fixtures():
    rows = []
    def add(fn, label, types, constants):
        rows.append(dict(file=FILE, kernel=fn, label=label, types=types.split(),
                         constants=constants, num_warps=4, num_stages=3))
    for dtype in ("i32","i64"):
        add("_check_window_rows", dtype, f"*{dtype} *i32 *i32 *i64 *i32 *i32 " + "i32 " * 7,
            {"BLOCK":BLOCK})
    for batch in (1,2,3,8,32):
        for width in (512,2047+512,16384+2047):
            total = batch * triton.cdiv(width,BLOCK)
            if width*batch<=262144:
                add("_finish_window_rows",f"b{batch}-w{width}","*i64 *i32 *i32 *i64 i32 i32",dict(
                    SINGLE=batch==1, ROW_BLOCK=triton.next_power_of_2(batch),
                    TILE_BLOCK=triton.next_power_of_2(total)))
    return rows


def expected(table, lengths, offsets, left, total, width):
    starts, sizes = [], []
    valid = offsets[0] == 0 and offsets[-1] == total
    for row, length in enumerate(lengths):
        q = int(offsets[row+1]) - int(offsets[row])
        k = int(length)
        first = max(0,k-q-left)
        visible = k-first
        if not (0<=offsets[row]<offsets[row+1]<=total and q<=k<=table.shape[1] and 0<visible<=width):
            return False, None, None
        slots = table[row,first:k].astype(np.int64)
        valid &= ((slots>0)&(slots<2**31)).all() and np.all(slots[1:]-slots[:-1]==1)
        starts.append(int(slots[0]))
        sizes.append(visible)
    stride = sizes[0] if len(starts)==1 else starts[1]-starts[0]
    good = (stride>=sizes[0] and len(set(sizes))==1
            and all(v==starts[0]+i*stride for i,v in enumerate(starts)))
    return bool(valid and good), sizes, [starts[0],stride,sizes[0],starts[-1]+sizes[-1]-1]


def run_case(mod, keys, queries, left, dtype, kind):
    keys = np.asarray(keys,dtype=np.int32); qs=np.asarray(queries,dtype=np.int32)
    b=len(keys); sw=int(max(keys)); total=int(sum(qs)); width=min(sw,int(max(qs))+left)
    pitch=sw+9
    table=np.full((b,pitch),-17279,dtype=dtype)
    offsets=np.concatenate(([0],np.cumsum(qs))).astype(np.int32)
    for i,(k,q) in enumerate(zip(keys,qs)):
        start=max(0,int(k-q)-left)
        # Expired rows include zero, invalid, duplicate and unowned physical slots.
        table[i,:start]=np.arange(start)%5-2
        first=1+i*(width+33)
        table[i,start:k]=first+np.arange(k-start)
    if kind=="fragment": table[-1,keys[-1]-2]+=45
    if kind=="zero": table[0,keys[0]-1]=0
    if kind=="ceiling": table[0,keys[0]-1]=np.iinfo(np.int32).max
    if kind=="overflow" and dtype==np.int64: table[-1,keys[-1]-1]=2**31+20
    if kind=="decreasing" and b>1: table[1,:keys[1]]=table[0,:keys[1]]
    if kind=="affine" and b>2: table[-1,:keys[-1]]+=1
    if kind=="capacity": width=max(1,width-1)
    if kind=="qzero":offsets[1]=0
    if kind=="qstart":offsets[0]=1
    if kind=="qpadded":total+=3
    if kind=="overlong":keys[-1]=sw+1
    if kind=="negkey":keys[-1]=-3
    if kind=="underq":keys[-1]=qs[-1]-1
    if kind=="leftzero":assert left==0
    tiles=triton.cdiv(width,BLOCK)
    first=np.zeros(b,dtype=np.int64)
    visible=np.full(b,-1,dtype=np.int32)
    valid=np.full((b,tiles),-2,dtype=np.int32)
    status=np.full(8,-17,dtype=np.int64)
    fn=mod._check_window_rows
    fn[(tiles,b)](buf(table,tl.int32 if dtype==np.int32 else tl.int64),buf(keys,tl.int32),
        buf(offsets,tl.int32),buf(first,tl.int64),buf(visible,tl.int32),buf(valid,tl.int32),
        pitch,sw,width,b,tiles,total,left,BLOCK=BLOCK)
    mod._finish_window_rows[(1,)](buf(first,tl.int64),buf(visible,tl.int32),buf(valid,tl.int32),
        buf(status[1:-1],tl.int64),b,b*tiles,SINGLE=b==1,
        ROW_BLOCK=triton.next_power_of_2(b),TILE_BLOCK=triton.next_power_of_2(b*tiles))
    want,sizes,shape=expected(table[:,:sw],keys,offsets,left,total,width)
    assert bool(status[1])==want and status[2]==0 and status[0]==status[-1]==-17,(kind,status,want)
    if want:
        assert status[3:7].tolist()==shape and visible.tolist()==sizes,(kind,status,visible,sizes)
    return int(want)


def main(path):
    scenarios=[([519],[513],0),([512,882],[512,512],0),([513],[513],2047),
        ([8192],[1024],2047),([900,1400],[513,513],23),([900,1400,2000],[513,513,513],23),
        ([1420,513],[513,513],2047),([600,1300],[313,513],31),
        ([74160],[16384],2047),([20000,36000,52500,70000],[2000]*4,2047)]
    kinds=["linear","fragment","zero","ceiling","overflow","decreasing","affine","capacity","qzero",
           "qstart","qpadded","overlong","negkey","underq"]
    counts={"cases":0,"dense":0}
    with tempfile.TemporaryDirectory() as tmp:
        mod=load_jit_module(path,Path(tmp))
        for keys,queries,left in scenarios:
            # Long fixtures still exercise every tile and the final partial tile.
            todo=kinds if max(keys)<6000 else ["linear","fragment","overlong","qpadded"]
            for dtype in (np.int32,np.int64):
                for kind in todo:
                    counts["dense"]+=run_case(mod,keys,queries,left,dtype,kind)
                    counts["cases"]+=1
    return dict(**counts,stock_fallback=counts["cases"]-counts["dense"],triton=triton.__version__,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physical_gpu=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("module",type=Path)
    parser.add_argument("--fixtures",action="store_true")
    args=parser.parse_args()
    print(json.dumps(fixtures() if args.fixtures else main(args.module),indent=2))
