"""Exercise the submitted two-stage GPU certificate using Triton's CPU interpreter."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs"))
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf


def case(module, lengths, declared, dtype, kind, residue):
    batch, width = len(lengths), max(lengths)
    tiles = (width + 159) // 160
    base = 3 * 160 + residue
    starts = base + np.arange(batch) * ((width // 160 + 6) * 160)
    if kind == "unequal_stride":
        starts[-1] += 160
    col = np.arange(width)
    rows = np.full((batch, width + 7), -999, dtype=dtype)
    for b, length in enumerate(lengths):
        row = starts[b] + col[:length]
        if kind in ("relocated", "wrong_residue"):
            row = row + (col[:length] // 160 % 2) * (3200 + (kind == "wrong_residue"))
        if kind == "fragment":
            row[length // 2] += 9
        if kind == "zero_slot":
            row[-1] = 0
        if kind == "overflow":
            row[0] = 2**31 + 7
        rows[b, :length] = row
    lengths_gpu = np.array(declared, dtype=np.int32)
    p = np.full((batch,tiles), -9, dtype=np.int32)
    flags = np.full_like(p, -99)
    high = np.full_like(p, -99)
    status = np.full(8, -123, dtype=np.int64)
    irows = tl.int64 if dtype == np.int64 else tl.int32
    module._check_span_tiles[(tiles,batch)](
        buf(rows,irows),buf(lengths_gpu,tl.int32),buf(p,tl.int32),buf(flags,tl.int32),
        buf(high,tl.int32),width+7,width,tiles,SINGLE=batch==1,
    )
    module._finish_span_check[(1,)](
        buf(rows,irows),buf(lengths_gpu,tl.int32),buf(flags,tl.int32),buf(high,tl.int32),
        buf(status[1:-1],tl.int64),width+7,batch*tiles,SINGLE=batch==1,
        BLOCK=triton.next_power_of_2(batch*tiles),
    )
    f = int(rows[0,0])
    gap = int(rows[1,0])-f if batch > 1 else declared[0]
    legal = all(0 < n <= width and (rows[b,:n] > 0).all() and (rows[b,:n] < 2**31).all()
                for b,n in enumerate(declared))
    d = legal and len(set(declared)) == 1 and gap >= declared[0] and all(
        np.array_equal(rows[b,:n].astype(np.int64), f+b*gap+np.arange(n)) for b,n in enumerate(declared)
    )
    c = legal and all(n % 160 == 0 for n in declared)
    if c:
        for b,n in enumerate(declared):
            for t in range(0,n,160):
                tile = rows[b,t:t+160]
                c &= int(tile[0]) % 160 == f % 160 and np.array_equal(tile, tile[0]+np.arange(160))
    assert (bool(status[1]),bool(status[2])) == (bool(d),bool(c)), (kind,declared,status)
    assert status[3:6].tolist() == [f,gap,declared[0]]
    assert status[0] == status[-1] == -123
    if d or c:
        assert status[6] == max(int(rows[b,:n].max()) for b,n in enumerate(declared))
        for b,n in enumerate(declared):
            if c:
                np.testing.assert_array_equal(p[b,:n//160]*160+f%160,rows[b,:n:160])
    return d, c


def main(path):
    counts = {"cases":0,"dense":0,"coarse":0,"neither":0}
    with tempfile.TemporaryDirectory() as work:
        module = load_jit_module(path, Path(work))
        for dtype in (np.int32, np.int64):
            for lens in ([512],[640],[160,480],[641,641],[320,320,320],[1221,743,950]):
                for kind in ("linear","unequal_stride","relocated","wrong_residue","fragment","zero_slot","overflow"):
                    if kind == "overflow" and dtype == np.int32:
                        continue
                    for residue in (0,1,159):
                        d,c = case(module, lens, lens, dtype, kind, residue)
                        counts["cases"]+=1
                        counts["dense"]+=int(d)
                        counts["coarse"]+=int(c)
                        counts["neither"]+=int(not (d or c))
            for declared in ([0,160],[-1,160],[161,160]):
                case(module,[160,160],declared,dtype,"linear",1)
                counts["cases"]+=1
                counts["neither"]+=1
        for lens,kind in (([74160],"linear"),([16320]*4,"relocated"),([16384,16384],"fragment")):
            d,c=case(module,lens,lens,np.int64,kind,1)
            counts["cases"]+=1
            counts["dense"]+=int(d)
            counts["coarse"]+=int(c)
            counts["neither"]+=int(not (d or c))
    return dict(**counts, triton=triton.__version__, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                interpreter=os.environ.get("TRITON_INTERPRET"), physical_gpu=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("module",type=Path)
    print(json.dumps(main(parser.parse_args().module),indent=2))
