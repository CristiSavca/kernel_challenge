"""Check the production Triton kernel with arbitrary BF16 codes and physical cache slots."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs"))
from compile_decode_sm90 import load_jit_module
from interpret_decode_kernels import buf


def compare(kernel, lut, rng, rows, query_width, cache_width, offset, loc_dtype, exhaustive=False,
            bad_slot=None):
    qs, ks, vs = query_width + offset, cache_width + 3, cache_width + 9
    dest_k, dest_v = cache_width + 7, cache_width + 11
    capacity = rows + 19
    loc = np.full((rows, 2), capacity + 500, dtype=loc_dtype)
    loc[:, 0] = rng.permutation(np.arange(1, capacity))[:rows]
    q, k, v = (rng.integers(0, 65536, (rows, pitch), dtype=np.uint16) for pitch in (qs, ks, vs))
    loc[-1, 0] = 0
    if exhaustive:
        for inp, width, shift in ((q,query_width,12345), (k,cache_width,0), (v,cache_width,32767)):
            inp[:, :width] = (np.arange(rows * width, dtype=np.uint32) + shift).reshape(rows, width)
            assert len(np.unique(inp[:-1, :width])) == 65536
    if bad_slot is not None:
        loc[0, 0] = bad_slot if bad_slot < 0 else capacity
    qout = np.full((rows, query_width), 150, dtype=np.uint8)
    kc = rng.integers(0, 256, (capacity + 2, dest_k), dtype=np.uint8)
    vc = rng.integers(0, 256, (capacity + 2, dest_v), dtype=np.uint8)
    ek, ev = kc.copy(), vc.copy()
    block = min(256, triton.next_power_of_2(query_width))
    kernel[(rows, triton.cdiv(query_width, block))](
        buf(q,tl.uint16),buf(k,tl.uint16),buf(v,tl.uint16),buf(qout,tl.uint8),
        buf(kc,tl.uint8),buf(vc,tl.uint8),buf(loc,tl.int64 if loc_dtype==np.int64 else tl.int32),
        buf(lut,tl.uint8),qs,ks,vs,dest_k,dest_v,2,capacity,
        Q_WIDTH=query_width,KV_WIDTH=cache_width,BLOCK=block,num_warps=4,debug=True,
    )
    live = (loc[:, 0] > 0) & (loc[:, 0] < capacity)
    ek[loc[live,0], :cache_width] = lut[k[live,:cache_width]]
    ev[loc[live,0], :cache_width] = lut[v[live,:cache_width]]
    assert np.array_equal(qout, lut[q[:,:query_width]])
    assert np.array_equal(ek,kc) and np.array_equal(ev,vc)
    return qout.size+kc.size+vc.size


def run(source, work, table_path):
    module = load_jit_module(source, work)
    kernel = module._encode_and_scatter
    lut = np.fromfile(table_path, dtype=np.uint8)
    assert lut.size == 65536
    rng = np.random.default_rng(3491)
    total, cases = 0, 0
    for rows in (1, 8, 31, 128, 256):
        for q, kv in ((384,96), (1024,128), (512,384)):
            for t in (np.int32, np.int64):
                total += compare(kernel, lut, rng, rows, q, kv, 0 if rows % 2 else 13, t)
                cases += 1
    for t in (np.int32,np.int64):
        total += compare(kernel, lut, rng, 513, 1024, 128, 17, t, exhaustive=True)
        cases += 1
    faults = 0
    for slot in (-1, 1000):
        try:
            total += compare(kernel, lut, rng, 4, 96, 64, 2, np.int64, bad_slot=slot)
        except Exception as exc:
            if 'KV cache slot out of range' not in str(exc):
                raise
            faults += 1
    return dict(mode="Triton CPU interpreter", triton=triton.__version__, configurations=cases,
                bytes_compared=total, bounds_tests=2, interpreter_raised_device_assert=faults,
                exhaustive_codes_per_input=65536, table_sha256=hashlib.sha256(lut.tobytes()).hexdigest())


def assemble(source, work):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.backends.nvidia.compiler import get_ptxas
    kernel = load_jit_module(source, work)._encode_and_scatter
    rows=[]
    for q,kv in ((128,64),(384,96),(512,384),(1024,128),(2048,256)):
        for loc in ('i32','i64'):
            constants={'Q_WIDTH':q,'KV_WIDTH':kv,'BLOCK':min(256,triton.next_power_of_2(q))}
            dynamic=[a for a in kernel.arg_names if a not in constants]
            types=['*u16']*3+['*u8']*3+['*'+loc,'*u8']+['i32']*7
            sig=dict(zip(dynamic,types,strict=True))
            sig.update({a:'constexpr' for a in constants})
            binary=triton.compile(ASTSource(kernel,sig,constexprs=constants),
                                  target=GPUTarget('cuda',90,32),options={'num_warps':4,'debug':True})
            ptx=work/'kernel.ptx'
            ptx.write_text(binary.asm['ptx'])
            assert '__assertfail' in binary.asm['ptx'] or 'trap;' in binary.asm['ptx']
            tool=get_ptxas(90).path
            result=subprocess.run([tool,'-arch=sm_90a','-v',str(ptx),'-o',str(work/'test.cubin')],
                                  capture_output=True,text=True,check=True)
            stats=[line.strip() for line in result.stderr.splitlines()
                   if re.search(r'spill stores|Used [0-9]+ registers',line)]
            rows.append(dict(q_width=q,kv_width=kv,slot=loc,device_assert=True,
                             sha256=hashlib.sha256(binary.asm['cubin']).hexdigest(),stats=stats))
    return dict(mode='SM90 offline assembly',triton=triton.__version__,cases=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--table',type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        result=run(args.source,Path(tmp),args.table) if args.table else assemble(args.source,Path(tmp))
    result['source_sha256']=hashlib.sha256(args.source.read_bytes()).hexdigest()
    print(json.dumps(result,indent=2))
