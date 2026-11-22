"""Compile or interpret the applied prompt pack. Provide the serving runs directory as the last argument."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import triton
import triton.language as tl

def interpret(kernel):
    from interpret_decode_kernels import buf
    import numpy as np
    rng = np.random.default_rng(221)
    results = []
    observed = set()
    for case in range(36):
        rows, hidden, topk = [(1, 7, 1), (9, 4096, 8), (21, 4097, 3), (23, 8192, 4),
                              (65, 63, 5), (8192, 11, 8), (16384, 5, 8), (1, 65536, 1),
                              (97, 257, 7)][case % 9]
        layout = case // 9
        col_stride = [1, 2, rows, 0][layout]
        row_stride = 1 if layout == 2 else (max(1, col_stride) * hidden + layout)
        if rows == 1 and hidden == 65536:
            col_stride, row_stride = 1, hidden
        storage = ((np.arange((rows-1)*row_stride+(hidden-1)*col_stride+1+4, dtype=np.uint32)
                    + 3741*case) % 65536).astype('u2')
        old = storage.copy()
        words = storage[2:]
        slots = rows * topk
        ids = np.full(slots, -1, 'i4')
        live = slots if rows == 1 or case % 5 == 0 and rows < 200 else min(slots, 43+case)
        picked = rng.choice(slots, live, replace=False)
        ids[picked] = rng.integers(0, 8, live) if case%3 else 7
        order = np.argsort(ids, kind='stable').astype('i8')
        first = slots-live
        mapped = np.full(slots+8, -991, 'i8' if case%2 else 'i4')
        packed = np.full(live*hidden+8, 0xdead, 'u2')
        kernel[(live+triton.cdiv(slots,256),)](buf(words,tl.bfloat16),buf(order,tl.int64),
            buf(packed[3:],tl.bfloat16),buf(mapped[3:],tl.int64 if case%2 else tl.int32),
            first,live,slots,row_stride,col_stride,TOPK=topk,HIDDEN=hidden,
            BLOCK_H=min(triton.next_power_of_2(hidden),4096),BLOCK_MAP=256,num_warps=4)
        addresses = (order[first:] // topk)[:,None]*row_stride + np.arange(hidden)*col_stride
        wanted = np.full_like(mapped,-991)
        wanted[3+order] = np.maximum(np.arange(slots)-first, -1)
        np.testing.assert_array_equal(mapped, wanted)
        np.testing.assert_array_equal(packed[3:-5].reshape(live,hidden), words[addresses])
        np.testing.assert_array_equal(storage,old)
        assert (packed[:3]==0xdead).all() and (packed[-5:]==0xdead).all()
        if hidden == 65536: observed.update(np.unique(packed[3:-5]).tolist())
        results.append(dict(rows=rows, hidden=hidden, topk=topk, live=live, col_stride=col_stride,
                            row_stride=row_stride, route_slots=slots, bf16_words=live*hidden))
    assert len(observed)==65536
    return dict(cases=results,distinct_bf16_encodings=len(observed),
                checked_bf16_words=sum(r['bf16_words'] for r in results),
                checked_route_slots=sum(r['route_slots'] for r in results))

def compile_sm90(kernel,tmp):
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from itertools import product
    rows=[]
    ptxas=Path(triton.__file__).parent/'backends/nvidia/bin/ptxas'
    for h, mtype, contig in product((31,4096,8192),('i32','i64'),(False,True)):
        constants=dict(TOPK=8,HIDDEN=h,BLOCK_H=min(triton.next_power_of_2(h),4096),BLOCK_MAP=256)
        types=dict(Hidden='*bf16',Slots='*i64',Inputs='*bf16',RowMap='*'+mtype,
                   first_local='i32',num_routes='i32',num_slots='i32',hidden_row_stride='i32')
        if contig:
            constants['hidden_col_stride']=1
        else:
            types['hidden_col_stride']='i32'
        types.update({k:'constexpr' for k in constants})
        attrs={(i,):[['tt.divisibility',16]] for i in (1,2,3)}
        if contig:
            attrs[(0,)]=[['tt.divisibility',16]]
            if h%16==0: attrs[(7,)]=[['tt.divisibility',16]]
        binary=triton.compile(ASTSource(kernel,types,constexprs=constants,attrs=attrs),
                              target=GPUTarget('cuda',90,32),options=dict(num_warps=4))
        ptx=binary.asm['ptx'];cub=binary.asm['cubin']
        assert not re.search(r'\bcvt\.[^;\n]*\b(?:bf16|f16|f32)',ptx) and 'st.global' in ptx and 'ld.global' in ptx
        (tmp/'probe.ptx').write_text(ptx)
        p=subprocess.run([ptxas,'-arch=sm_90a','-v',str(tmp/'probe.ptx'),'-o',str(tmp/'probe.cubin')],
                          capture_output=True,text=True,check=True)
        regs=int(re.search(r'Used (\d+) registers',p.stderr)[1])
        spills=[int(s) for s in re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',p.stderr).groups()]
        assert spills==[0,0]
        rows.append(dict(hidden=h,map_type=mtype,unit_col_stride=contig,registers=regs,
                         spill_bytes=spills,shared=binary.metadata.shared,
                         vector_global_loads=len(re.findall(r'ld\.global[^; ]*\.v[24]',ptx)),
                         cubin_bytes=len(cub),cubin_sha256=hashlib.sha256(cub).hexdigest()))
    return dict(cases=rows)

if __name__=='__main__':
    mode, source, tests = sys.argv[1:]
    sys.path.insert(0,tests)
    from compile_decode_sm90 import load_jit_module
    source=Path(source)
    with tempfile.TemporaryDirectory() as d:
        tmp=Path(d)
        kernel=load_jit_module(source,tmp)._pack_prompt_routes
        report=interpret(kernel) if mode=='interpret' else compile_sm90(kernel,tmp)
        print(json.dumps(dict(mode=mode,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                              triton=triton.__version__,physical_H200=False,**report),indent=2))
