"""Run current and ordinary metadata specializations using the actual Triton source."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import triton
import triton.language as tl


def extract(source, temp):
    nodes = [n for n in ast.parse(source.read_text()).body if getattr(n, 'name', '') in
             ('_resolve_metadata_page_ids', '_fused_decode_metadata_kernel')]
    lines = source.read_text().splitlines(keepends=True)
    path = temp/'verify_metadata_source.py'
    path.write_text('import triton\nimport triton.language as tl\n'+''.join(
        ''.join(lines[n.decorator_list[0].lineno-1:n.end_lineno])+'\n' for n in nodes))
    spec=importlib.util.spec_from_file_location(path.stem,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod._fused_decode_metadata_kernel


def compile_kernel(kernel):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    receipts=[]
    for win, page, translated, swa in [(2047,1,False,False),(-1,1,False,False),(-1,16,True,False),
                                      (-1,1,False,True),(-1,1,True,True)]:
        for idx in ('i32','i64'):
            for seq in ('i32','i64'):
                block=128 if translated or swa else 256
                cs=dict(PAGE_SIZE=page,SEQ_LEN_DELTA=8,BLOCK_COLS=block,SKIP_PAGE_TABLE=False,
                        ZERO_TAIL=translated,WINDOW_LEFT=win)
                ptrs=dict(seq_lens='*'+seq,req_to_token='*i32',req_pool_indices='*'+idx,
                          cache_seqlens_int32='*i32',cu_seqlens_k='*i32',page_table='*i32')
                for name,present in [('swa_page_table',swa),('full_v2p',translated),('swa_v2p',swa and translated),
                                     ('full_to_swa_index_mapping',swa and not translated)]:
                    if present:ptrs[name]='*i32'
                    else:cs[name]=None
                sig={a:('constexpr' if a in cs else ptrs.get(a,'i32')) for a in kernel.arg_names}
                bin=triton.compile(ASTSource(kernel,sig,constexprs=cs),target=GPUTarget('cuda',90,32),
                                   options={'num_warps':block//32,'num_stages':3})
                assert len(bin.asm['cubin'])>1000
                receipts.append(dict(window=win,page_size=page,translated=translated,swa=swa,req_dtype=idx,seq_dtype=seq,
                                     cubin_sha256=hashlib.sha256(bin.asm['cubin']).hexdigest()))
    return dict(arch='sm90',specializations=receipts)


def interpret(kernel):
    import numpy as np
    from interpret_decode_kernels import buf
    rng=np.random.default_rng(188)
    counts=dict(calls=0,real_rows=0,matched_physical_ids=0,checked_padding_columns=0)
    for B in (1,4,16,32):
        pitch=4489
        R=rng.permutation((B+3)*pitch).reshape(B+3,pitch).astype('i4')
        if B==32: R[0,:]=0
        for it in range(6):
            perm=rng.choice(B+3,B,replace=False).astype('i8' if it%2 else 'i4')
            P=rng.integers(0,pitch-8,size=B,dtype='i4')
            edges=(0,1,2046,2047,2048,2174,2175,2176,pitch-8)
            P[:min(B,4)]=np.resize(np.roll(edges,it),min(B,4))
            for win in (2047,-1):
                S=-np.ones(B,'i4');C=-np.ones(B+1,'i4');pt=np.full((B,pitch),-114188,'i4')
                width=pitch if win<0 else min(win+8+127,pitch)
                b=256
                kernel[(B,triton.cdiv(width,b))](buf(P,tl.int32),1,buf(R,tl.int32),pitch,1,
                    buf(perm,tl.int64 if it%2 else tl.int32),1,buf(S,tl.int32),1,buf(C,tl.int32),1,
                    buf(pt,tl.int32),pitch,1,None,0,0,None,0,None,0,None,0,B,width,
                    PAGE_SIZE=1,SEQ_LEN_DELTA=8,BLOCK_COLS=b,SKIP_PAGE_TABLE=False,ZERO_TAIL=False,WINDOW_LEFT=win)
                starts=np.maximum(P-2047,0)//128*128 if win>=0 else np.zeros_like(P)
                np.testing.assert_array_equal(S,P+8-starts)
                np.testing.assert_array_equal(C,np.r_[0,S.cumsum()])
                for row,(length,start,req) in enumerate(zip(S,starts,perm)):
                    np.testing.assert_array_equal(pt[row,:length],R[req,start:start+length])
                    assert np.all(pt[row,length:]==-114188)
                    counts['checked_padding_columns']+=int(pitch-length)
                counts['calls']+=1;counts['real_rows']+=B;counts['matched_physical_ids']+=int(S.sum())
    return counts


if __name__=='__main__':
    method,source=sys.argv[1:3];source=Path(source)
    with tempfile.TemporaryDirectory() as name:
        kernel=extract(source,Path(name))
        result=compile_kernel(kernel) if method=='aot' else interpret(kernel)
        print(json.dumps(dict(metadata_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                              triton=triton.__version__,method=method,physical_H200=False,**result),indent=2))
