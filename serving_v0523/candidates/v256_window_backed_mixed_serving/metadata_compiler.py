"""Compile and interpret V256's combined draft, native and target-certificate metadata kernel."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import triton
import triton.language as tl


def compile_modes(kernel):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    artifacts=[]
    # mode, SWA rail, window, translated pages, zero tail, page size, FULL/SWA certificates
    modes=[
        ("v188-full-draft",False,2047,False,False,1,False,False),
        ("v255-physical-draft",True,2047,False,False,1,False,False),
        ("v255-single-position",True,0,False,False,1,False,False),
        ("stock-full",False,-1,False,False,1,False,False),
        ("stock-swa",True,-1,False,False,1,False,False),
        ("stock-unified",True,-1,True,True,16,False,False),
        ("v248-full-certificates",True,-1,False,False,1,True,False),
        ("v248-swa-certificates",True,-1,False,False,1,False,True),
        ("v248-both-certificates",True,-1,False,False,1,True,True)]
    for mode,swa,win,unified,clear,page,full_cert,swa_cert in modes:
        for index in ("i32","i64"):
            for lengths in ("i32","i64"):
                block=128 if swa or unified else 256
                cs=dict(PAGE_SIZE=page,SEQ_LEN_DELTA=8,BLOCK_COLS=block,SKIP_PAGE_TABLE=False,
                    ZERO_TAIL=clear,WINDOW_LEFT=win,ORIGIN_STRIDE=1024 if full_cert else 0,
                    SWA_ORIGIN_STRIDE=34 if swa_cert else 0,SWA_WINDOW=2047 if swa_cert else -1,
                    VERIFY_WIDTH=8 if swa_cert else 0)
                ptrs=dict(seq_lens="*"+lengths,req_to_token="*i32",req_pool_indices="*"+index,
                    cache_seqlens_int32="*i32",cu_seqlens_k="*i32",page_table="*i32")
                optional={"swa_page_table":swa,"full_v2p":unified,"swa_v2p":swa and unified,
                    "full_to_swa_index_mapping":swa and not unified,"origins":full_cert,"swa_origins":swa_cert}
                for key,active in optional.items():
                    if active: ptrs[key]="*i64" if key=="full_to_swa_index_mapping" else "*i32"
                    else: cs[key]=None
                sig={key:("constexpr" if key in cs else ptrs.get(key,"i32")) for key in kernel.arg_names}
                compiled=triton.compile(ASTSource(kernel,sig,constexprs=cs),
                    target=GPUTarget("cuda",90,32),options={"num_warps":block//32,"num_stages":3})
                code=compiled.asm["cubin"]
                assert len(code)>1000
                artifacts.append(dict(mode=mode,index=index,lengths=lengths,cubin_bytes=len(code),
                    cubin_sha256=hashlib.sha256(code).hexdigest()))
    return {"target":"sm90","specializations":artifacts}


def native_controls(kernel):
    import numpy as np
    from interpret_decode_kernels import buf
    from math import ceil
    rng=np.random.default_rng(256)
    counts={"launches":0,"unbacked_live_IDS":0,"full_tile_certificates":0,"swa_tile_certificates":0}
    for batch in (1,3,8):
        pitch=3191
        seq=np.resize(np.array([2451,3180,199,0,127,2175,4,2176],np.int32),batch)
        rows=np.arange(batch,dtype=np.int64)[::-1].copy()
        table=np.arange(batch*pitch,dtype=np.int32).reshape(batch,pitch)+1
        for i in range(batch):
            table[i,141],table[i,142]=table[i,142],table[i,141]
        mapping=np.arange(batch*pitch+1,dtype=np.int64)+65;mapping[0]=0
        mapping[::227]=0
        for mode in ("unbacked","both"):
            uses_swa=mode=="both"; win=-1 if uses_swa else 2047
            extent=pitch if uses_swa else win+8+127
            block=128 if uses_swa else 256
            size_full=ceil(pitch/128);size_swa=ceil((2047+8+63)/64)
            dense=np.full((batch,pitch),-91,np.int32);swa=dense.copy()
            origins=np.full((batch,size_full),-411,np.int32);so=np.full((batch,size_swa),-612,np.int32)
            lens=np.empty(batch,np.int32);cu=np.empty(batch+1,np.int32)
            kwargs=dict(PAGE_SIZE=1,SEQ_LEN_DELTA=8,BLOCK_COLS=block,SKIP_PAGE_TABLE=False,ZERO_TAIL=False,WINDOW_LEFT=win)
            if uses_swa:
                kwargs.update(origins=buf(origins,tl.int32),ORIGIN_STRIDE=size_full,
                    swa_origins=buf(so,tl.int32),SWA_ORIGIN_STRIDE=size_swa,SWA_WINDOW=2047,VERIFY_WIDTH=8)
            kernel[(batch,triton.cdiv(extent,block))](buf(seq,tl.int32),1,buf(table,tl.int32),pitch,1,
                buf(rows,tl.int64),1,buf(lens,tl.int32),1,buf(cu,tl.int32),1,buf(dense,tl.int32),pitch,1,
                buf(swa,tl.int32) if uses_swa else None,pitch if uses_swa else 0,1 if uses_swa else 0,
                buf(mapping,tl.int64) if uses_swa else None,1 if uses_swa else 0,None,0,None,0,batch,extent,**kwargs)
            off=np.zeros_like(seq) if uses_swa else np.maximum(seq-win,0)//128*128
            np.testing.assert_array_equal(lens,seq+8-off)
            np.testing.assert_array_equal(cu,np.r_[0,lens.cumsum()])
            for b,(row,l,start) in enumerate(zip(rows,lens,off)):
                expected=table[row,start:start+l]
                np.testing.assert_array_equal(dense[b,:l],expected)
                if not uses_swa:
                    counts["unbacked_live_IDS"]+=int(l);continue
                np.testing.assert_array_equal(swa[b,:l],mapping[expected])
                for j in range(ceil(l/128)):
                    run=expected[128*j:128*(j+1)]
                    want=run[0] if len(run)==128 and np.array_equal(run,run[0]+np.arange(128)) else -1
                    assert origins[b,j]==want
                    counts["full_tile_certificates"]+=1
                first=max(int(seq[b])-2047,0); first_tile=first//64
                for tile in range(first_tile,ceil(l/64)):
                    left=tile*64; right=left+64; run=mapping[expected[left:right]]
                    valid=left>=first and right<=l and run[0]>0 and np.array_equal(run,run[0]+np.arange(64))
                    assert so[b,tile-first_tile]==(run[0] if valid else -1)
                    counts["swa_tile_certificates"]+=1
            counts["launches"]+=1
    return counts

mode=sys.argv[1];source=Path(sys.argv[2]);donor=Path(sys.argv[3])
spec=importlib.util.spec_from_file_location("v255_kernel",donor)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
with tempfile.TemporaryDirectory() as td:
    if mode=="schedule":
        from compile_decode_sm90 import load_jit_module
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        fn=load_jit_module(source,Path(td))._build_verify_raster
        schedules=[]
        for batch in (3,4,16,32):
            for win,part,tile in ((-1,132,128),(2047,66,64)):
                cs=dict(B=batch,PARTITIONS=part,WINDOW_LEFT=win,TILE_N=tile,
                    BLOCK_B=triton.next_power_of_2(batch),BLOCK_P=16)
                sig={"Lengths":"*i32","CuQ":"*i32","Plan":"*i32",**{k:"constexpr" for k in cs}}
                cu=triton.compile(ASTSource(fn,sig,constexprs=cs),target=GPUTarget("cuda",90,32),
                    options={"num_warps":4,"num_stages":3}).asm["cubin"]
                schedules.append(dict(batch=batch,window=win,bytes=len(cu),sha256=hashlib.sha256(cu).hexdigest()))
        result={"target":"sm90","target_verify_schedule":schedules}
    else:
        kernel=base.extract(source,Path(td))
        result=compile_modes(kernel) if mode=="compile" else dict(
            physical_window=base.interpret(kernel),other_controls=native_controls(kernel))
    print(json.dumps(dict(patch_sha256="68ae741c12ba65c54580ab0d855f60a16ad4bbb35ad99be4e6688f1e87e10494",
        source_path=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),triton=triton.__version__,
        mode=mode,physical_H200=False,**result),indent=2))
