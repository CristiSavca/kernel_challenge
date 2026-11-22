"""Exercise the production peer fanout with separately allocated host owners, or emit SM90 cubins."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import triton
import triton.language as tl


def aot(kernel):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    assembler = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
    results = []
    for topk, hidden in ((1, 32), (3, 257), (4, 1024), (5, 2049), (8, 4096), (16, 8192)):
        width = min(triton.next_power_of_2(hidden), 1024)
        sig = dict(zip(kernel.arg_names, ["*i64","*i32","*i16","i32",*(["constexpr"]*4)]))
        bound = dict(TOPK=topk, H=hidden, WIDTH=width, ROUTES=triton.next_power_of_2(topk))
        bin = triton.compile(ASTSource(kernel, sig, constexprs=bound), target=GPUTarget("cuda",90,32))
        ptx = bin.asm["ptx"]
        # One statically shared producer load instruction sequence services all TOPK destinations.
        loads = [line.strip() for line in ptx.splitlines() if "ld.global" in line and ".b16" in line]
        stores = [line.strip() for line in ptx.splitlines() if "st.global" in line and ".b16" in line]
        assert loads and len(stores)==len(loads)*topk, (loads, stores)
        with tempfile.TemporaryDirectory() as scratch:
            src = Path(scratch) / "kernel.ptx"
            src.write_text(ptx)
            log = subprocess.run([assembler, "-v", "-arch=sm_90a",src,"-o",src.with_suffix(".cubin")],check=True,capture_output=True,text=True).stderr
        assert "0 bytes spill stores, 0 bytes spill loads" in log
        results.append(dict(topk=topk,hidden=hidden,registers=int(re.search(r"Used (\d+) registers",log)[1]),shared_bytes=bin.metadata.shared,loads=len(loads),stores=len(stores),cubin_sha256=hashlib.sha256(bin.asm["cubin"]).hexdigest()))
    return dict(configs=results,spills=0)


def interpret(kernel):
    from interpret_decode_kernels import buf
    import numpy as np
    rng = np.random.default_rng(230)
    words=redzones=cases=fanouts=empty=0
    seen=set()
    for size in (2,4,8):
        for topk,hidden in ((1,32),(3,257),(5,2049),(8,4096),(16,515)):
            for mode in ("remote","duplicates","all","mixed"):
                shard=7
                tokens=(size-1)*shard+4
                arr=np.arange(size*shard*hidden,dtype="u4").reshape(size,shard,hidden)
                arr=((arr*7411+cases*131)%65536).astype("u2")
                banks=[np.pad(block.reshape(-1),(5,7),constant_values=0xface) for block in arr]
                ptrs=np.asarray([b[5:].ctypes.data for b in banks],dtype="i8")
                ids=np.full((tokens,topk),-1,dtype="i4")
                if mode=="duplicates": ids[::3]=1
                if mode=="all": ids[:]=0
                if mode=="mixed": ids[:]=rng.integers(-8,8,ids.shape,dtype="i4").clip(-1,7)
                order=np.argsort(ids.ravel(),kind="stable")
                order=order[ids.ravel()[order]>=0]
                mapping=np.full(ids.shape,-1,'i4')
                mapping.ravel()[order]=np.arange(len(order),dtype="i4")
                target=arr.reshape(-1,hidden)[order//topk]
                raw=np.full(target.size+34,0xbeef,dtype='u2')
                width=min(1024,triton.next_power_of_2(hidden))
                kernel[(tokens,triton.cdiv(hidden,width))](buf(ptrs,tl.int64),buf(mapping,tl.int32),buf(raw[17:],tl.int16),shard,
                      TOPK=topk,H=hidden,WIDTH=width,ROUTES=triton.next_power_of_2(topk))
                np.testing.assert_array_equal(raw[17:-17],target.reshape(-1))
                assert (raw[:17]==0xbeef).all() and (raw[-17:]==0xbeef).all()
                for r,bank in enumerate(banks):
                    np.testing.assert_array_equal(bank[5:-7],arr[r].reshape(-1))
                    assert (bank[:5]==0xface).all() and (bank[-7:]==0xface).all()
                words+=target.size;cases+=1;redzones+=34+size*12
                fanouts+=int(((ids>=0).sum(1)>1).sum());empty+=int((ids<0).all(1).sum())
                seen.update(map(int,np.unique(target)))
    assert len(seen)==65536
    return dict(cases=cases,bf16_words=words,unchanged_guard_words=redzones,distinct_bf16_words=len(seen),multiple_destination_tokens=fanouts,no_destination_tokens=empty)

if __name__=="__main__":
    mode, path, runs = sys.argv[1:]
    sys.path.insert(0,runs)
    from compile_decode_sm90 import load_jit_module
    src=Path(path)
    with tempfile.TemporaryDirectory() as scratch:
        kernel=load_jit_module(src,Path(scratch))._peer_pack
        result=(aot if mode=="aot" else interpret)(kernel)
    print(json.dumps(dict(triton=triton.__version__,physical_gpu=False,kernel_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),**result),indent=2))
