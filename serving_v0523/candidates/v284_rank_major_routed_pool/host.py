"""Run the frozen CUDA arithmetic, layout and route scatter together against an independent native traversal."""
import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

CAND = Path(__file__).resolve().parent
HEADER = "python/sglang/kernels/jit/csrc/muse_spark/ranked_pool_norm.cuh"


def b32(words):
    return (words.astype(np.uint32)<<16).view(np.float32)


def bf16(x):
    bits = x.view(np.uint32)
    v = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    v[np.isnan(x)] = 0x7fff
    return v


def norm(rows, gain, eps):
    m,h = rows.shape
    vals=b32(rows)
    chunks=vals.reshape(m,-1,128,4)
    sums=np.zeros((m,128),np.float32)
    for v in range(chunks.shape[1]):
        for c in range(4):
            a=chunks[:,v,:,c].astype(np.float64)
            sums=(sums.astype(np.float64)+a*a).astype(np.float32)
    warps=sums.reshape(m,4,32)
    for stride in (16,8,4,2,1):
        warps=warps[:,:,:stride]+warps[:,:,stride:2*stride]
    sums=warps[:,:,0]
    total=(sums[:,0]+sums[:,2])+(sums[:,1]+sums[:,3])
    scale=np.float32(1)/np.sqrt(total/np.float32(h)+np.float32(eps))
    return bf16(b32(gain)*(scale[:,None]*vals))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    stage=parser.parse_args().stage
    source=(stage/HEADER).read_text()
    core=source[source.index("struct Params {"):source.index("}  // namespace ranked_pool")]
    core=core.replace("__global__ __launch_bounds__(kThreads) void", "Step")
    old="__shfl_down_sync(0xffffffff, sum, step)"
    assert core.count(old)==1 and core.count("__syncthreads();")==2
    core=core.replace(old,"(co_await Shuffle{sum,step})").replace("__syncthreads();","co_await std::suspend_always{};")
    rng=np.random.default_rng(28401)
    sums=dict(staged=0,padding=0,normed_token_words=0)
    cases=[]
    with tempfile.TemporaryDirectory(prefix="v284-host-") as name:
        tmp=Path(name); (tmp/"kernel.inc").write_text(core)
        lib=tmp/"kernel.so"
        subprocess.run(["c++","-std=c++20","-O2","-ffp-contract=off","-fPIC","-shared","-I",str(tmp),str(CAND/"kernel_host.cpp"),"-o",str(lib)],check=True)
        f=ctypes.CDLL(str(lib)).run
        f.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*4+[ctypes.c_float]
        f.restype=None
        for tp in (4,8):
            for m,h,k,density in [(1,2048,8,0.),(8,4096,8,.13),(17,4096,8,.4),(32,4096,8,1.),
                                   (128,4096,8,.13),(256,4096,8,.13),(16,1024,1,.5),(17,4096,8,0.)]:
                rows=bf16(rng.normal(size=(m,h)).astype(np.float32)); gain=bf16(rng.normal(size=h).astype(np.float32))
                if m==17:
                    rows[0,0:4]=[0x8000,0x8001,0x7f80,0x7fc5]
                    rows[1,0:4]=[0xffc1,0xff80,0x7f81,0x007f]
                packed=np.ascontiguousarray(rows.reshape(m,tp,h//tp).transpose(1,0,2))
                live=rng.random(m*k)<density
                ids=rng.integers(0,32,m*k)
                live_slots=np.flatnonzero(live)
                selected=live_slots[np.argsort(ids[live_slots],kind="stable")]
                prefix=m*k-len(selected)
                mapping=np.full(m*k,-1,np.int32)
                mapping[selected]=np.arange(prefix,m*k)
                oracle=np.full((m*k,h),0xa343,np.uint16); actual=oracle.copy()
                with np.errstate(invalid="ignore",over="ignore"):
                    oracle[prefix:]=norm(rows,gain,1e-6)[selected//k]
                    f(*[x.ctypes.data for x in (packed,gain,mapping,actual)],tp,m,h,k,1e-6)
                assert np.array_equal(actual,oracle),(m,tp,np.argwhere(actual!=oracle)[:5])
                sums["normed_token_words"]+=m*h
                sums["padding"]+=prefix*h
                sums["staged"]+=len(selected)*h
                cases.append(dict(tp=tp,rows=m,h=h,k=k,live_slots=len(selected)))
    print(json.dumps(dict(header_sha256=sha256(source.encode()).hexdigest(),cases=cases,words=sums,
                          mismatches=0,method="Literal CUDA body with host FP32 primitives, shared memory and coroutine shuffle/barriers; packed inputs vs independently computed ordinary ATen traversal, all nonlocal arena words preserved. Host rsqrt shared reference."),indent=2))


if __name__=="__main__":
    main()
