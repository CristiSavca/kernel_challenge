"""Replay the unchanged CUDA body on the new one-partition direct-output geometry."""
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

here=Path(__file__).resolve().parent
donor=here.parent/"v284_rank_major_routed_pool"
sys.path.insert(0,str(donor))
import host
from host import HEADER,b32,bf16

def build(stage,tmp):
    source=(stage/HEADER).read_text()
    core=source[source.index("struct Params {"):source.index("}  // namespace ranked_pool")]
    core=core.replace("__global__ __launch_bounds__(kThreads) void","Step")
    core=core.replace("__shfl_down_sync(0xffffffff, sum, step)","(co_await Shuffle{sum,step})")
    assert core.count("__syncthreads();")==2
    (tmp/"kernel.inc").write_text(core.replace("__syncthreads();","co_await std::suspend_always{};"))
    lib=tmp/"kernel.so"
    subprocess.run(["c++","-std=c++20","-O2","-ffp-contract=off","-fPIC","-shared","-I",tmp,donor/"kernel_host.cpp","-o",lib],check=True)
    return lib

def load(lib):
    f=ctypes.CDLL(str(lib)).run
    f.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*4+[ctypes.c_float]
    f.restype=None
    return f

def reference(words,gain,eps):
    rows,width=words.shape
    if width>=512:return host.norm(words,gain,eps)
    vals=b32(words)
    sums=np.zeros((rows,128),np.float32)
    for i in range(4):
        a=vals[:,i::4].astype(np.float64)
        sums[:,:width//4]=(sums[:,:width//4].astype(np.float64)+a*a).astype(np.float32)
    warps=sums.reshape(rows,4,32)
    for n in (16,8,4,2,1):warps=warps[:,:,:n]+warps[:,:,n:2*n]
    s=warps[:,:,0]
    inv=np.float32(1)/np.sqrt(((s[:,0]+s[:,2])+(s[:,1]+s[:,3]))/np.float32(width)+np.float32(eps))
    return bf16(b32(gain)*(inv[:,None]*vals))

def main(stage):
    rng=np.random.default_rng(293)
    cases=[]; words=0;unused=0
    with tempfile.TemporaryDirectory(prefix="v293-kernel-") as td:
        f=load(build(stage,Path(td)))
        for n,h in ((1,4),(2,16),(15,256),(127,512),(257,2048),(1024,4096),(2048,4096),(2080,4096),(5,8192)):
            x=bf16(rng.normal(size=(n,h)).astype(np.float32)); gain=bf16(rng.normal(1,.4,size=h).astype(np.float32))
            out=np.full((2,n+3,h),0x5379,dtype=np.uint16);dest=out[1,0:n]
            m=np.arange(n,dtype=np.int32);eps=1e-6
            f(x.ctypes.data,gain.ctypes.data,m.ctypes.data,dest.ctypes.data,1,n,h,1,eps)
            want=reference(x,gain,eps)
            assert np.array_equal(dest,want) and np.all(out[0]==0x5379) and np.all(out[1,n:]==0x5379)
            words+=dest.size;unused+=(n+6)*h;cases.append([n,h])
    print(json.dumps(dict(cases=cases,words=int(words),neighbor_bank_and_padding=int(unused),mismatches=0,
        unchanged_cuda_sha256=sha256((stage/HEADER).read_bytes()).hexdigest(),
        scope="Host executes production CUDA per-lane body with explicit warp barriers, independent ATen vector/warp traversal. CPU rsqrt shared between both arithmetic models; physical H200 population remains unmeasured."),indent=2))
if __name__=="__main__":main(Path(sys.argv[1]).resolve())
