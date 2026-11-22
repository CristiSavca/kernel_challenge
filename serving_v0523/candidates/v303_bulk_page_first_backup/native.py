"""Build the exact gather lane body and DMA commands for use by the host differential."""
import ctypes
import subprocess
from pathlib import Path
from verify import digest
HEADER="python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh"
RELAY=HEADER.replace("staged_write_back","relayout")
def compile_cpu(dir,source):
    body=source[RELAY][source[RELAY].index("struct HicacheRelayoutParams"):source[RELAY].index("template <int64_t kElementSize, bool kIsMLA>\ninline void launch_")]
    h=source[HEADER]
    chunk=h[h.index("    if constexpr (kContiguous) {",h.index("auto stream = LaunchKernel::resolve_device(device);",h.index("struct HiCacheStagedWriteBackKernel"))):]
    copy=chunk[chunk.index("      const size_t bytes"):chunk.index("      return;")]
    code=r"""
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
struct alignas(16) uint4 {uint32_t a,b,c,d;};
struct {uint32_t x;} blockIdx,threadIdx,blockDim,gridDim;
#define __global__
#define __grid_constant__
#define RuntimeDeviceCheck(n) do {if((n)!=0) std::abort();} while(0)
namespace pointer {
void* offset(void *p,int64_t n){return static_cast<char*>(p)+n;}
const void* offset(const void *p,int64_t n){return static_cast<const char*>(p)+n;}
}
namespace device {namespace details {
uint4 load_nc(const uint4 *p){uint4 v;std::memcpy(&v,p,sizeof(v));return v;}
void store_nc(uint4 *p,uint4 v){std::memcpy(p,&v,sizeof(v));}
}}
struct Tensor {void* p;void *data_ptr(){return p;}};
struct Size {int64_t n;int64_t unwrap(){return n;}};
static const int stream=71,cudaMemcpyDeviceToHost=2;
static size_t copied,calls;
static int cudaMemcpyAsync(void*d,const void*s,size_t bytes,int direction,int on){
    if(direction!=cudaMemcpyDeviceToHost || on!=stream)return 1;
    std::memcpy(d,s,bytes);copied+=bytes;++calls;return 0;
}
""" +body+r"""
template<class I,int64_t kElementSize> int run(void*k,void*v,void*sk,void*sv,void*ids,
                                             void*ks,void*vs,int n,int layers){
    HicacheRelayoutParams params{sk,sv,ids,ks,vs,(uint32_t)n,(uint32_t)layers,1};
    blockDim.x=256;
    gridDim.x=(static_cast<uint64_t>(n)*layers*(kElementSize/16)+255)/256;
    for(blockIdx.x=0;blockIdx.x<gridDim.x;++blockIdx.x)
        for(threadIdx.x=0;threadIdx.x<blockDim.x;++threadIdx.x)
            hicache_relayout_kernel<I,kElementSize,false>(params);
    Tensor k_cache_dst{k},v_cache_dst{v},staging_k{sk},staging_v{sv};
    Size T{n},N{layers};calls=copied=0;
"""+copy+r"""
    return calls==2 && copied==2ULL*n*layers*kElementSize ? 0:1;
}
extern "C" int transfer(void*k,void*v,void*sk,void*sv,void*i,void*ks,void*vs,int n,int l,int w,int it){
#define CALL(I,W) run<I,W>(k,v,sk,sv,i,ks,vs,n,l)
    if(w==128)return it==4?CALL(int32_t,128):CALL(int64_t,128);
    if(w==256)return it==4?CALL(int32_t,256):CALL(int64_t,256);
    return 4;
}
"""
    dep=Path(dir);(dep/"check.cc").write_text(code)
    subprocess.run(["clang++","-Wall","-std=c++20","-O2","-shared","-fPIC","check.cc","-o","check.so"],cwd=dep,check=True)
    fn=ctypes.CDLL(str(dep/"check.so")).transfer
    fn.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*4;fn.restype=ctypes.c_int
    def launch(k,v,d,sk,sv,indices,ks,vs):
        pointers=lambda seq:(ctypes.c_void_p*len(seq))(*[a.data_ptr() for a in seq])
        n,ls,width=k.shape
        assert sk.shape==sv.shape==v.shape==k.shape
        assert fn(k.data_ptr(),v.data_ptr(),sk.data_ptr(),sv.data_ptr(),indices.data_ptr(),pointers(ks),pointers(vs),
                  n,ls,width*k.element_size(),indices.element_size())==0
    return launch,dict(gather_sha256=digest(body.encode()),dma_commands_sha256=digest(copy.encode()))
