"""Compile and execute the actual CUDA lane/address control flow with scalar CPU memory ops."""
from pathlib import Path
import json
import subprocess
import tempfile
from verify import candidate,digest
header="python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh"
with candidate() as (_,tree,git):source=git("show",tree+":"+header).decode()
device=source[source.index("namespace device {"):source.index("#define SGL_HICACHE_KERNEL")]
left=device.index("// NVIDIA exposes")
right=device.index("}  // namespace details",left)
device=device[:left]+"""template<class T>T load_nc(const T *p){T v;std::memcpy(&v,p,sizeof(v));return v;}
template<class T>void store_nc(T *p,T v){std::memcpy(p,&v,sizeof(v));}
"""+device[right:]
params=source[source.index("struct HicacheKernelParams {"):source.index("///",source.index("struct HicacheKernelParams {"))]
begin=source.rindex("template <",0,source.index("SGL_HICACHE_KERNEL void hicache_transfer_all_layer"))
kernel=source[begin:source.index("/**",source.index("SGL_HICACHE_KERNEL void hicache_transfer_all_layer"))]
preamble="""
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <type_traits>
#define SGL_DEVICE inline
#define SGL_HICACHE_KERNEL
#define __grid_constant__
struct {uint32_t x;} blockIdx,threadIdx;
struct alignas(4) uint1{uint32_t a;};
struct alignas(8) uint2{uint32_t a,b;};
struct alignas(16) uint4{uint32_t a,b,c,d;};
namespace pointer{
const void *offset(const void*p,int64_t o){return static_cast<const char*>(p)+o;}
void *offset(void*p,int64_t o){return static_cast<char*>(p)+o;}
}
"""
test=r"""
template <typename T,int W> size_t drive(int layers,int length,int mode){
    int slots=3*length+23, pitch=layers*W, stride=W;
    size_t total=(size_t)slots*layers*W;
    std::vector<uint8_t> src(2*total), dst(2*total,0xab),expected(dst);
    uint32_t state=12345;
    for(auto& elem:src){state=1664525u*state+1013904223u;elem=state>>24;}
    std::vector<T> read(length),write(length);
    for(int i=0;i<length;i++){read[i]=((int64_t)i*104729+13)%slots;write[i]=mode? 3*i+2:i+7;}
    std::vector<const void*> sk(layers),sv(layers);
    std::vector<void*> dk(layers),dv(layers);
    for(int l=0;l<layers;l++){
        sk[l]=src.data()+(size_t)l*slots*W;sv[l]=src.data()+total+(size_t)l*slots*W;
        dk[l]=dst.data()+l*W;dv[l]=dst.data()+total+l*W;
        for(int i=0;i<length;i++)for(int side=0;side<2;side++)
            std::memcpy(expected.data()+side*total+((size_t)write[i]*layers+l)*W,
                        src.data()+side*total+((size_t)l*slots+read[i])*W,W);
    }
    HicacheKernelParams params{dk.data(),dv.data(),write.data(),sk.data(),sv.data(),read.data(),
                               stride,pitch,(uint32_t)length,(uint32_t)layers};
    const int blocks=std::min((length+127)/128,2);
    for(blockIdx.x=0;blockIdx.x<blocks;blockIdx.x++)
        for(threadIdx.x=0;threadIdx.x<1024;threadIdx.x++)
            hicache_transfer_all_layer<T,W,4,2,1024>(params);
    if(dst!=expected)std::abort();
    return dst.size();
}
int main(){
    size_t bytes=0,cases=0;
    for (int n:{1,127,128,129,255,256,257,769,2049}){
        for (int mode:{0,1}){
            bytes+=drive<int64_t,128>(16,n,mode)+drive<int32_t,128>(46,n,mode);
            bytes+=drive<int64_t,256>(5,n,mode)+drive<int32_t,256>(7,n,mode);
            cases+=4;
        }
    }
    std::cout<<cases<<" "<<bytes<<"\n";
}
"""
with tempfile.TemporaryDirectory() as d:
    parent=Path(d);(parent/"replay.cc").write_text(preamble+device+params+kernel+test)
    subprocess.run(["clang++","-std=c++20","-O2","-Wno-unknown-pragmas","replay.cc","-o","replay"],cwd=parent,check=True)
    cases,bytes_=map(int,subprocess.check_output([parent/"replay"],text=True).split())
print(json.dumps(dict(tree=tree,header_sha256=digest(source.encode()),native_kernel_sha256=digest(kernel.encode()),
    cases=cases,checked_host_bytes=bytes_,scope="Exact all-layer K/V, worker and lane loops from published header. Replaced inline PTX vector loads/stores with host memcpy. Checked all untouched/gap bytes."),indent=2))
