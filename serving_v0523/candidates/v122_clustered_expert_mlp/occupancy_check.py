"""Compile the production occupancy gate against checked CUDA device-query fixtures."""
from pathlib import Path
from itertools import product
import ctypes
import json
import subprocess
import sys
import tempfile

MOCK = r"""
#include <cstdint>
#include <cstddef>
#include <stdexcept>
using bf16_t=uint16_t;
#define CHECK_CUDA(stmt) do {if ((stmt)!=0) throw std::runtime_error("CUDA error");} while(0)
struct dim3 { unsigned x,y,z; dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){} };
struct cudaLaunchAttribute { int id; struct {dim3 clusterDim;} val; };
struct cudaLaunchConfig_t {dim3 gridDim,blockDim; size_t dynamicSmemBytes; cudaLaunchAttribute *attrs; int numAttrs;};
enum {cudaDevAttrClusterLaunch=1,cudaDevAttrMultiProcessorCount,cudaDevAttrMaxSharedMemoryPerBlock,
      cudaLaunchAttributeClusterDimension,cudaErrorInvalidValue,cudaErrorInvalidConfiguration,cudaErrorInvalidClusterSize,
      cudaErrorLaunchOutOfResources,cudaErrorNotSupported};
namespace muse_cluster_mlp {constexpr unsigned kCluster=8,kWarps=8,kRows=16;void run_kernel(){}}
int capable_,sms_,limit_,active_,status_,queried_,cleared_,experts_,inner_;
int cudaGetDevice(int* p){*p=0; return 0;}
int cudaDeviceGetAttribute(int* p,int attr,int dev){
 if(dev!=0) return 100; *p=attr==1 ? capable_:attr==2?sms_:limit_; return 0;}
int cudaGetLastError(){++cleared_;return 0;}
int cudaOccupancyMaxActiveClusters(int* p,void(*fn)(),cudaLaunchConfig_t* cfg){
 ++queried_;dim3 c=cfg->attrs->val.clusterDim,g=cfg->gridDim,b=cfg->blockDim;
 if(cfg->numAttrs!=1 || cfg->attrs->id!=4 || c.x!=8 || c.y!=1 || c.z!=1 || g.x!=8 ||
    g.y!=experts_ || g.z!=1 || b.x!=256 || b.y!=1 || b.z!=1 ||
    cfg->dynamicSmemBytes!=16*(inner_/8+8)*2 || fn!=muse_cluster_mlp::run_kernel)
   return 101;
 *p=active_;return status_;
}
"""
DRIVER = r"""
extern "C" int check(int dev,int h,int i,int e,int cap,int sms,int smem,int active,int status){
 capable_=cap;sms_=sms;limit_=smem;active_=active;status_=status;inner_=i;experts_=e;queried_=cleared_=0;
 int a=0;
 try{a=Probe::supported(dev,h,i,e)?1:0;}catch(std::runtime_error&){a=2;}
 return a + 10*queried_+100*cleared_;
}
"""

def main(stage):
    header=(stage/"python/sglang/kernels/jit/csrc/muse_spark/expert_cluster_mlp.cuh").read_text()
    gate=header.split("struct MuseClusterMLP {",1)[1].split("  static void run(",1)[0]
    n=0
    with tempfile.TemporaryDirectory() as work:
        folder=Path(work)
        src=folder/"check.cpp"
        src.write_text(MOCK+"struct Probe {"+gate+"};\n"+DRIVER)
        lib=folder/"check.so"
        subprocess.run(["c++","-std=c++20","-O2","-shared","-fPIC",str(src),"-o",str(lib)],check=True)
        check=ctypes.CDLL(str(lib)).check
        for dev,h,i,e,cap,sms,smem,active,status in product(
            [0,1],[0,256,512,4096],[512,4096,16384],[0,7,17,32],[0,1],
            [16,132],[16384,49152],[0,2],[0,5,6,7,8,9,99]
        ):
            probed=(dev==0 and h>0 and h%512==0 and i%512==0 and e>0 and cap and
                    e*8>=sms and 32*(i//8+8)<=smem)
            expected=0
            if probed:
                expected=10+(int(active>0) if status==0 else 100 if status<99 else 2)
            assert check(dev,h,i,e,cap,sms,smem,active,status)==expected,(dev,h,i,e,cap,sms,smem,active,status)
            n+=1
    print(json.dumps(dict(cases=n,checked_grid_block_cluster_smem=True,physical_hardware=False),indent=2))

if __name__=="__main__":
    main(Path(sys.argv[1]))
