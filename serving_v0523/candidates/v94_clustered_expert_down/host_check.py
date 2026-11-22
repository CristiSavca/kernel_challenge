"""Test the actual load-time CUDA cluster probe against deterministic device API responses."""
import ctypes
import json
import pathlib
import subprocess
import sys
import tempfile


MOCK=r"""
#include <cstdint>
#include <stdexcept>
#define CHECK_CUDA(status) do { if ((status)!=0) throw std::runtime_error("unexpected CUDA error"); } while(false)
struct dim3 {unsigned x,y,z; dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){}};
struct cudaLaunchAttribute {int id; struct {dim3 clusterDim;} val;};
struct cudaLaunchConfig_t {dim3 gridDim,blockDim; cudaLaunchAttribute* attrs;int numAttrs;};
constexpr int cudaLaunchAttributeClusterDimension=1,cudaDevAttrClusterLaunch=2,cudaErrorInvalidValue=3;
constexpr int cudaErrorInvalidConfiguration=4,cudaErrorInvalidClusterSize=5,cudaErrorLaunchOutOfResources=6;
constexpr int cudaErrorNotSupported=7;
int device_,supported_,mask_,failure_;
int cudaGetDevice(int* p) {*p=device_;return 0;}
int cudaDeviceGetAttribute(int* p,int attr,int device) {
 if(attr!=cudaDevAttrClusterLaunch || device!=device_) return 102;
 *p=supported_;return 0;
}
int cudaGetLastError() {return 0;}
namespace muse_activated_down {
constexpr unsigned kColumns=512,kWarps=8;
template<unsigned C> void run_kernel(){}
}
int cudaOccupancyMaxActiveClusters(int* out,void(*func)(),cudaLaunchConfig_t* cfg) {
 using namespace muse_activated_down;
 unsigned c=cfg->attrs->val.clusterDim.x;
 if(cfg->numAttrs!=1 || cfg->attrs->id!=cudaLaunchAttributeClusterDimension ||
    cfg->gridDim.x%c || cfg->gridDim.y!=1 || cfg->gridDim.z!=1 || cfg->blockDim.x!=256 ||
    cfg->attrs->val.clusterDim.y!=1 || cfg->attrs->val.clusterDim.z!=1 ||
    ((c==2 && func!=run_kernel<2>) || (c==4 && func!=run_kernel<4>) || (c==8 && func!=run_kernel<8>))) return 103;
 if(failure_) return failure_;
 *out=(mask_&c)?1:0;
 return 0;
}
"""
def main(stage):
    source=(stage/'python/sglang/kernels/jit/csrc/muse_spark/expert_activated_down.cuh').read_text()
    src=MOCK+'struct Probe {\n'+source.split('struct MuseActivatedDown {\n',1)[1].split('  static void run(',1)[0]+'};\n'
    src+='''
extern "C" int probe(int device,int flag,int mask,int failure,int hidden) {
 device_=0;supported_=flag;mask_=mask;failure_=failure;
 try {return Probe::cluster_width(device,hidden);} catch (std::runtime_error&) {return -1;}
}'''
    cases=0
    with tempfile.TemporaryDirectory() as n:
        base=pathlib.Path(n); (base/'probe.cpp').write_text(src)
        out=base/'probe.so'
        subprocess.run(['c++','-std=c++20','-O2','-shared','-fPIC',str(base/'probe.cpp'),'-o',str(out)],check=True)
        probe=ctypes.CDLL(str(out)).probe
        for device in [0,1]:
            for flag in [0,1]:
                for mask in range(0,16,2):
                    for hidden in [-512,0,256,512,1024,1536,2048,3072,4096,8192]:
                        for error in [0,3,4,5,6,7,99]:
                            expected=1
                            if device==0 and flag and hidden>0 and hidden%512==0:
                                legal=[c for c in (8,4,2) if hidden//512%c==0]
                                if error==99 and legal: expected=-1
                                if error==0: expected=next((c for c in legal if mask&c),1)
                            assert probe(device,flag,mask,error,hidden)==expected
                            cases+=1
    print(json.dumps(dict(occupancy_probe_cases=cases),indent=2))


if __name__=='__main__': main(pathlib.Path(sys.argv[1]))
