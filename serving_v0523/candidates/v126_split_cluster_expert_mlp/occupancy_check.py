"""Exercise the exact split-kernel occupancy probe and inspect every CUDA launch attribute."""
import ctypes
import importlib.util
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
source=Path(__file__).resolve().parents[1]/"v122_clustered_expert_mlp/occupancy_check.py"
spec=importlib.util.spec_from_file_location("v122_occupancy",source)
parent=importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
parent_mock=parent.MOCK

def main(stage):
    h=(stage/"python/sglang/kernels/jit/csrc/muse_spark/expert_split_mlp.cuh").read_text()
    body=h.split("struct MuseSplitMLP {",1)[1].split("  static void run(",1)[0]
    pre=parent_mock.split("namespace muse_cluster_mlp {",1)[0]
    pre+=r"""
namespace muse_split_mlp {
constexpr uint32_t kCluster=8,kWarps=8;using Stripes=char[32768];
template<unsigned P>void split_kernel(){}
}
int cap_,sms_,limit_,inner_,experts_,mask_,err_,clears_,tested_;
int cudaGetDevice(int* out){*out=0;return 0;}
int cudaDeviceGetAttribute(int* out,int attr,int dev){if(dev!=0)return 100;*out=attr==1?cap_:attr==2?sms_:limit_;return 0;}
int cudaGetLastError(){++clears_;return 0;}
int cudaOccupancyMaxActiveClusters(int* out,void(*fn)(),cudaLaunchConfig_t* c){
 using namespace muse_split_mlp;
 unsigned p=c->gridDim.z;
 if((p!=2 && p!=4) || c->gridDim.x!=8 || c->gridDim.y!=experts_ || c->numAttrs!=1 || c->attrs->id!=4 ||
    c->attrs->val.clusterDim.x!=8 || c->attrs->val.clusterDim.y!=1 || c->attrs->val.clusterDim.z!=1 ||
    c->blockDim.x!=256 || c->blockDim.y!=1 || c->blockDim.z!=1 ||
    c->dynamicSmemBytes!=16*(inner_/(8*p)+8) || fn!=(p==2?split_kernel<2>:split_kernel<4>))return 100;
 tested_|=p;*out=(mask_&p)?1:0;return err_;
}
"""
    tail=r"""
extern "C" int probe(int dev,int h,int i,int e,int cap,int sms,int shared,int mask,int error){
 cap_=cap;sms_=sms;limit_=shared;inner_=i;experts_=e;mask_=mask;err_=error;clears_=tested_=0;
 try{int ret=Probe::supported(dev,h,i,e);return ret+10*tested_+100*clears_;}catch(std::runtime_error&){return -1;}
}
"""
    with tempfile.TemporaryDirectory() as temp:
        d=Path(temp);(d/"f.cpp").write_text(pre+"struct Probe {"+body+"};\n"+tail)
        subprocess.run(["c++","-std=c++20","-O2","-shared","-fPIC",str(d/"f.cpp"),"-o",str(d/"f.so")],check=True)
        f=ctypes.CDLL(str(d/"f.so")).probe;n=0
        for v in product([0,1],[0,256,512,4096],[512,1024,2048,4096,8192],[4,20,32],[0,1],
                         [16,132],[32768,49152],[0,2,4,6],[0,5,7,9,99]):
            dev,h,i,e,cap,sms,limit,mask,err=v
            legal=[p for p in [2,4] if dev==0 and cap and h>0 and h%512==0 and i%(512*p)==0 and
                   e*p*8>=sms and 32768+16*(i//(8*p)+8)<=limit]
            expected=10*sum(legal)+(sum(p for p in legal if mask&p) if err==0 else 100*len(legal))
            if legal and err==99:expected=-1
            assert f(*v)==expected,v
            n+=1
    print(json.dumps(dict(cases=n,both_specializations=True,physical_gpu=False),indent=2))

if __name__=="__main__":main(Path(sys.argv[1]))
