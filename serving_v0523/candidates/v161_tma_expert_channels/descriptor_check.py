"""Compile the production CUDA-driver encoder call with a recording CPU driver stub."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile
src=(Path(sys.argv[1])/"python/sglang/kernels/jit/csrc/muse_spark/expert_tma_route_pair.cuh").read_text()
fn=src[src.index("inline bool encode("):src.index("inline void check_map_storage(")]
prog=r"""
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
using cuuint64_t=uint64_t;using cuuint32_t=uint32_t;
constexpr unsigned kKSteps=8;
enum {CUDA_SUCCESS, CU_TENSOR_MAP_DATA_TYPE_UINT32, CU_TENSOR_MAP_DATA_TYPE_UINT16,
 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE};
struct CUtensorMap{void* base; unsigned type; uint64_t dims[2],stride;unsigned box[2];};
namespace tvm::ffi {
struct TensorView {void* base;int64_t dims[3];int rank;bool contig;
 bool IsContiguous()const{return contig;}int ndim()const{return rank;}int64_t size(int i)const{return dims[i];}void*data_ptr()const{return base;}};
}
bool device_online=true;
int emulated_driver(CUtensorMap* o,int dt,int rank,void* addr,const uint64_t* sz,const uint64_t* pitch,const unsigned* box,
 const unsigned* elem,int layout,int swizzle,int l2,int oob){
 assert(rank==2 && elem[0]==1 && elem[1]==1 && layout==CU_TENSOR_MAP_INTERLEAVE_NONE && swizzle==CU_TENSOR_MAP_SWIZZLE_NONE
        && l2==CU_TENSOR_MAP_L2_PROMOTION_NONE && oob==CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 if(*pitch%16 || box[0]>256 || box[1]>256)return 1;
 *o={addr,unsigned(dt),{sz[0],sz[1]},pitch[0],{box[0],box[1]}};return CUDA_SUCCESS;
}
auto encoder(){return device_online?&emulated_driver:nullptr;}
"""+fn+r"""
int main(){
 unsigned tested=0;
 for(auto e:{1,32,64})for(auto k:{128,256,4096,8192})for(auto n:{128,384,4096,8192})
 for(bool weight:{false,true})for(int bad=0;bad<8;bad++){
  tvm::ffi::TensorView b{(void*)0x10000,{e,k/16,weight?2*n:n},3,true};
  device_online=bad!=1;if(bad==2)b.contig=false;if(bad==3)b.rank=2;if(bad==4)b.base=(void*)0x10008;
  if(bad==5)b.dims[0]=0;if(bad==6)b.dims[1]=std::numeric_limits<int32_t>::max();
  if(bad==7)b.dims[2]=3;
  CUtensorMap map{};
  bool ok=encode(b,weight,&map);
  assert(ok==(bad==0 || (bad==6 && e==1)));
  if(ok){
   assert(map.base==b.base && map.type==unsigned(weight?CU_TENSOR_MAP_DATA_TYPE_UINT32:CU_TENSOR_MAP_DATA_TYPE_UINT16));
   assert(map.dims[0]==(weight?2*n:n/2) && map.dims[1]==b.dims[0]*b.dims[1] && map.stride==(weight?8*n:n));
   assert(map.box[0]==(weight?256:64) && map.box[1]==8);
  }
  ++tested;
 }std::printf("%u\n",tested);
}
"""
with tempfile.TemporaryDirectory() as d:
    cpp=Path(d)/"descriptor.cpp";exe=Path(d)/"descriptor";cpp.write_text("#include <initializer_list>\n"+prog)
    subprocess.run(["c++","-std=c++17","-O2",cpp,"-o",exe],check=True)
    tested=int(subprocess.check_output([exe]))
print(json.dumps(dict(native_encoder_sha256=hashlib.sha256(fn.encode()).hexdigest(),cases=tested,
                      cuda_driver="recording CPU stub",hardware=False),indent=2))
