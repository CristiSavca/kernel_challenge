"""Exercise the deployed C++ launch selector via a host-only recording LaunchKernel."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(sys.argv[1])/"python/sglang/kernels/jit/csrc/muse_spark"
base=(root/"expert_route_pair.cuh").read_text()
header=(root/"expert_tma_route_pair.cuh").read_text()
aligned=base[base.index("inline bool aligned_packed_projection"):base.index("template <bool Gate>",base.index("inline bool aligned_packed_projection"))]
launch=header[header.index("template <bool Gate>\ninline void launch_medium"):header.index("}  // namespace muse_tma_route")]
source=r"""
#include <cstdint>
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
struct BankMaps {};
struct DLDevice {};
struct Projection {const void *inputs; struct {const void *weight,*scale;} bank;unsigned output_width;};
struct dim3 {unsigned x,y; dim3(unsigned a,unsigned b):x(a),y(b){}};
constexpr unsigned kWarps=2,kRows=8,kChannelWarps=8;
struct Call {unsigned x,y,threads,kind;bool gate;};
std::vector<Call> calls;
template<bool Gate> void multirow_projection() {}
namespace muse_medium_route_pair { template<bool Gate> void projection(){} }
namespace host {
struct LaunchKernel {
 dim3 grid;unsigned threads;
 LaunchKernel(dim3 d,unsigned t,DLDevice):grid(d),threads(t){}
 void operator()(void(*fn)(),const Projection&,const BankMaps&,unsigned){
  assert(fn==multirow_projection<false> || fn==multirow_projection<true>);
  calls.push_back({grid.x,grid.y,threads,1,fn==multirow_projection<true>});
 }
 void operator()(void(*fn)(),const Projection&,unsigned){
  assert(fn==muse_medium_route_pair::projection<false> || fn==muse_medium_route_pair::projection<true>);
  calls.push_back({grid.x,grid.y,threads,0,fn==muse_medium_route_pair::projection<true>});
 }
};}
"""
source += aligned + launch + r"""
int main(){
 alignas(128) uint8_t storage[160]{};
 unsigned total=0,stream=0;
 for (unsigned a:{0,2,4,8,16})for(unsigned b:{0,4,8,16})for(unsigned c:{0,1,8,16})
 for (unsigned width:{128,256,640,4096,8192})for (unsigned experts:{1,2,32,64})
 for (unsigned stride:{2,8,9,16,17,33,257,2049})for (bool gate:{false,true}){
  calls.clear();Projection p{storage+a,{storage+b,storage+c},width};
  if(gate)launch_medium<true>(p,BankMaps{},experts,stride,{});else launch_medium<false>(p,BankMaps{},experts,stride,{});
  assert(calls.size()==1);auto v=calls.back();
  const bool wanted=!(a%16 || b%16 || c%16) && stride>8;
  assert(v.x==width/128 && v.y==experts && v.threads==(wanted?288:64) && v.kind==unsigned(wanted) && v.gate==gate);
  stream+=wanted;++total;
 }
 std::cout<<"{\"cases\":"<<total<<",\"single_multirow_launch\":"<<stream<<",\"single_vector_fallback\":"<<total-stream<<"}";
}
"""
with tempfile.TemporaryDirectory() as scratch:
    scratch=Path(scratch)
    (scratch/"case.cpp").write_text(source)
    subprocess.run(["c++","-std=c++17","-Wall","-Wextra","-Werror","-O2",scratch/"case.cpp","-o",scratch/"check"],check=True)
    print(json.dumps(json.loads(subprocess.check_output([scratch/"check"])),indent=2))
