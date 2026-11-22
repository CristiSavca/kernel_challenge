"""Compile both actual native launch decisions with a recording host launcher."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]) / "python/sglang/kernels/jit/csrc/muse_spark/expert_route_pair.cuh"
header = source.read_text()
start = header.index("inline bool aligned_packed_projection(")
stop = header.index("}  // namespace muse_route_pair", start)
host = header[start:stop]
assert all(f"launch_projection<{b}>({t}, uint32_t(groups), dev.unwrap());" in header
           for b,t in [("true","up"),("false","out")])
program = r"""
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <initializer_list>
struct DLDevice { int idx; };
struct dim3 { unsigned x,y,z; dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct Bank {const uint32_t* weight;const uint8_t* scale;};
struct Projection {const uint16_t* inputs; Bank bank; uint32_t output_width;};
constexpr unsigned kWarps=2,kChannelWarps=8;
unsigned threads,phase; dim3 grid(0); DLDevice dev{};
template<bool Gate>void projection_channels(Projection) {phase=Gate?1:2;}
template<bool Gate>void projection_kernel(Projection) {phase=Gate?3:4;}
namespace host {struct LaunchKernel {
 LaunchKernel(dim3 g,unsigned n,DLDevice d){grid=g;threads=n;dev=d;}
 void operator()(void(*fn)(Projection),Projection p){fn(p);}
};}
""" + host + r"""
int main(){
 unsigned count=0,coop=0,otherwise=0;
 for(unsigned w:{128u,512u,4096u,8192u})for(unsigned groups:{1u,7u,8u,32u})
 for(unsigned x:{0u,4u,8u,12u})for(unsigned weight:{0u,8u})for(unsigned scale:{0u,8u})
 for(unsigned mode:{0u,1u}){
  Projection p{reinterpret_cast<const uint16_t*>(0x1000+x),
      {reinterpret_cast<const uint32_t*>(0x2000+weight),reinterpret_cast<const uint8_t*>(0x3000+scale)},w};
  if(mode)launch_projection<true>(p,groups,{7});else launch_projection<false>(p,groups,{7});
  bool expected=!(x||weight||scale);
  assert(grid.x==w/128 && grid.y==groups && grid.z==1 && dev.idx==7);
  assert(threads==(expected?256:64) && phase==(expected?(mode?1:2):(mode?3:4)));
  ++count;coop+=expected;otherwise+=!expected;
 }
 std::printf("%u %u %u\n",count,coop,otherwise);
}
"""
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/"dispatch.cpp";exe=Path(tmp)/"dispatch";path.write_text(program)
    subprocess.run(["c++","-std=c++20","-O2",path,"-o",exe],check=True)
    cases,cooperative,fallback=map(int,subprocess.check_output([exe]).split())
print(json.dumps(dict(header_sha256=hashlib.sha256(header.encode()).hexdigest(),cases=cases,
      cooperative=cooperative,parent_fallback=fallback,native_ffi_call_sites=2),indent=2))
