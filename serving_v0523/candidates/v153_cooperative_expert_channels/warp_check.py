"""Execute the production cooperative copy, warp contractions, barriers and stores on the CPU."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

PARENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENTS / "v129_routed_projection_pair"))
import warp_check as parent
import order_check as sensitive

COPY = r"""
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>
struct Pending { void* dest; std::array<unsigned char,16> bytes; };
std::vector<Pending> waiting[256];
unsigned epoch = 0;
uint64_t copied = 0, zeroed = 0, rendezvous = 0;
void copy16(void* dest, const void* src, unsigned valid) {
  if (reinterpret_cast<uintptr_t>(dest)%16 || reinterpret_cast<uintptr_t>(src)%16) std::abort();
  Pending p{dest,{}};
  if (valid==16) {std::memcpy(p.bytes.data(),src,16); ++copied;}
  else {if(valid) std::abort(); ++zeroed;}
  if ((threadIdx.x+epoch)%2) std::memcpy(dest,p.bytes.data(),16);
  else waiting[threadIdx.x].push_back(p);
}
void flush(unsigned lane){
  for(const auto& p:waiting[lane]) std::memcpy(p.dest,p.bytes.data(),16);
  waiting[lane].clear();
}
"""

DRIVER = r"""
using namespace sglang::muse_route_pair;
template<bool Gate>
void invoke(const Projection& p, unsigned groups) {
  for(unsigned group=0;group<groups;++group) for(unsigned col=0;col<p.output_width;col+=128){
    PackedTile<Gate> buffers[2]; std::memset(buffers,0xa5,sizeof(buffers));
    for(unsigned j=group;j<static_cast<unsigned>(p.plan[0]);j+=groups){
      Routine lanes[256];
      for(auto& lane:lanes) lane=sim<Gate>(p,p.plan+1+9*j,col,buffers);
      for(;;){
        unsigned finished=0,barriers=0;
        for(unsigned iter=0;iter<8;iter++){
          const unsigned w=(3*iter+epoch)%8;
          while(true){
            for(unsigned i=0;i<32;i++){threadIdx.x=32*w+i; lanes[32*w+i].h.resume();}
            bool done=lanes[32*w].h.done();
            for(unsigned i=0;i<32;i++) if(lanes[32*w+i].h.done()!=done) std::abort();
            if(done){++finished;break;}
            bool mma=lanes[32*w].h.promise().op.accum!=nullptr;
            for(unsigned i=0;i<32;i++) if((lanes[32*w+i].h.promise().op.accum!=nullptr)!=mma)std::abort();
            if(mma)warp_mma(lanes+32*w);
            else{for(unsigned i=0;i<32;i++)flush(32*w+i);++barriers;break;}
          }
        }
        if(finished==8)break;
        if(barriers!=8||finished)std::abort();
        ++rendezvous; ++epoch;
      }
      for(auto& l:lanes) l.h.destroy();
    }
  }
}
extern "C" void simulate(bf16_t* x,const uint32_t* gate,const uint8_t* gs,const bf16_t* gscale,
  const uint32_t* down,const uint8_t* ds,const bf16_t* dscale,bf16_t* act,bf16_t* out,const int32_t* plan,
  uint32_t h,uint32_t inner,uint32_t top,uint32_t groups){
  Projection up{x,act,{gate,gs},gscale,plan,h,inner,top};
  Projection dn{act,out,{down,ds},dscale,plan,inner,h,1};
  invoke<true>(up,groups);invoke<false>(dn,groups);
}
extern "C" uint64_t count(unsigned i){return i==0?copied:i==1?zeroed:rendezvous;}
"""


def compile_device(stage, path):
    header=stage/"python/sglang/kernels/jit/csrc/muse_spark/expert_route_pair.cuh"
    src=header.read_text()
    pre=src[src.index("namespace sglang {"):src.index("using Words =")]
    pre,n=re.subn(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',
                   r"\1 = pair_mul(\2, \3);",pre)
    assert n==2
    body=src[src.index("SGL_DEVICE float scaled_bf16"):src.index("struct Slice {")]
    body+=src[src.index("constexpr uint32_t kChannelWarps"):src.index("template <bool Gate>\n__global__ __launch_bounds__(32 * kChannelWarps)")]
    body,n=re.subn(r"  const uint32_t shared = .*?\n  asm volatile\([^\n]+;", "  copy16(dest,src,16);",body,flags=re.S)
    assert n==1
    body,n=re.subn(r"    const uint32_t address = .*? : \"memory\"\);",
                   "    copy16(&dest.inputs[step][route][4 * half],input + 8 * lane,valid_bytes);",body,flags=re.S)
    assert n==1
    body=body.replace('asm volatile("cp.async.commit_group;" ::: "memory");','')
    body=re.sub(r"SGL_DEVICE void publish_packed\(\).*?\n}\n","",body,flags=re.S)
    body=body.replace("void contract_fragment(","Op contract_fragment(")
    body,n=re.subn(r'asm volatile\(\s*"mma.sync.*?\)\);','return Op{&sum,{a0,a1,a2,a3},{lo,hi}};',body,flags=re.S)
    assert n==1 and "asm volatile" not in body
    body=body.replace("void project_channels(","Routine sim(").replace("publish_packed();","co_yield Op{};")
    body=body.replace("contract_fragment(gate,","co_yield contract_fragment(gate,")
    body=body.replace("contract_fragment(up,","co_yield contract_fragment(up,")
    code=parent.v86.PRELUDE+COPY+parent.base.v89.HOST_MUL+pre+body+"\n}}\n"
    code+=parent.v86.POSTLUDE.split('extern "C" void simulate')[0]+DRIVER
    cpp=path/"warp.cpp";lib=path/"warp.so";cpp.write_text(code)
    subprocess.run(["c++","-std=c++20","-fPIC","-shared","-fno-strict-aliasing","-O2",cpp,"-o",lib],check=True)
    dll=ctypes.CDLL(str(lib));dll.simulate.argtypes=[ctypes.c_void_p]*10+[ctypes.c_uint32]*4
    dll.simulate.restype=None;dll.count.argtypes=[ctypes.c_uint32];dll.count.restype=ctypes.c_uint64
    return dll,hashlib.sha256(src.encode()).hexdigest()


if __name__=="__main__":
    rng=np.random.default_rng(153)
    fixtures=[(128,128,2,np.full((8,3),-1),3),(256,128,4,rng.integers(-1,5,(11,3)),1),
              (128,512,3,np.zeros((19,2),np.int32),2),(512,256,3,rng.integers(-2,4,(8,8)),2),
              (384,640,3,np.array([0]*8+[1]*16+[2]*17+[-1]*4).reshape(9,5),3),
              (4096,4096,1,np.array([[-1,0],[-1,-1]]),1)]
    with tempfile.TemporaryDirectory() as d:
        dll,digest=compile_device(Path(sys.argv[1]),Path(d))
        cases=[parent.run_case(dll,rng,*f) for f in fixtures]
        order=[sensitive.check(dll,split) for split in (2,4)]
        counters=[dll.count(i) for i in range(3)]
    print(json.dumps(dict(header_sha256=digest,cases=cases,ascending_k_cancellation=order,
          valid_16byte_copies=counters[0],zero_filled_vectors=counters[1],cta_rendezvous=counters[2],
          mixed_immediate_and_delayed_copies=True,hardware=False),indent=2))
