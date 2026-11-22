"""Run the production W2 tile and warp prefix scan with CPU CUDA instruction emulators."""

import argparse
import ctypes
import importlib.util
import json
import pathlib
import re
import subprocess
import tempfile

import numpy as np


def load_module(name, relative):
    path = pathlib.Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v86 = load_module('v86', 'v86_skinny_expert_gate_up/cpu_check.py')
v89 = load_module('v89', 'v89_paired_expert_dequant/cpu_check.py')

SHUFFLES = r"""
#include <algorithm>
#include <barrier>
#include <thread>
#include <vector>
using std::min;
struct ScanWarp {
  static constexpr unsigned threads = 32;
  std::barrier<> sync{threads};
  unsigned values[threads]{};
  unsigned shuffle(unsigned x, unsigned src) {
    values[threadIdx.x] = x;
    sync.arrive_and_wait();
    unsigned result = values[src];
    sync.arrive_and_wait();
    return result;
  }
  unsigned ballot(bool x) {
    values[threadIdx.x] = x;
    sync.arrive_and_wait();
    unsigned result=0;
    for (unsigned i=0;i<threads;++i) result |= values[i]<<i;
    sync.arrive_and_wait();
    return result;
  }
};
thread_local ScanWarp* scan;
unsigned __shfl_sync(unsigned, unsigned x, int source) {return scan->shuffle(x,source);}
unsigned __shfl_up_sync(unsigned, unsigned x, unsigned delta) {
 return scan->shuffle(x,threadIdx.x >= delta ? threadIdx.x-delta : threadIdx.x);
}
unsigned __ballot_sync(unsigned, bool pred) {return scan->ballot(pred);}
int __ffs(unsigned value) { return __builtin_ffs(value); }
"""

DRIVER = r"""
using namespace sglang::muse_activated_down;
using Row = uint32_t;
extern "C" void plan(const int32_t* bounds, Row rows, Row experts, Row* result) {
  ScanWarp group;
  Params p{nullptr,nullptr,nullptr,nullptr,nullptr,bounds,0,0,experts};
  Row first = std::min(rows,experts), jobs = first + (rows-first)/kRows;
  std::vector<std::thread> workers;
  for (unsigned lane=0;lane<32;++lane) workers.emplace_back([&,lane] {
    threadIdx.x = lane;
    scan=&group;
    for (Row job=0;job<jobs;++job) {
      Tile t=locate(p,job);
      if (!lane) { result[job*3]=t.expert; result[job*3+1]=t.first; result[job*3+2]=t.rows; }
    }
  });
  for (auto& w:workers) w.join();
}
extern "C" void simulate(const bf16_t* gate, bf16_t* out, const uint32_t* weight,
                         const uint8_t* scales, const bf16_t* globals, const int32_t* offsets,
                         uint32_t inner,uint32_t hidden,uint32_t experts,const uint32_t* tiles,uint32_t jobs) {
  Params p{gate,out,weight,scales,globals,offsets,inner,hidden,experts};
  for (unsigned job=0;job<jobs;++job) {
    Tile t{tiles[job*3],tiles[job*3+1],tiles[job*3+2]};
    if (!t.rows) continue;
    for (blockIdx.x=0;blockIdx.x<hidden/kColumns;++blockIdx.x) {
      alignas(16) bf16_t active[kRows*kStride];
      Routine lanes[32*kWarps];
      for (auto& lane:lanes) lane=select_lane(p,t,active);
      while (!lanes[0].h.done()) {
        for (unsigned w=0;w<kWarps;++w) {
          for (unsigned l=0;l<32;++l) {threadIdx.x=w*32+l; lanes[threadIdx.x].h.resume();}
          if (!lanes[w*32].h.done() && lanes[w*32].h.promise().op.accum) warp_mma(lanes+w*32);
        }
      }
      for (auto& l:lanes) l.h.destroy();
    }
  }
}
"""


def compile_kernel(folder, stage):
    headers=stage/'python/sglang/kernels/jit/csrc/muse_spark'
    packed=(headers/'expert_skinny_gate_up.cuh').read_text()
    packed=packed[packed.index('SGL_DEVICE uint32_t broadcast_scale'):packed.index('SGL_DEVICE void accumulate')]
    packed,count=re.subn(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',
                          r'\1 = pair_mul(\2, \3);',packed)
    assert count==2
    source=(headers/'expert_activated_down.cuh').read_text().split('namespace sglang {',1)[1]
    body=source[:source.index('__global__')]
    body=body.replace('SGL_DEVICE void run_tile(', 'Routine run_tile(')
    body=body.replace('__syncthreads();','co_yield Op{};')
    start=body.index('          asm volatile(')
    end=body.index(';\n',body.index(': "r"(a0)',start))+1
    body=body[:start]+'          co_yield Op{&d,{a0,a1,a2,a3},{b0,b1}};'+body[end:]
    dispatch=source[source.index('  if (tile.rows > 16)'):source.index('\n}',source.index('  if (tile.rows > 16)'))]
    dispatch=dispatch.replace('    run_tile<','    return run_tile<')
    csrc=folder/'sim.cpp'
    csrc.write_text(
        v86.PRELUDE.replace('Index threadIdx, blockIdx;', 'thread_local Index threadIdx; Index blockIdx;')
        +v89.HOST_MUL+SHUFFLES+'namespace sglang { namespace muse_skinny_gate_up {\n'+packed+'}\n'
        +body+'Routine select_lane(const Params& p, Tile tile, bf16_t* active) {\n'+dispatch+'\nstd::abort();\n}\n}}\n'
        +v86.POSTLUDE.split('extern "C" void simulate')[0]+DRIVER
    )
    lib=folder/'sim.so'
    subprocess.run(['c++','-std=c++20','-O2','-fno-strict-aliasing','-pthread','-shared','-fPIC',
                    str(csrc),'-o',str(lib)],check=True)
    dll=ctypes.CDLL(str(lib))
    ptr=ctypes.c_void_p
    u=ctypes.c_uint32
    dll.plan.argtypes=[ptr,u,u,ptr]
    dll.plan.restype=None
    dll.simulate.argtypes=[ptr]*6+[u,u,u,ptr,u]
    dll.simulate.restype=None
    return dll


def plan(dll,counts,prefix):
    bounds=np.array([prefix,*(prefix+np.cumsum(counts))],np.int32)
    rows=int(bounds[-1])
    first=min(rows,len(counts))
    jobs=first+(rows-first)//32
    out=np.full((jobs,3),0xabcdef,np.uint32)
    dll.plan(bounds.ctypes.data,rows,len(counts),out.ctypes.data)
    expected=[(e,r,min(32,end-r)) for e,(lo,end) in enumerate(zip(bounds,bounds[1:])) for r in range(lo,end,32)]
    expected+=[(0,0,0)]*(jobs-len(expected))
    np.testing.assert_array_equal(np.array(expected,dtype=np.uint32).reshape(-1,3),out)
    return out,bounds


def numerical(dll,rng,inner,hidden,counts,prefix):
    tiles,bounds=plan(dll,counts,prefix); experts=len(counts); rows=int(bounds[-1])
    raw=rng.integers(0,16,(experts,hidden,inner),dtype=np.uint8)
    hs=rng.choice(np.array([0,8,16,38,49,54,56,63],np.uint8),size=(experts,hidden,inner//16))
    values=rng.integers(-8,9,(rows,2*inner)).astype(np.float32)/4
    x=v86.bf16(values)
    x[:prefix]=0x7fc1
    global_scales=v86.bf16(rng.choice(np.array([.125,.5,.75,1,1.5],np.float32),experts))
    w=v86.pack_codes(raw)
    s=hs.transpose(0,2,1).copy().reshape(experts,inner//16,hidden//64,8,8).swapaxes(-1,-2).copy()
    out=np.full((rows,hidden),0x7fc2,np.uint16)
    dll.simulate(*(a.ctypes.data for a in [x,out,w,s,global_scales,bounds]),inner,hidden,experts,tiles.ctypes.data,len(tiles))
    assert (out[:prefix]==0x7fc2).all()
    lut=np.array([0,.5,1,1.5,2,3,4,6],np.float32)
    for e,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
        if lo==hi: continue
        gate,up=np.split(v86.floats(v86.bf16(v86.floats(x[lo:hi])*v86.floats(global_scales[e]))),2,axis=1)
        active=v86.floats(v86.bf16((gate/(1+np.exp(-gate)))*up))
        weight=np.where(raw[e]&8,-1,1).astype(np.float32)*lut[raw[e]&7]
        weight*=np.repeat(v86.convert_scales(hs[e]),16,axis=-1)
        reference=np.zeros((hi-lo,hidden),np.float32)
        for k in range(inner):
            reference+=active[:,k,None]*weight[None,:,k]
        np.testing.assert_array_equal(out[lo:hi],v86.bf16(reference))
    return int((rows-prefix)*hidden),int(prefix*hidden)


def main(stage):
    rng=np.random.default_rng(7890)
    live=inactive=plans=0
    with tempfile.TemporaryDirectory() as temp:
        dll=compile_kernel(pathlib.Path(temp),stage)
        for experts in [1,7,31,32,33,55,64]:
            for mode in range(4):
                counts=np.zeros(experts,int)
                if mode==1: counts[-1]=137
                if mode==2: counts[:]=1
                if mode==3: counts=rng.integers(0,41,experts)
                plan(dll,counts,17);plans+=1
        cases=[(128,512,[0,0,0],7),(256,512,[1,0,8],4),(128,1024,[9,16,17],7),
               (256,512,[0,31,32,33],2),(128,512,[0,65,0],3),(4096,512,[0,2],3),
               (128,4096,[0,3],1)]
        for args in cases:
            n,z=numerical(dll,rng,*args);live+=n;inactive+=z
    print(json.dumps(dict(warp_scan_plans=plans+len(cases),numerical_cases=len(cases),
                          live_bf16_values=live,unchanged_poisoned_prefix=inactive),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',type=pathlib.Path)
    main(ap.parse_args().stage)
