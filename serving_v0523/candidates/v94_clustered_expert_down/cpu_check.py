"""Emulate actual clustered CUDA tile publication and run asynchronous warp schedules."""
import argparse
import ctypes
import importlib.util
import json
import pathlib
import re
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np

base=pathlib.Path(__file__).resolve().parents[1]/'v90_activated_expert_down'/'cpu_check.py'
spec=importlib.util.spec_from_file_location('v90',base)
v90=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v90)
v86,v89=v90.v86,v90.v89

CLUSTER = r"""
unsigned sim_rank;
bf16_t* banks[8];
uint64_t published=0;
namespace cooperative_groups {
struct Cluster {
  unsigned block_rank() const { return sim_rank; }
  bf16_t* map_shared_rank(bf16_t* dest, unsigned rank) const {
    auto index=dest-banks[sim_rank];
    if (index<0 || index+4>32*136 || index%4) std::abort();
    ++published; return banks[rank]+index;
  }
  void sync() {}
};
Cluster this_cluster() {return {};}
}
"""

DRIVER = r"""
template <unsigned C>
void launch(const Params& p, Tile tile, unsigned first_band) {
  alignas(16) bf16_t shared[C][kRows*kStride];
  Routine lanes[C][kWarps*32];
  for (unsigned b=0;b<C;++b) {
    banks[b]=shared[b];
    std::fill_n(banks[b],kRows*kStride,static_cast<bf16_t>(0x7fff));
    for(auto& l:lanes[b]) l=select_lane<C>(p,tile,banks[b]);
  }
  // Let a warp run ahead of peers up to a barrier, varying the order each
  // phase. A barrier waits for all warps in every block of the same cluster.
  unsigned phase=0;
  while(!lanes[0][0].h.done()) {
    for (unsigned task=0;task<C*kWarps;++task) {
      unsigned index = (task*(C*kWarps-1)+phase*3)%(C*kWarps);
      unsigned b=index/kWarps,w=index%kWarps;
      sim_rank=b; blockIdx.x=first_band+b;
      auto* warp=&lanes[b][w*32];
      for (;;) {
        for (unsigned i=0;i<32;++i) { threadIdx.x=w*32+i; warp[i].h.resume(); }
        if (warp[0].h.done() || !warp[0].h.promise().op.accum) break;
        warp_mma(warp);
      }
    }
    ++phase;
  }
  for (auto& block:lanes) for (auto& l:block) l.h.destroy();
}
extern "C" void simulate(const bf16_t* gate, bf16_t* out, const uint32_t* weight,
                         const uint8_t* scales, const bf16_t* globals, const int32_t* ends,
                         uint32_t inner,uint32_t hidden,uint32_t experts,const uint32_t* tiles,
                         uint32_t jobs,unsigned cluster) {
  Params p{gate,out,weight,scales,globals,ends,inner,hidden,experts};
  activations=published=0;
  for (unsigned j=0;j<jobs;++j) {
    Tile tile{tiles[j*3],tiles[j*3+1],tiles[j*3+2]};
    if (!tile.rows) continue;
    for (unsigned x=0;x<hidden/kColumns;x+=cluster) {
      switch (cluster) {
        case 1: launch<1>(p,tile,x); break;
        case 2: launch<2>(p,tile,x); break;
        case 4: launch<4>(p,tile,x); break;
        case 8: launch<8>(p,tile,x); break;
        default: std::abort();
      }
    }
  }
}
extern "C" uint64_t count_activations() {return activations/2;}
extern "C" uint64_t count_stores() {return published;}
"""


def compile_kernel(folder,stage):
    head=stage/'python/sglang/kernels/jit/csrc/muse_spark'
    packed=(head/'expert_skinny_gate_up.cuh').read_text()
    packed=packed[packed.index('SGL_DEVICE uint32_t broadcast_scale'):packed.index('SGL_DEVICE void accumulate')]
    packed,num=re.subn(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',
                         r'\1 = pair_mul(\2, \3);',packed)
    assert num==2
    full=(head/'expert_activated_down.cuh').read_text().split('namespace sglang {',1)[1]
    body=full[:full.rindex('template <uint32_t Cluster>',0,full.index('__global__'))]
    body=body.replace('SGL_DEVICE void run_tile(', 'Routine run_tile(')
    a=body.index('          asm volatile('); b=body.index(';\n',body.index(': "r"(a0)',a))+1
    body=body[:a]+'          co_yield Op{&d,{a0,a1,a2,a3},{b0,b1}};'+body[b:]
    body=body.replace('slab_barrier<Cluster>();','co_yield Op{};').replace('__syncthreads();','return;')
    dispatch=full[full.index('  if (tile.rows > 16)'):full.index('\n}',full.index('  if (tile.rows > 16)'))]
    dispatch=dispatch.replace('    run_tile<','    return run_tile<')
    pre=v86.PRELUDE.replace('Index threadIdx, blockIdx;','thread_local Index threadIdx; Index blockIdx;')
    pre=pre.replace('float scale_and_round(bf16_t x, float g) {','float scale_and_round(bf16_t x, float g) {++::activations;')
    pre=pre.replace('namespace muse_expert_scaled_silu {','uint64_t activations=0;\nnamespace muse_expert_scaled_silu {')
    pre=pre.replace('  T operator[](int i) const', '''  void load(const T* p) {for (int i=0;i<N;++i) values[i]=p[i];}
  void store(T* p) const {for(int i=0;i<N;++i) p[i]=values[i];}
  void fill(T x) {for(int i=0;i<N;++i) values[i]=x;}
  T& operator[](int i) {return values[i];}
  T operator[](int i) const''')
    pre=pre+v89.HOST_MUL+v90.SHUFFLES+CLUSTER
    code=(pre+'namespace sglang { namespace muse_skinny_gate_up {\n'+packed+'}\n'+body+
          'template <unsigned Cluster> Routine select_lane(const Params& p,Tile tile,bf16_t* active) {\n'+
          dispatch+'\nstd::abort();\n}\n}}\n'+v86.POSTLUDE.split('extern "C" void simulate')[0]+
          v90.DRIVER.split('extern "C" void simulate')[0]+DRIVER)
    src=folder/'sim.cpp';src.write_text(code);out=folder/'sim.so'
    subprocess.run(['c++','-std=c++20','-O2','-fno-strict-aliasing','-pthread','-shared','-fPIC',
                    str(src),'-o',str(out)],check=True)
    dll=ctypes.CDLL(str(out)); ptr=ctypes.c_void_p; u=ctypes.c_uint32
    dll.plan.argtypes=[ptr,u,u,ptr]; dll.plan.restype=None
    dll.simulate.argtypes=[ptr]*6+[u,u,u,ptr,u,u]; dll.simulate.restype=None
    dll.count_activations.restype=dll.count_stores.restype=ctypes.c_uint64
    return dll


def main(stage):
    rng=np.random.default_rng(3894)
    live=inactive=calls=0
    counts_by_geometry=[]
    with tempfile.TemporaryDirectory() as scratch:
        lib=compile_kernel(pathlib.Path(scratch),stage)
        for cluster in [1,2,4,8]:
            cases=[(128,512*cluster,[0,0],3),(256,512*cluster,[0,1,9],5),
                   (128,512*cluster,[0,17,33],2),(128,4096,[1,0,0,8],1)]
            if cluster==8: cases.append((4096,4096,[1],2))
            driver=SimpleNamespace(plan=lib.plan,simulate=lambda *args:lib.simulate(*args,cluster))
            for inner,hidden,counts,prefix in cases:
                n,z=v90.numerical(driver,rng,inner,hidden,counts,prefix)
                live+=n;inactive+=z;calls+=1
                expected=sum(counts)*inner*(hidden//512//cluster)
                assert lib.count_activations()==expected
                padded=sum((min(32,v-offset)+7)//8 if min(32,v-offset)<=16 else 4
                           for v in counts for offset in range(0,v,32))*8
                assert lib.count_stores()==(0 if cluster==1 else padded*inner*(hidden//512)//4)
                counts_by_geometry.append([cluster,hidden,inner,sum(counts),expected])
    print(json.dumps(dict(cta_cluster_cases=calls,live_bf16_values=live,untouched_prefix=inactive,
                          exact_activation_counts=counts_by_geometry),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',type=pathlib.Path)
    main(ap.parse_args().stage)
