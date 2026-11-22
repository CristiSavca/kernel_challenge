"""Execute the new compact-projection body lane by lane against V129's independent dense oracle."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

parent_dir = Path(__file__).resolve().parents[1] / "v129_routed_projection_pair"
spec = importlib.util.spec_from_file_location("old_warp",parent_dir / "warp_check.py")
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
base = parent.base

DRIVER = r"""
using namespace sglang::muse_route_pair;
using namespace sglang::muse_medium_route_pair;
template <bool Gate, unsigned Groups>
void launch_warp(const Projection& p,unsigned expert,const int32_t* slots,unsigned live,unsigned col) {
    Routine lanes[32];
    for (auto& lane:lanes) lane=sim<Gate,Groups>(p,expert,slots,live,col);
    for (;;) {
      for(unsigned n=0;n<32;++n){threadIdx.x=n;lanes[n].h.resume();}
      bool done=lanes[0].h.done();for(auto& lane:lanes)if(lane.h.done()!=done)std::abort();
      if(done)break;warp_mma(lanes);
    }
    for(auto& lane:lanes)lane.h.destroy();
}
extern "C" void simulate(const bf16_t* x,const uint32_t* gate,const uint8_t* gs,const bf16_t* gscale,
                         const uint32_t* down,const uint8_t* ds,const bf16_t* dscale,bf16_t* act,bf16_t* out,
                         const int32_t* boxes,uint32_t h,uint32_t inner,uint32_t top,uint32_t unused) {
  const uint32_t e=boxes[1],stride=boxes[2]+1;const int32_t* plan=boxes+3;
  Projection up{x,act,{gate,gs},gscale,plan,h,inner,top};
  Projection dn{act,out,{down,ds},dscale,plan,inner,h,1};
  for(unsigned phase=0;phase<2;++phase){
    const Projection& p=phase?dn:up;
    for(blockIdx.y=0;blockIdx.y<e;++blockIdx.y){
      for(unsigned band=0;band<p.output_width;band+=64){
        blockIdx.x=band/128;threadIdx.x=(band%128)/2;
        if(phase) run_grid<false>(p,stride);else run_grid<true>(p,stride);
      }
    }
  }
}
"""


def compile_kernel(stage, tmp):
    directory=stage/"python/sglang/kernels/jit/csrc/muse_spark"
    medium=(directory/"expert_medium_route_pair.cuh").read_text()
    orig=(directory/"expert_route_pair.cuh").read_text()
    pre=orig[orig.index("namespace sglang {"):orig.index("SGL_DEVICE void contract(")]
    pre,num=re.subn(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',r"\1 = pair_mul(\2, \3);",pre)
    assert num==2
    types=orig[orig.index("SGL_DEVICE float scaled_bf16"):orig.index("template <bool Gate>")]
    contract=base.body(medium,"void contract_rows(")
    contract,num=re.subn(r"asm volatile\(.*?\)\);","co_yield Op{&d,{a0,a1,a2,a3},{x[g].lo,x[g].hi}};",contract,flags=re.S)
    assert num==1
    body=base.body(medium,"void project_rows(")
    def replace(call):
        accum,addr=call.groups()
        c=re.sub(r"\bbase\b","address",contract)
        return "{auto& out="+accum+"; const Bank bank=p.bank; size_t address="+addr+";"+c+"}"
    body,num=re.subn(r"contract_rows\((\w+), p.bank, ([^,]+), x\);",replace,body)
    assert num==2
    fn="template<bool Gate,unsigned Groups> Routine sim(const Projection& p,unsigned expert,const int32_t* slots,unsigned live,unsigned column){"+body+"}"
    grid=base.body(medium,"void projection(").replace("project_rows<","launch_warp<").replace("min(tile_rows, rows - first)","std::min(tile_rows, rows - first)")
    gridfn="template<bool Gate> void run_grid(const Projection& p,unsigned stride){"+grid+"}"
    lead="namespace muse_medium_route_pair {using namespace muse_route_pair;\n"
    inp=medium[medium.index("struct Inputs {"):medium.index("template <unsigned Groups>")]
    code="#include <algorithm>\n"+parent.v86.PRELUDE+base.v89.HOST_MUL+pre+types+"}\n"+lead+inp+fn+"\n}}\n"
    code+=parent.v86.POSTLUDE.split('extern "C" void simulate')[0]+DRIVER.replace('extern "C" void simulate',gridfn+'\nextern "C" void simulate')
    cpp=tmp/"warp.cc";lib=tmp/"warp.so";cpp.write_text(code)
    subprocess.run(["c++","-std=c++20","-O2","-fno-strict-aliasing","-fPIC","-shared",str(cpp),"-o",str(lib)],check=True)
    dll=ctypes.CDLL(str(lib));dll.simulate.argtypes=[ctypes.c_void_p]*10+[ctypes.c_uint32]*4;dll.simulate.restype=None
    return dll,hashlib.sha256(medium.encode()).hexdigest()


def plan(ids,experts,rng):
    flat=ids.reshape(-1);data=np.full((experts,flat.size+1),0x3fffffff,np.int32)
    jobs=0
    for e in range(experts):
        loc=np.flatnonzero(flat==e)
        data[e,0]=len(loc);data[e,1:len(loc)+1]=loc
        jobs+=(len(loc)+15)//16+(len(loc)+31)//32
    return np.concatenate((np.array([jobs,experts,flat.size],np.int32),data.ravel()))


if __name__=="__main__":
    rng=np.random.default_rng(141)
    cases=[(128,256,3,np.full((9,3),-1)),(256,128,4,rng.integers(-1,5,(26,8))),
           (128,128,5,np.zeros((48,3),dtype=np.int32)),(128,256,3,np.array([0]*8+[1]*16+[2]*17+[-1]*4).reshape(9,5)),
           (128,128,4,np.array([0]*9+[1]*32+[2]*33+[3]*31+[-1]*7).reshape(14,8)),
           (4096,4096,1,np.array([[0,-1],[-1,-1]]))]
    with tempfile.TemporaryDirectory() as td, patch.object(parent,"plan_jobs",plan):
        dll,sha=compile_kernel(Path(sys.argv[1]),Path(td))
        results=[parent.run_case(dll,rng,*case,1) for case in cases]
    print(json.dumps(dict(header_sha256=sha,production_projection_with_emulated_mma=True,cases=results),indent=2))
