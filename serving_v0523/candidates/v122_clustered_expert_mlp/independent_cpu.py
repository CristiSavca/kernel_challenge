"""Run actual V122 gate/down tile bodies with CPU-emulated warp MMA and cluster barriers."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


v90_path = Path(__file__).resolve().parents[1] / "v90_activated_expert_down/cpu_check.py"
spec = importlib.util.spec_from_file_location("v90", v90_path)
v90 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v90)
v86, v89 = v90.v86, v90.v89

DSMEM = r"""
bf16_t* bands[8];
uint32_t current_rank, phase;
uint64_t remote_reads;
namespace cooperative_groups {
struct Cluster {
  uint32_t block_rank() const {return current_rank;}
  bf16_t* map_shared_rank(bf16_t* ptr, unsigned peer) const {
    if (ptr!=bands[current_rank] || peer>=8 || phase!=1) std::abort();
    remote_reads+=peer!=current_rank;
    return bands[peer];
  }
};
Cluster this_cluster(){return {};}
}
"""

DRIVER = r"""
using namespace sglang::muse_cluster_mlp;
extern "C" void plan(Params* p, uint32_t jobs, uint32_t* out) {
  ScanWarp group;
  std::vector<std::thread> workers;
  for (unsigned lane=0;lane<32;++lane) workers.emplace_back([&,lane] {
    threadIdx.x=lane; scan=&group;
    for(unsigned job=0;job<jobs;++job) {
      Tile t=locate(*p,job);
      if(lane==0){out[3*job]=t.expert;out[3*job+1]=t.first;out[3*job+2]=t.count;}
    }
  });
  for(auto& w:workers)w.join();
}
extern "C" uint64_t simulate(bf16_t* rows, const uint32_t* w13, const uint8_t* s13,
  const uint32_t* w2, const uint8_t* s2, const bf16_t* gs, const int32_t* ends,
  uint32_t h, uint32_t inner, uint32_t experts, uint32_t jobs, uint32_t* tiles) {
  Params p{rows,{w13,s13},{w2,s2},gs,ends,h,inner,experts}; plan(&p,jobs,tiles);
  const uint32_t stride=inner/kCluster+8, slots=kRows*stride, guard=16;
  std::vector<bf16_t> storage[kCluster]; remote_reads=0;
  for(uint32_t b=0;b<kCluster;++b){storage[b].resize(slots+2*guard,0x7fc1);bands[b]=storage[b].data()+guard;}
  for(uint32_t cohort=0;cohort<experts;++cohort) for(uint32_t j=cohort;j<jobs;j+=experts){
    Tile t{tiles[3*j],tiles[3*j+1],tiles[3*j+2]}; if(!t.count) break;
    std::vector<bf16_t> original(rows+t.first*h,rows+(t.first+t.count)*h);
    Routine lanes[kCluster][32*kWarps];
    for(uint32_t b=0;b<kCluster;++b) for(auto& l:lanes[b]) {
      l=t.count<=8 ? project<1>(p,t,bands[b]) : project<2>(p,t,bands[b]);
    }
    for(phase=0;phase<3;++phase){
      for(uint32_t task=0;task<kCluster*kWarps;++task){
        unsigned id=(task*(kCluster*kWarps-1)+j*7+phase*5)%(kCluster*kWarps);
        current_rank=id/kWarps;
        auto w=id%kWarps; auto* warp=lanes[current_rank]+w*32;
        while(true){
          for(unsigned i=0;i<32;++i){threadIdx.x=w*32+i;warp[i].h.resume();}
          bool done=warp[0].h.done();
          for(unsigned i=0;i<32;++i) if(warp[i].h.done()!=done) std::abort();
          if(done){if(phase!=2)std::abort();break;}
          if(!warp[0].h.promise().op.accum){if(phase>=2)std::abort();break;}
          warp_mma(warp);
        }
      }
      if(phase==0 && !std::equal(original.begin(),original.end(),rows+t.first*h))std::abort();
    }
    for(auto& c:lanes)for(auto& l:c)l.h.destroy();
    for(uint32_t b=0;b<kCluster;++b){
      for(unsigned r=0;r<kRows;++r)for(unsigned x=inner/kCluster;x<stride;++x)
        if(bands[b][r*stride+x]!=0x7fc1)std::abort();
      for(unsigned x=0;x<guard;++x)
        if(storage[b][x]!=0x7fc1 || storage[b][guard+slots+x]!=0x7fc1)std::abort();
    }
  }
  return remote_reads;
}
"""


def body(source, marker):
    start = source.index("{", source.index(marker))
    depth = 1
    for i in range(start+1, len(source)):
        depth += (source[i] == "{") - (source[i] == "}")
        if not depth:
            return source[start+1:i]
    raise ValueError(marker)


def compile_actual(stage, root):
    src = stage / "python/sglang/kernels/jit/csrc/muse_spark"
    original = (src / "expert_cluster_mlp.cuh").read_text()
    packed = (src / "expert_skinny_gate_up.cuh").read_text()
    packed = packed[packed.index("SGL_DEVICE uint32_t broadcast_scale"):packed.index("SGL_DEVICE void accumulate")]
    packed, n = re.subn(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',
                         r'\1 = pair_mul(\2, \3);', packed)
    assert n == 2
    pre = v86.PRELUDE.replace("Index threadIdx, blockIdx;", "thread_local Index threadIdx; Index blockIdx;")
    src_body = original.split("namespace sglang {", 1)[1]
    src_body = src_body[:src_body.rindex("template <uint32_t Rows>",0,src_body.index("void contract("))]
    contract = body(original, "void contract(")
    contract, n = re.subn(r'asm volatile\(.*?\)\);',
        'co_yield Op{&d,{a0,a1,a2,a3},{x.low[r],x.high[r]}};', contract, flags=re.S)
    assert n == 1
    def inline_kernel(name):
        src = body(original, f"void {name}(")
        def call(match):
            out, bank, base = match.groups()
            expr = re.sub(r"\bbase\b", "contract_base", contract)
            return "{ auto& out="+out+"; const Bank bank="+bank+"; const size_t contract_base="+base+";"+expr+"}"
        src, n = re.subn(r"contract\((\w+), (p\.\w+), ([^,]+), x\);", call, src)
        assert n == (2 if name=="gate_rows" else 1)
        return "{"+src+"}"
    project = body(original, "void project(")
    for fn in ("gate_rows", "down_rows"):
        call = fn+"<Rows>(p, tile, rank, active);"
        assert project.count(call)==1
        project = project.replace(call, inline_kernel(fn))
    assert project.count("cluster.sync();")==2
    project = project.replace("cluster.sync();", "co_yield Op{};")
    src_body += "template<unsigned Rows> Routine project(const Params& p,Tile tile,bf16_t* active){"+project+"}\n}}\n"
    code = (pre+v89.HOST_MUL+v90.SHUFFLES+DSMEM+"namespace sglang {namespace muse_skinny_gate_up {\n"+
            packed+"}\n"+src_body+v86.POSTLUDE.split('extern "C" void simulate')[0]+DRIVER)
    cpp, out = root/"probe.cpp", root/"probe.dylib"
    cpp.write_text(code)
    subprocess.run(["c++","-std=c++20","-O2","-fno-strict-aliasing","-pthread","-shared","-fPIC",str(cpp),"-o",str(out)],check=True)
    dll = ctypes.CDLL(str(out))
    dll.simulate.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_uint32]*4 + [ctypes.c_void_p]
    dll.simulate.restype = ctypes.c_uint64
    return dll, hashlib.sha256(original.encode()).hexdigest()


def weight_bank(rng, e, n, k):
    codes = rng.integers(0,16,(e,n,k),dtype=np.uint8)
    if e==1 and k>=4096:
        # Sparse numerical values keep the real-shape dense oracle bounded.
        codes[:, :, 16:] = 0
        codes[:, :, -16:] = rng.integers(0,16,(e,n,16),dtype=np.uint8)
    scales = rng.choice(np.array([0,8,16,38,49,54,56,63],np.uint8),size=(e,n,k//16))
    packed = v86.pack_codes(codes)
    stored = scales.transpose(0,2,1).copy().reshape(e,k//16,n//64,8,8).swapaxes(-1,-2).copy()
    return codes, scales, packed, stored


def reference(x,codes,scales):
    lut = np.array([0,.5,1,1.5,2,3,4,6],np.float32)
    result = np.zeros((len(x),codes.shape[0]),dtype=np.float32)
    for k in range(x.shape[1]):
        c = codes[:,k]
        if not np.any(c & 7):
            continue
        g = v86.convert_scales(scales[:,k//16])
        w = np.where(c&8,-lut[c&7],lut[c&7])*g
        result += x[:,k,None] * w[None,:]
    return v86.bf16(result)


def run_case(dll,rng,h,inner,counts,prefix):
    experts = len(counts)
    bounds = np.array([prefix,*(prefix+np.cumsum(counts))],np.int32)
    last = int(bounds[-1]); total = last+2
    gate = weight_bank(rng,experts,2*inner,h)
    down = weight_bank(rng,experts,h,inner)
    rows = v86.bf16(rng.integers(-8,9,(total,h)).astype(np.float32)/16)
    rows[:prefix] = 0x7fc2
    rows[last:] = 0x7fc3
    expected = rows.copy()
    global_scale = v86.bf16(rng.choice(np.array([.125,.5,1,1.5],np.float32),experts))
    for e,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
        if lo==hi: continue
        projected = reference(v86.floats(rows[lo:hi]),gate[0][e],gate[1][e])
        gate_value,up_value = np.split(v86.floats(v86.bf16(v86.floats(projected) * v86.floats(global_scale[e]))),2,axis=1)
        with np.errstate(over="ignore"):
            active = v86.bf16((gate_value/(1+np.exp(-gate_value)))*up_value)
        expected[lo:hi] = reference(v86.floats(active),down[0][e],down[1][e])
    tiles = [(e,j,min(16,int(end)-j)) for e,(start,end) in enumerate(zip(bounds,bounds[1:]))
             for j in range(int(start),int(end),16)]
    jobs = len(tiles)+experts
    actual_tiles = np.full((jobs,3),0xffffffff,np.uint32)
    reads = dll.simulate(*(a.ctypes.data for a in [rows,gate[2],gate[3],down[2],down[3],global_scale,bounds]),
                         h,inner,experts,jobs,actual_tiles.ctypes.data)
    np.testing.assert_array_equal(actual_tiles,np.array(tiles+[(0,0,0)]*experts,np.uint32))
    np.testing.assert_array_equal(rows,expected)
    return {"hidden":h,"intermediate":inner,"count":counts,"untouched_prefix":prefix,
            "checked_live_bf16":sum(counts)*h,"checked_guard_bf16":(prefix+2)*h,
            "tile_jobs":len(tiles),"software_remote_map_calls":reads}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",type=Path);args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix="v122-cpu-") as d:
        dll,h=compile_actual(args.stage,Path(d));rng=np.random.default_rng(122)
        cases=[(512,512,[0,0],3),(512,512,[1,7,8],2),(512,1024,[9,16,17],5),
               (1024,512,[0,35,0],1),(4096,4096,[1],3)]
        out=[run_case(dll,rng,*c) for c in cases]
        print(json.dumps({"header_sha256":h,"cases":out,"tool":"CPU CUDA coroutine/warp MMA emulator",
                          "gpu_timing":None},indent=2))
