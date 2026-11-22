"""Execute production producer/consumer loops as 288 coroutines plus an asynchronous copy engine."""
import ctypes, hashlib, importlib.util, json, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

here=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("v153_warp",here.parent/"v153_cooperative_expert_channels/warp_check.py")
ref=importlib.util.module_from_spec(sp);sp.loader.exec_module(ref)
SHIM=r"""
struct Map { const char* base; uint32_t width,rows,cols,bytes; };
struct BankMaps {Map weight,scale;};
struct MBar {uint32_t parity,initial,remaining,bytes;};
struct Copy {char* dest;const char* source;uint32_t row_bytes,rows,src_pitch;MBar* bar;};
std::vector<Copy> pending;
uint64_t clock_tick=0,waits=0,arrivals=0,tensor_copies=0,row_copies=0,total_bytes=0,skewed_schedules=0;
int policy=0;
Index gridDim;
void init(MBar* bar,unsigned n){*bar={0,n,n,0};}
bool ready(MBar* bar,unsigned phase){if(bar->parity==phase){++waits;return false;}return true;}
void check(MBar* b){if(!b->bytes && !b->remaining){b->parity^=1;b->remaining=b->initial;}}
void arrive(MBar* bar,unsigned bytes=0){if(!bar->remaining)std::abort();--bar->remaining;bar->bytes+=bytes;++arrivals;check(bar);}
void issue_tensor(void* dest,const Map* map,uint32_t x,uint32_t y,MBar* bar){
 if(threadIdx.x%32 || x+map->cols>map->width || y+8>map->rows || uintptr_t(dest)%128)std::abort();
 pending.push_back({(char*)dest,map->base+(y*size_t(map->width)+x)*map->bytes,map->cols*map->bytes,8,map->width*map->bytes,bar});
 ++tensor_copies;
}
void issue_input(void* dest,const void* src,uint32_t bytes,MBar* bar){
 if((uintptr_t(dest)|uintptr_t(src))&15 || bytes!=256 || threadIdx.x%32>=8)std::abort();
 pending.push_back({(char*)dest,(char const*)src,bytes,1,bytes,bar});++row_copies;
}
void move(unsigned i){
 const auto t=pending.at(i);pending.erase(pending.begin()+i);
 for(unsigned r=0;r<t.rows;r++)std::memcpy(t.dest+r*t.row_bytes,t.source+r*t.src_pitch,t.row_bytes);
 const auto bytes=t.row_bytes*t.rows;if(t.bar->bytes<bytes)std::abort();
 t.bar->bytes-=bytes;total_bytes+=bytes;check(t.bar);
}
void advance(){
 ++clock_tick;if(policy==1 && clock_tick%5!=0)return;
 unsigned count=policy==2?std::min(size_t(1),pending.size()):pending.size();
 for(unsigned n=0;n<count;n++)move((clock_tick*7+n)%pending.size());
}
"""

DRIVER=r"""
using namespace sglang::muse_route_pair;
template<bool Gate>
void invoke(const Projection& p,unsigned groups){
 unsigned E=0;
 for(int j=0;j<p.plan[0];j++)E=std::max(E,1u+unsigned(p.plan[1+9*j]));
 const unsigned width=(Gate?2u:1u)*p.output_width;
 BankMaps maps{{(char const*)p.bank.weight,2*width,E*p.input_width/16,256,4},
               {(char const*)p.bank.scale,width/2,E*p.input_width/16,64,2}};
 gridDim={p.output_width/128,groups};
 for(blockIdx.y=0;blockIdx.y<gridDim.y;blockIdx.y++)for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++){
   Routine lanes[288]; for(auto& l:lanes)l=sim<Gate>(p,maps);
   bool done[9]{};uint32_t rounds=0;
   for(;;){
     if(++rounds>100000)std::abort();
     advance();
     bool any=false;
     for(unsigned i=0;i<9;i++){
       const unsigned w=(i+clock_tick)%9;
       if(done[w])continue;
       if(policy==2 && (clock_tick%7)!=(w%7) && clock_tick%4) {++skewed_schedules;any=true;continue;}
       auto* warp=lanes+w*32;
       for(unsigned lane=0;lane<32;lane++){threadIdx.x=w*32+lane;warp[lane].h.resume();}
       done[w]=warp[0].h.done();
       for(unsigned lane=0;lane<32;lane++)if(warp[lane].h.done()!=done[w])std::abort();
       if(!done[w]){
         bool mma=warp[0].h.promise().op.accum!=nullptr;
         for(unsigned lane=0;lane<32;lane++)if((warp[lane].h.promise().op.accum!=nullptr)!=mma)std::abort();
         if(mma)warp_mma(warp);any=true;
       }
     }
     if(!any)break;
   }
   if(!pending.empty())std::abort();
   for(auto& l:lanes)l.h.destroy();
 }
}
extern "C" void simulate(bf16_t* x,const uint32_t* gate,const uint8_t* gs,const bf16_t* gscale,
  const uint32_t* down,const uint8_t* ds,const bf16_t* dscale,bf16_t* act,bf16_t* out,const int32_t* plan,
  uint32_t h,uint32_t inner,uint32_t top,uint32_t groups){
  Projection up{x,act,{gate,gs},gscale,plan,h,inner,top},dn{act,out,{down,ds},dscale,plan,inner,h,1};
  invoke<true>(up,groups);invoke<false>(dn,groups);
}
extern "C" void configure(unsigned p){policy=p;}
extern "C" uint64_t metric(unsigned i){return std::array<uint64_t,6>{waits,arrivals,tensor_copies,row_copies,total_bytes,skewed_schedules}[i];}
"""

def build(stage,tmp):
    root=stage/"python/sglang/kernels/jit/csrc/muse_spark"
    src=(root/"expert_route_pair.cuh").read_text();tma=(root/"expert_tma_route_pair.cuh").read_text()
    pre=src[src.index("namespace sglang {"):src.index("using Words =")]
    pre,n=re.subn(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',
                  r"\1 = pair_mul(\2, \3);",pre);assert n==2
    body=src[src.index("SGL_DEVICE float scaled_bf16"):src.index("struct Slice {")]
    body+=src[src.index("constexpr uint32_t kChannelWarps"):src.index("template <bool Gate>\nstruct alignas(16)")]
    body+=src[src.index("SGL_DEVICE void contract_fragment"):src.index("struct VectorStage")]
    body+=src[src.index("template <bool Gate>\nSGL_DEVICE void finish_channels"):src.index("template <bool Gate, class Stage")]
    body,n=re.subn(r'asm volatile\(\s*"mma.sync.*?\)\);','return Op{&sum,{a0,a1,a2,a3},{lo,hi}};',body,flags=re.S)
    assert n==1;body=body.replace("void contract_fragment(","Op contract_fragment(")
    body+=SHIM+tma[tma.index("template <bool Gate>\nstruct alignas(128) StreamTile"):tma.index("SGL_DEVICE uint32_t shared_addr")]
    body+=tma[tma.index("template <bool Gate>\nSGL_DEVICE void produce("):tma.index("template <bool Gate>\ninline uint32_t resident_stream_blocks")]
    body=body.replace('uint64_t* full','MBar* full').replace('const uint32_t lane = threadIdx.x % 32, bar = shared_addr(full);','const uint32_t lane = threadIdx.x % 32; MBar* bar=full;')
    body,n=re.subn(r'asm volatile\("mbarrier.arrive.expect_tx.*? : "memory"\);','arrive(bar,total);',body,flags=re.S);assert n==1
    body,n=re.subn(r'asm volatile\("cp.async.bulk.shared::cluster.*? : "memory"\);',
                   'issue_input(&tile.inputs[lane][0],src,bytes,bar);',body,flags=re.S);assert n==1
    body=body.replace('  __syncwarp();','')
    body=body.replace('void streamed_projection(', 'Routine sim(').replace('__global__ ','')
    body=body.replace('__shared__ uint64_t full[stages], empty[stages];','__shared__ MBar full[stages], empty[stages];')
    body=body.replace('if (blockIdx.y >= jobs) return;','if (blockIdx.y >= jobs) co_return;')
    body,n=re.subn(r'asm volatile\("mbarrier.init.*? : "memory"\);','init(full+lane,1); init(empty+lane,kChannelWarps);',body,flags=re.S);assert n==1
    # The driver stages CTA initialization first; all warps rendezvous before entering the loop.
    body=body.replace('__syncthreads();','co_yield Op{};')
    body=body.replace('__popc(__ballot_sync(0xffffffffu, slot >= 0))','std::count_if(job+1,job+9,[](int z){return z>=0;})')
    body=body.replace('if (sequence >= stages) wait_phase(empty + index, ((sequence / stages) - 1) & 1);',
                     'while(sequence >= stages && !ready(empty + index,((sequence / stages)-1)&1)) co_yield Op{};')
    body=body.replace('wait_phase(full + index, (sequence / stages) & 1);',
                     'while(!ready(full + index,(sequence / stages)&1)) co_yield Op{};')
    body=body.replace('chunk, slot, live);','chunk, slot, live); co_yield Op{};')
    body=body.replace('contract_fragment(gate,','co_yield contract_fragment(gate,').replace('contract_fragment(up,','co_yield contract_fragment(up,')
    body,n=re.subn(r'asm volatile\("mbarrier.arrive.shared::cta.*? : "memory"\);','arrive(empty+index);',body);assert n==1
    assert 'asm volatile' not in body and 'wait_phase(' not in body
    start=ref.parent.v86.PRELUDE+'\n#include <algorithm>\n#include <vector>\n#include <array>\n#include <cstring>\n'
    code=start+ref.parent.base.v89.HOST_MUL+pre+body+'\n}}\n'+ref.parent.v86.POSTLUDE.split('extern "C" void simulate')[0]+DRIVER
    # The synthetic coroutine scheduler needs one uniform first tick for block arrival.
    code=code.replace('if(policy==2 && (clock_tick%7)', 'if(rounds>1 && policy==2 && (clock_tick%7)')
    cpp=tmp/'sim.cpp';out=tmp/'sim.so';cpp.write_text(code)
    subprocess.run(['c++','-std=c++20','-O2','-shared','-fPIC','-fno-strict-aliasing',cpp,'-o',out],check=True)
    dll=ctypes.CDLL(str(out));dll.simulate.argtypes=[ctypes.c_void_p]*10+[ctypes.c_uint32]*4
    dll.simulate.restype=None;dll.metric.argtypes=[ctypes.c_uint32];dll.metric.restype=ctypes.c_uint64
    return dll,[hashlib.sha256(x.encode()).hexdigest() for x in (src,tma)]

if __name__=="__main__":
    rng=np.random.default_rng(165)
    fixtures=[(128,128,2,np.full((8,3),-1),3),(256,128,4,rng.integers(-1,5,(11,3)),1),
              (128,512,3,np.zeros((19,2),np.int32),2),(512,256,3,rng.integers(-2,4,(8,8)),2),
              (384,640,3,np.array([0]*8+[1]*16+[2]*17+[-1]*4).reshape(9,5),3),
              (4096,4096,1,np.array([[-1,0],[-1,-1]]),1)]
    with tempfile.TemporaryDirectory() as tmp:
        dll,digests=build(Path(sys.argv[1]),Path(tmp))
        cases=[];sensitive=[]
        for schedule in (0,1,2):
            dll.configure(schedule)
            for v in fixtures:
                cases.append(ref.parent.run_case(dll,rng,*v))
            sensitive.extend(ref.sensitive.check(dll,s) for s in (2,4))
        stats=[dll.metric(i) for i in range(6)]
    print(json.dumps(dict(source_sha256=digests,schedules=3,cases=cases,ascending_k=sensitive,
        counts=dict(zip(["wait_polls","arrival_instructions","tensor_copies","activation_rows","copied_bytes","deferred_warps"],
                        stats)),hardware=False),indent=2))
