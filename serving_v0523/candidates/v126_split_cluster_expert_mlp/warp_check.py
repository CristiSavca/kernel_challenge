"""Execute the production split projection with a CPU coroutine for each CUDA lane."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"v122_clustered_expert_mlp"))
import independent_cpu as base
v86=base.v86

DRIVER=r"""
using namespace sglang::muse_split_mlp;
extern "C" void run_split(const uint16_t* rows,const uint32_t* w13,const uint8_t* gs,const uint32_t* w2,
    const uint8_t* ds,const uint16_t* global,const int32_t* offsets,uint32_t h,uint32_t inner,uint32_t experts,
    uint32_t limit,uint32_t splits,uint32_t groups,float* output) {
  Params p{const_cast<bf16_t*>(rows),{w13,gs},{w2,ds},global,offsets,h,inner,experts};
  const uint32_t tiles=experts+limit/16+1;
  std::vector<uint32_t> coords(3*tiles); plan(&p,tiles,coords.data());
  const uint32_t width=inner/(8*splits), shared=8*(width+8),guard=16;
  std::vector<bf16_t> store[8];
  for(unsigned b=0;b<8;++b) {store[b].resize(guard*2+shared,0x7fd1);bands[b]=store[b].data()+guard;}
  Stripes stripes[8];
  for(unsigned split=0;split<splits;++split) for(unsigned cohort=0;cohort<groups;++cohort)
    for(unsigned j=cohort;j<tiles;j+=groups) {
      unsigned count=coords[j*3+2]; if (!count) break;
      for(unsigned i=0;i<count;i+=8) {
        Tile t{coords[3*j],coords[3*j+1]+i,std::min(8u,count-i)};
        for(unsigned b=0;b<8;++b)std::fill_n(bands[b],shared,0x7fd2);
        for(phase=0;phase<2;++phase) {
          Routine lanes[8][256];
          for(unsigned b=0;b<8;++b) for(auto& l:lanes[b]){
            float* dst=output+size_t(split)*limit*h;
            if(phase==0) l=splits==2 ? sim_gate<2>(p,t,split,b,stripes[b],bands[b]) : sim_gate<4>(p,t,split,b,stripes[b],bands[b]);
            else l=splits==2 ? sim_down<2>(p,t,split,b,bands[b],dst) : sim_down<4>(p,t,split,b,bands[b],dst);
          }
          unsigned pass=0;
          for(;;++pass) {
            unsigned finished=0;
            for(unsigned idx=0;idx<64;++idx) {
              unsigned id=(idx*63+j*7+phase*5+pass*9)%64, w=id%8;
              current_rank=id/8;auto* warp=lanes[current_rank]+32*w;
              while(true){
                for(unsigned n=0;n<32;++n){threadIdx.x=32*w+n;warp[n].h.resume();}
                bool done=warp[0].h.done();
                for(unsigned n=0;n<32;++n)if(warp[n].h.done()!=done)std::abort();
                if(done){++finished;break;}
                if(!warp[0].h.promise().op.accum)break;
                warp_mma(warp);
              }
            }
            if(finished){if(finished!=64)std::abort();break;}
          }
          for(auto& c:lanes)for(auto& l:c)l.h.destroy();
        }
        for(auto& buf:store)for(unsigned n=0;n<guard;++n)
          if(buf[n]!=0x7fd1 || buf[guard+shared+n]!=0x7fd1)std::abort();
      }
    }
}
"""

def compile_actual(stage,root):
    folder=stage/"python/sglang/kernels/jit/csrc/muse_spark"
    common=(folder/"expert_cluster_mlp.cuh").read_text()
    split=(folder/"expert_split_mlp.cuh").read_text()
    common=common.split("namespace sglang {",1)[1]
    front=common[:common.rindex("template <uint32_t Rows>",0,common.index("void contract("))]
    front,n=re.subn(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',r'\1=pair_mul(\2,\3);',front)
    assert n==2
    contract=base.body(common,"void contract(").replace("Rows","1")
    contract,n=re.subn(r'asm volatile\(.*?\)\);','co_yield Op{&d,{a0,a1,a2,a3},{x.low[r],x.high[r]}};',contract,flags=re.S)
    assert n==1
    routines=[]
    for name,args in [("gate_split","Stripes& partial, bf16_t* active"),("down_split","bf16_t* active,float* partial_rows")]:
        body=base.body(split,f"void {name}(")
        def inline(m):
            out,bank,addr=m.groups()
            expr=re.sub(r'\bbase\b','contract_base',contract)
            return f"{{auto& out={out}; const Bank bank={bank}; size_t contract_base={addr};"+expr+"}"
        body,n=re.subn(r'contract\((\w+), (p\.\w+), ([^,]+), x\);',inline,body)
        assert n==(2 if name=="gate_split" else 1)
        body=body.replace("__syncthreads();","co_yield Op{};")
        routines.append(f"template<unsigned Splits> Routine sim_{name.split('_')[0]}(const Params& p,Tile tile,unsigned split,unsigned rank,{args}){{"+body+"}")
    pre=v86.PRELUDE.replace("Index threadIdx, blockIdx;","thread_local Index threadIdx; Index blockIdx;")
    stripes=next(x for x in split.splitlines() if x.startswith('using Stripes'))
    plan=base.DRIVER[:base.DRIVER.index('extern "C" uint64_t simulate')]
    source=pre+base.v89.HOST_MUL+base.v90.SHUFFLES+base.DSMEM+"namespace sglang {"+front+"}\nnamespace muse_split_mlp {using namespace muse_cluster_mlp;\n"+stripes+"\n"+"\n".join(routines)+"\n}}\n"+v86.POSTLUDE.split('extern "C" void simulate')[0]+plan+DRIVER
    cpp,so=root/"run.cpp",root/"run.so";cpp.write_text(source)
    subprocess.run(["c++","-std=c++20","-O2","-fno-strict-aliasing","-pthread","-shared","-fPIC",str(cpp),"-o",str(so)],check=True)
    dll=ctypes.CDLL(str(so));dll.run_split.argtypes=[ctypes.c_void_p]*7+[ctypes.c_uint32]*6+[ctypes.c_void_p];dll.run_split.restype=None
    return dll,hashlib.sha256(split.encode()).hexdigest()

def weights_in_columns(codes, scales, k):
    lut=np.array([0,.5,1,1.5,2,3,4,6],np.float32);c=codes[:,k]
    return np.where(c&8,-lut[c&7],lut[c&7])*v86.convert_scales(scales[:,k//16])

def dot(x,b,e,p,gate):
    num,inner=x.shape
    partials=np.zeros((p,num,b[0].shape[1]),np.float32)
    for k in range(inner):
        codes=b[0][e]
        if not np.any(codes[:,k]&7):continue
        piece=(k//16)%p if gate else k//(inner//p)
        partials[piece]+=x[:,k,None]*weights_in_columns(codes,b[1][e],k)[None,:]
    return partials

def run_case(dll,rng,h,inner,p,counts,pre,groups):
    e=len(counts);bounds=np.array([pre,*(pre+np.cumsum(counts))],np.int32);last=int(bounds[-1]);slots=last+2
    gate=base.weight_bank(rng,e,2*inner,h);down=base.weight_bank(rng,e,h,inner)
    x=v86.bf16(rng.integers(-8,9,(slots,h)).astype(np.float32)/16)
    x[:pre]=0x7fc1;x[last:]=0x7fc2;original=x.copy()
    global13=v86.bf16(rng.choice(np.array([.125,.5,1,1.5],np.float32),e))
    out=np.full((p,slots,h),np.float32(-91681.25));expected=out.copy();near=0
    for expert,(a,z) in enumerate(zip(bounds,bounds[1:])):
        if a==z:continue
        parts=dot(v86.floats(x[a:z]),gate,expert,p,True)
        accum=np.zeros_like(parts[0])
        for v in parts:accum+=v
        scaled=v86.floats(v86.bf16(v86.floats(v86.bf16(accum))*v86.floats(global13[expert])))
        g,u=np.split(scaled,2,axis=-1)
        with np.errstate(over='ignore'):activated=v86.floats(v86.bf16((g/(1+np.exp(-g)))*u))
        expected[:,a:z]=dot(activated,down,expert,p,False)
    dll.run_split(*(a.ctypes.data for a in [x,gate[2],gate[3],down[2],down[3],global13,bounds]),h,inner,e,slots,p,groups,out.ctypes.data)
    np.testing.assert_array_equal(x,original)
    np.testing.assert_array_equal(out.view(np.uint32),expected.view(np.uint32))
    return dict(P=p,H=h,I=inner,local_counts=counts,workgroups=groups,live_fp32=sum(counts)*h*p,
                untouched_fp32=(pre+2)*h*p,unchanged_bf16_input=slots*h)

if __name__=="__main__":
    stage=Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as temp:
        dll,digest=compile_actual(stage,Path(temp));rng=np.random.default_rng(126)
        cases=[(512,1024,2,[0,0],2,1),(512,1024,2,[1,7,9,0],3,2),
               (512,2048,4,[17,0,2],1,1),(512,8192,4,[1],3,1),
               (4096,4096,2,[1],2,1),(4096,4096,4,[1],3,1)]
        results=[run_case(dll,rng,*case) for case in cases]
        print(json.dumps(dict(header_sha256=digest,cases=results,physical_gpu=False),indent=2))
