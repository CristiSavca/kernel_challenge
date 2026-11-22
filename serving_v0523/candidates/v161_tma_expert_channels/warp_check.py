"""Run the TMA coordinates, double buffering and exact production MMA/epilogue using a CTA CPU emulator."""
import ctypes, hashlib, importlib.util, json, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("cooperative_warp",here.parent/"v153_cooperative_expert_channels/warp_check.py")
ref=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

TMA=r"""
struct CUtensorMap { const char* data; unsigned long long x,y,pitch; unsigned bx,by,element; };
struct BankMaps {CUtensorMap weight,scale;};
unsigned long long tma_tx=0,tma_bytes=0;
unsigned remaining=0, issued_phase=0;
void issue_tensor(void* dest,const CUtensorMap* m,unsigned x,unsigned y,unsigned bar){
  if(threadIdx.x || x+m->bx>m->x || y+m->by>m->y || !bar || m->element*m->bx%16)std::abort();
  unsigned count=m->element*m->bx;
  if(count*m->by>remaining)std::abort();
  remaining-=count*m->by;
  ++tma_tx; tma_bytes+=count*m->by;
  for(unsigned row=0;row<m->by;row++)for(unsigned offset=0;offset<count;offset+=16)
    copy16((char*)dest+row*count+offset,m->data+(y+row)*m->pitch+x*m->element+offset,16);
}
"""

def compile_device(stage, path):
    root=stage/"python/sglang/kernels/jit/csrc/muse_spark"
    src=(root/"expert_route_pair.cuh").read_text()
    tma=(root/"expert_tma_route_pair.cuh").read_text()
    pre=src[src.index("namespace sglang {"):src.index("using Words =")]
    pre,n=re.subn(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',
                  r"\1 = pair_mul(\2, \3);",pre)
    assert n==2
    body=src[src.index("SGL_DEVICE float scaled_bf16"):src.index("struct Slice {")]
    body+=src[src.index("constexpr uint32_t kChannelWarps"):src.index("SGL_DEVICE void stage_packed")].rsplit("template",1)[0]
    body+=src[src.index("SGL_DEVICE void contract_fragment"):src.index("struct VectorStage")]
    issue=tma[tma.index("struct TensorStage {"):tma.index("template <bool Gate>\n__global__")]
    issue,n=re.subn(r'asm volatile\("mbarrier.arrive.expect_tx.*? : "memory"\);',
                    'if(remaining)std::abort();remaining=bytes;',issue,flags=re.S)
    assert n==1
    issue,n=re.subn(r"uint32_t complete;.*?while \(!complete\);",
                    'if(remaining || *phase!=issued_phase)std::abort();issued_phase^=1;',issue,flags=re.S)
    assert n==1
    issue=issue.replace("    publish_packed();","")
    body+=TMA+issue
    body+=src[src.index("template <bool Gate, class Stage = VectorStage>"):src.index("template <bool Gate>\n__global__ __launch_bounds__(32 * kChannelWarps)")]
    body,n=re.subn(r"  const uint32_t shared = .*?\n  asm volatile\([^\n]+;", "  copy16(dest,src,16);",body,flags=re.S)
    assert n==1
    body,n=re.subn(r'    const uint32_t address = .*? : "memory"\);',
                   "    copy16(&dest.inputs[step][route][4 * half],input + 8 * lane,valid_bytes);",body,flags=re.S)
    assert n==1
    body=body.replace('asm volatile("cp.async.commit_group;" ::: "memory");','')
    body=body.replace("void contract_fragment(","Op contract_fragment(")
    body,n=re.subn(r'asm volatile\(\s*"mma.sync.*?\)\);','return Op{&sum,{a0,a1,a2,a3},{lo,hi}};',body,flags=re.S)
    assert n==1 and "asm volatile" not in body
    body=body.replace("class Stage = VectorStage","class Stage = TensorStage").replace("void project_channels(","Routine sim(")
    body=body.replace("stage.publish();","stage.publish(); co_yield Op{};").replace("contract_fragment(gate,","co_yield contract_fragment(gate,").replace("contract_fragment(up,","co_yield contract_fragment(up,")
    driver=ref.DRIVER.replace('PackedTile<Gate> buffers[2];','''alignas(128) PackedTile<Gate> buffers[2];
    const unsigned weights=(Gate?2:1)*p.output_width;
    const unsigned max_expert=*std::max_element(p.plan+1, p.plan+1+p.plan[0]*9);
    BankMaps maps{{(char*)p.bank.weight,2*weights,(max_expert+1ul)*(p.input_width/16),8*weights,256,8,4},
                  {(char*)p.bank.scale,weights/2,(max_expert+1ul)*(p.input_width/16),weights,64,8,2}};
    uint32_t phase[256]{}; issued_phase=0;''').replace(
        "for(auto& lane:lanes) lane=sim<Gate>(p,p.plan+1+9*j,col,buffers);",
        "for(unsigned l=0;l<256;l++) lanes[l]=sim<Gate>(p,p.plan+1+9*j,col,buffers,TensorStage{&maps,1,phase+l});")
    # p.plan may have zero live jobs; use the expert fields only.
    driver=driver.replace('const unsigned max_expert=*std::max_element(p.plan+1, p.plan+1+p.plan[0]*9);',
        'unsigned max_expert=0; for(int j=0;j<p.plan[0];j++)max_expert=std::max(max_expert,unsigned(p.plan[1+9*j]));')
    code=ref.parent.v86.PRELUDE+ref.COPY+ref.parent.base.v89.HOST_MUL+pre+body+"\n}}\n"
    code+=ref.parent.v86.POSTLUDE.split('extern "C" void simulate')[0]+driver
    code+='extern "C" uint64_t bulk(unsigned i){return i?tma_bytes:tma_tx;}'
    cpp=path/"warp.cpp";lib=path/"warp.so";cpp.write_text(code)
    subprocess.run(["c++","-std=c++20","-fPIC","-shared","-fno-strict-aliasing","-O2",cpp,"-o",lib],check=True)
    dll=ctypes.CDLL(str(lib));dll.simulate.argtypes=[ctypes.c_void_p]*10+[ctypes.c_uint32]*4
    dll.simulate.restype=None
    for name in ("count","bulk"):
        f=getattr(dll,name);f.argtypes=[ctypes.c_uint32];f.restype=ctypes.c_uint64
    return dll,[hashlib.sha256(h.encode()).hexdigest() for h in (src,tma)]


if __name__=="__main__":
    rng=np.random.default_rng(161)
    fixtures=[(128,128,2,np.full((8,3),-1),3),(256,128,4,rng.integers(-1,5,(11,3)),1),
              (128,512,3,np.zeros((19,2),np.int32),2),(512,256,3,rng.integers(-2,4,(8,8)),2),
              (384,640,3,np.array([0]*8+[1]*16+[2]*17+[-1]*4).reshape(9,5),3),
              (4096,4096,1,np.array([[-1,0],[-1,-1]]),1)]
    with tempfile.TemporaryDirectory() as d:
        dll,digests=compile_device(Path(sys.argv[1]),Path(d))
        cases=[ref.parent.run_case(dll,rng,*f) for f in fixtures]
        order=[ref.sensitive.check(dll,split) for split in (2,4)]
        counts=[dll.count(i) for i in range(3)]
        bulk=[dll.bulk(i) for i in range(2)]
    print(json.dumps(dict(header_sha256=digests,cases=cases,ascending_k_cancellation=order,
        valid_16byte_vectors=counts[0],zero_fill_vectors=counts[1],cta_rendezvous=counts[2],
        tensor_transactions=bulk[0],tensor_payload_bytes=bulk[1],hardware=False),indent=2))
