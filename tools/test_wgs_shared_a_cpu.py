#!/usr/bin/env python3
"""Source-bound CPU checks for fixed shared-A WGS; never imports/executes CUDA."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile

import build_wgs_shared_a_candidate as builder
from test_wgs_wait2_cpu import literal, segment, preprocess, comments

ROOT = builder.ROOT
CANDIDATE = builder.CANDIDATE
OUT = ROOT / "experiments/iterations/20261104-wgs-shared-a-cpu.json"


def sha(b):
    return hashlib.sha256(b).hexdigest()


class NativeCheckFailed(AssertionError):
    pass


def function(src, name):
    masked = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/', lambda m: " "*len(m[0]), src)
    m = re.search(r'\b'+re.escape(name)+r'\s*\([^;{}]*\)\s*\{', masked)
    assert m, name
    start = masked.index("{", m.start()); depth = 1
    for end in range(start+1, len(src)):
        depth += (masked[end] == "{") - (masked[end] == "}")
        if depth == 0:
            return src[m.start():end+1]
    raise AssertionError(name)


def compile_run(code, args=()):
    with tempfile.TemporaryDirectory(prefix="okc-shared-a-cpu-") as td:
        exe = str(Path(td)/"check")
        p = subprocess.run(["clang++", "-x", "c++", "-std=c++17", "-O2", "-o", exe, "-"], input=code, text=True, capture_output=True)
        assert p.returncode == 0, p.stderr
        p = subprocess.run([exe, *map(str,args)], text=True, capture_output=True)
        if p.returncode != 0:
            raise NativeCheckFailed(p.stderr)
        return p.stdout, p.stderr


def wrappers(src):
    a=src.index("template <int NTT> struct WGSMMA;")
    b=src.index("\n};",src.index("template <> struct WGSMMA<8>"))+3
    return src[a:b]


def conformance(parent, candidate):
    pt,ct=ast.parse(parent),ast.parse(candidate)
    def key(n):
        return n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
    pn={key(n):n for n in pt.body if key(n)};cn={key(n):n for n in ct.body if key(n)}
    assert pn.keys()==cn.keys()
    for k in pn:
        if k not in ("CUDA_SRC","_ext"):assert ast.dump(pn[k])==ast.dump(cn[k]),k
    ext=ast.get_source_segment(candidate,cn['_ext'])
    ext=ext.replace("okc_moe_wgs_shared_a_v15_","okc_moe_kperm3_v9_")
    ext=ext.replace('verbose="-DOKC_WGS=1" in defs,  # preserve WGS compiler diagnostics','verbose=False,')
    assert ext==ast.get_source_segment(parent,pn['_ext'])
    ps,cs=literal(pt,"CUDA_SRC"),literal(ct,"CUDA_SRC")
    pseg,cseg=segment(ps),segment(cs)
    assert cseg[cseg.index("  if (OKC_WGSX & 8)"):]==pseg[pseg.index("  if (OKC_WGSX & 8)"):]
    norm=cseg
    norm=re.sub(r'  static_assert\(GS == 2[^\n]*\n  const uint32_t aring_s[^\n]*', '  uint32_t af[2][GS][4];',norm)
    norm=re.sub(r'        const uint32_t atile[^\n]*\n        // Every lane[^\n]*\n        asm volatile\([^\n]*\n','',norm)
    norm=norm.replace('          wgs_a_store(atile + jj * 2048, wp, lane, bmul2l(r0, sgp), bmul2h(r1, sgp),\n                      bmul2l(r2, sgp), bmul2h(r3, sgp));',
        '          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);\n          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);')
    norm=re.sub(r'        // Publish generic[^\n]*\n        asm volatile\([^\n]*\n        asm volatile\([^\n]*\n        const uint64_t ad[^\n]*\n','',norm)
    norm=norm.replace('ad + (uint64_t)(jj * 128)', 'af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3]')
    norm=norm.replace('::mma0(d, ad, xd)', '::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd)')
    assert comments(norm)==comments(pseg),"segment change outside explicit storage/publication/descriptor substitution"
    # Independently inspect each emitted wrapper's output constraints and BF16 SS operands.
    cw=wrappers(cs);pw=wrappers(ps)
    for n in range(1,9):
        a=cw.index(f'template <> struct WGSMMA<{n}>');b=cw.find('template <>',a+1)
        one=cw[a:b if b>=0 else len(cw)]
        for zero,mode in ((0,'=f'),(1,'+f')):
            regs=', '.join(f'%{i}' for i in range(n*4))
            instruction=f'wgmma.mma_async.sync.aligned.m64n{n*8}k16.f32.bf16.bf16 {{{regs}}}, %{n*4}, %{n*4+1}, {zero}, 1, 1, 0, 0;'
            assert one.count(instruction)==1
            for i in range(n*4):assert one.count(f'"{mode}"(d[{i//4}][{i%4}])')==1
        assert one.count(': "l"(adesc), "l"(bdesc) : "memory"')==2
    restored=cs.replace(cseg,pseg).replace(cw,pw)
    restored=restored.replace(builder.A_HELPER,'')
    restored=restored.replace(builder.HOST+'\n','')
    restored=restored.replace('#if OKC_WGS\n    okc_shared_a_resources(sm, e0, e1);\n#endif\n','')
    restored=restored.replace(' + okc::WGS_A_BYTES;',';')
    restored=restored.replace('// ---------------------------------------------------------------- WGS: SS-wgmma','// ---------------------------------------------------------------- WGS: RS-wgmma')
    restored=restored.replace('(A = exact dequantized codes in two small shared banks, N = 8 * n-tiles)', '(A = dequantized codes in registers, N = 8 * n-tiles)')
    restored=restored.replace('#define OKC_WGSNX 2                             // fixed shared-A experiment: two X slots','#define OKC_WGSNX 3                             // x ring slots')
    restored=restored.replace('constexpr int WGS_NS = 2;', 'constexpr int WGS_NS = 3;')
    restored=restored.replace('static_assert(WGS_NS == 2 && WGS_NX == 2, "Shared-A uses exactly two joined ring slots");','static_assert(WGS_NS == WGS_NX, "Joined completion requires equal ring depths");')
    restored=restored.replace('constexpr int WGS_A_BANK_BYTES = 4 * 2 * 64 * 16 * 2;  // four WG-private GS2 tiles\nconstexpr int WGS_A_BYTES = 2 * WGS_A_BANK_BYTES;\n','')
    restored=restored.replace('constexpr int BW_F = 0, BW_E = WGS_NS, BX_F = 2 * WGS_NS, BX_E = 2 * WGS_NS + WGS_NX;  // barrier indices','constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices')
    assert restored==ps,"non-scope CUDA changed"
    assert 'except' not in ast.get_source_segment(candidate,cn['prepare'])
    assert 'except' not in ast.get_source_segment(candidate,cn['kernel_fn'])
    builds=0
    for name in ('TINY_DEFINES','TINY2_DEFINES','MID_DEFINES','MAIN_FALLBACK_DEFINES'):
        for mul in (0,1):
            for prmt in (0,1):
                defs=[*literal(pt,name),f'-DOKC_MULDEC={mul}',f'-DOKC_PRMTSC={prmt}','-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']
                assert preprocess(ps,defs)==preprocess(cs,defs),(name,mul,prmt)
                builds+=1
    return dict(restored_parent_cuda_sha256=sha(restored.encode()),non_wgs_preprocessed_builds=builds,ss_wrappers=16,
        unchanged_python_nodes=len(pn)-2,unchanged_producer_gather_route_epilogue_fixup_finalizer=True),ps,cs


def translated_store(cs):
    helper=function(cs,'wgs_a_store')
    pattern=r'asm volatile\("st.shared.u32 \[%0\], %1;\\n" :: "r"\(([^\n]*?)\), "r"\((r[0-3])\) : "memory"\);'
    helper,n=re.subn(pattern,lambda m:f'cpu_store({m[1]}, {m[2]});',helper)
    assert n==4
    return 'void '+helper


def native_layout(cs):
    desc='uint64_t '+function(cs,'wg_desc')
    code=r'''
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <cstring>
uint32_t words[8192]; int writes[8192];
void cpu_store(uint32_t addr,uint32_t value){assert(addr<32768&&!(addr&3));assert(!writes[addr/4]++);words[addr/4]=value;}
'''+desc+'\n'+translated_store(cs)+r'''
int main(){
 uint64_t checked=0;int banks=0;
 for(int bank=0;bank<2;bank++)for(int wg=0;wg<4;wg++)for(int jj=0;jj<2;jj++){
  uint32_t base=bank*16384+wg*4096+jj*2048;
  auto d=wg_desc(base,128,256);assert((d&0x3fff)*16==base&&((d>>16)&0x3fff)*16==128&&((d>>32)&0x3fff)*16==256);
  for(int wp=0;wp<4;wp++)for(int lane=0;lane<32;lane++){
   uint32_t v[4];for(int r=0;r<4;r++){int m=16*wp+(lane>>2)+8*(r&1),k=2*(lane&3)+8*(r>>1);
    v[r]=uint32_t(m*16+k)|(uint32_t(m*16+k+1)<<16);}
   wgs_a_store(base,wp,lane,v[0],v[1],v[2],v[3]);
  }
  for(int m=0;m<64;m++)for(int k=0;k<16;k++){
   int off=(m/8)*256+(k/8)*128+(m%8)*16+(k%8)*2;
   assert(((words[(base+off)/4]>>((off%4)*8))&65535)==m*16+k);checked++;
  }
  for(int wp=0;wp<4;wp++)for(int r=0;r<4;r++){
   bool seen[32]={};for(int lane=0;lane<32;lane++){
    int m=16*wp+(lane>>2)+8*(r&1),k=2*(lane&3)+8*(r>>1);
    int off=(m/8)*256+(k/8)*128+(m%8)*16+(k%8)*2;
    assert(!seen[(off/4)%32]);seen[(off/4)%32]=true;
   }banks++;
  }
 }
 for(int i=0;i<8192;i++)assert(writes[i]==1);
 // All possible BF16 bit strings survive both positions of the actual uint32 stores.
 uint64_t bits=0;
 for(uint32_t v=0;v<65536;v++){
  memset(writes,0,sizeof(writes));uint32_t x=v|((65535-v)<<16);
  wgs_a_store(0,0,0,x,x,x,x);for(int i:{0,64,32,96}){assert(words[i]==x);bits+=2;}
 }
 printf("%llu %d %llu\n",(unsigned long long)checked,banks,(unsigned long long)bits);
}
'''
    code=code.replace('#include <cstring>','#include <cstring>\n#include <initializer_list>')
    out,_=compile_run(code)
    return dict(zip(('bf16_coordinates','bank_conflict_free_warp_store_groups','exhaustive_bit_half_checks'),map(int,out.split())))


def native_trace(cs, shared):
    seg=segment(cs);body=seg[:seg.index('  if (OKC_WGSX & 8)')]+'}\n'
    ns=2 if shared else 3
    body=body.replace('float d[WGS_NT][4];','float d[WGS_NT][4] = {};').replace('float d_odd[WGS_NT][4];','float d_odd[WGS_NT][4] = {};')
    body=body.replace('asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");','cpu_bar(3 + wg);')
    body=body.replace('asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");','cpu_proxy();')
    prefix=r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <type_traits>
#define __device__
#define __forceinline__ inline
#define OKC_WGSGS 2
#define OKC_WGSDESC0 1
#define OKC_WGSFCI 3
#define WARP_IDX(x) ((x)>>5)
constexpr int WGS_NS=DEPTH,WGS_NX=DEPTH,WGS_NT=8,UNIT_BYTES=36864,WGS_PIECE=9216,WGS_XS=32768;
constexpr int BW_F=0,BW_E=WGS_NS,BX_E=3*WGS_NS,WGS_A_BANK_BYTES=16384;
struct uint4 {uint32_t x,y,z,w;}; struct uint2 {uint32_t x,y;};
struct Params {uint32_t zero=0;}; struct Smem {uint64_t bars[24]{};};
struct WgsPend {int kind,u,slot,nst;}; struct {int x;} threadIdx;
alignas(128) unsigned char ring_mem[3*UNIT_BYTES+3*WGS_XS+32768];
Smem* global_s;int seq_now,group_now,call_in_group,publication=0,stores=0;float*primary;
uint64_t trace_hash=1469598103934665603ull,mmacalls=0,groups=0,flushes=0,releases=0,bars=0;
struct Ref{uint32_t*p;uint32_t value;};struct Group{int seq;std::vector<Ref>refs;};
std::vector<Group>pending;std::vector<Ref>current;
void mix(uint64_t v){trace_hash=(trace_hash^v)*1099511628211ull;}
void check(){for(auto &g:pending)for(auto r:g.refs)assert(*r.p==r.value);}
uint32_t smem_u32(const void*p){return uint32_t((const unsigned char*)p-ring_mem);}
uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t c){return a^b^c;}
void dq4(uint32_t q,uint32_t&a,uint32_t&b,uint32_t&c,uint32_t&d){a=q^1;b=q^2;c=q^3;d=q^4;}
uint32_t bmul2l(uint32_t a,uint32_t b){return a^b;}
uint32_t bmul2h(uint32_t a,uint32_t b){return a^(b+12345);}
void cpu_bar(int id){check();assert(id==3+(threadIdx.x>>7));bars++;if(publication==0){publication=1;stores=0;}else{assert(publication==2&&stores==8);publication=3;}}
void cpu_proxy(){assert(publication==1&&stores==8);publication=2;}
void cpu_store(uint32_t addr,uint32_t value){check();assert(publication==1);assert(addr>=WGS_NS*UNIT_BYTES+WGS_NX*WGS_XS&&addr+4<=WGS_NS*UNIT_BYTES+WGS_NX*WGS_XS+32768);assert(!(addr&3));*(uint32_t*)(ring_mem+addr)=value;stores++;}
void wg_fence(){check();}
void wg_commit(){assert(call_in_group==2);check();pending.push_back({seq_now,current});current.clear();call_in_group=0;groups++;group_now++;publication=0;}
template<int N>void wg_wait(){check();while(pending.size()>N)pending.erase(pending.begin());}
void wgs_wait(uint64_t*bar,uint32_t parity,int tag){
 check();seq_now=tag-3000000;assert(parity==((seq_now/WGS_NS)&1));assert(bar==global_s->bars+seq_now%WGS_NS);group_now=0;
 for(auto &g:pending)assert(g.seq%WGS_NX!=seq_now%WGS_NX);
 auto*p=(uint32_t*)(ring_mem+(seq_now%WGS_NS)*UNIT_BYTES);
 for(int i=0;i<UNIT_BYTES/4;i++)p[i]=uint32_t(seq_now*100003+i*17+7);
}
void mbar_arrive1(uint64_t*p){
 check();auto i=p-global_s->bars;
 if(i>=BX_E&&i<BX_E+WGS_NX){for(auto &g:pending)assert(g.seq%WGS_NX!=i-BX_E);releases++;}
 else assert(i>=BW_E&&i<BW_E+WGS_NS);
}
template<bool PB>void wgs_flush(const Params&,Smem&,WgsPend&pd){assert(group_now==4&&pd.kind);check();pd.kind=0;flushes++;}
template<int NTT>void record(bool zero,float(&d)[8][4],uint32_t*a,uint32_t*b,uint32_t*c,uint32_t*e,uint64_t desc){
 check();if(!primary)primary=&d[0][0];bool odd=&d[0][0]!=primary;
 if(NTT==1)assert(odd==bool(group_now&1));else assert(!odd);
 for(auto*p:{a,b,c,e}){current.push_back({p,*p});mix(*p);}
 uint32_t byte=(desc&0x3fff)*16;
 assert(((desc>>16)&0x3fff)*16==128&&((desc>>32)&0x3fff)*16==4096);
 int step=(byte-WGS_NS*UNIT_BYTES-(seq_now%WGS_NX)*WGS_XS)/256;
 assert(step==2*group_now+call_in_group);
 mix(seq_now);mix(step);mix(NTT);mix(odd);mix(zero);mmacalls++;call_in_group++;
}
'''.replace('DEPTH',str(ns))
    desc='uint64_t '+function(cs,'wg_desc')
    if shared:
        stub=r'''
template<int NTT>struct WGSMMA{
 static void op(bool zero,float(&d)[8][4],uint64_t ad,uint64_t bd){
  assert(publication==3);assert(((ad>>16)&0x3fff)*16==128&&((ad>>32)&0x3fff)*16==256);
  uint32_t base=(ad&0x3fff)*16,tid=threadIdx.x,wp=(tid>>5)&3,lane=tid&31;
  assert(base==WGS_NS*UNIT_BYTES+WGS_NX*WGS_XS+(group_now&1)*16384+(tid>>7)*4096+call_in_group*2048);
  uint32_t*p[4];for(int r=0;r<4;r++){int m=16*wp+(lane>>2)+8*(r&1),k=2*(lane&3)+8*(r>>1);
   int off=(m/8)*256+(k/8)*128+(m%8)*16+(k%8)*2;p[r]=(uint32_t*)(ring_mem+base+off);}
  record<NTT>(zero,d,p[0],p[1],p[2],p[3],bd);
 }
 static void mma0(float(&d)[8][4],uint64_t ad,uint64_t bd){op(true,d,ad,bd);}
 static void mma(float(&d)[8][4],uint64_t ad,uint64_t bd){op(false,d,ad,bd);}
};
'''
        helper=translated_store(cs)
    else:
        stub=r'''
template<int NTT>struct WGSMMA{
 static void mma0(float(&d)[8][4],uint32_t&a,uint32_t&b,uint32_t&c,uint32_t&e,uint64_t desc){record<NTT>(true,d,&a,&b,&c,&e,desc);}
 static void mma(float(&d)[8][4],uint32_t&a,uint32_t&b,uint32_t&c,uint32_t&e,uint64_t desc){record<NTT>(false,d,&a,&b,&c,&e,desc);}
};
''';helper=''
    suffix=r'''
template<bool PB,int N>void run(){
 Params p;Smem s;global_s=&s;
 for(int start=0;start<6;start++)for(int n=1;n<=16;n++)for(int tid:{0,1,31,32,127,128,511}){
  threadIdx.x=tid;int seq=start;WgsPend pd{1,0,0,0};primary=nullptr;pending.clear();current.clear();call_in_group=0;publication=0;
  wgs_seg<PB,N>(p,s,ring_mem,0,0,n,-1,seq,pd);assert(seq==start+n&&pending.empty()&&current.empty());
 }
}
int main(){
 run<false,1>();run<false,2>();run<false,3>();run<false,4>();run<false,5>();run<false,6>();run<false,7>();run<false,8>();
 run<true,1>();run<true,2>();run<true,3>();run<true,4>();run<true,5>();run<true,6>();run<true,7>();run<true,8>();
 printf("%llu %llu %llu %llu %llu %llu\n",(unsigned long long)trace_hash,(unsigned long long)mmacalls,
 (unsigned long long)groups,(unsigned long long)flushes,(unsigned long long)releases,(unsigned long long)bars);
}
'''
    out,_=compile_run(prefix+desc+'\n'+helper+stub+body+suffix)
    return dict(zip(('trace_hash','mma_calls','commit_groups','flushes','x_releases','barriers'),map(int,out.split())))


def host_mock(cs):
    host='void '+function(cs,'okc_shared_a_resources')
    prefix=r'''
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("resource");}while(0)
using cudaError_t=int;constexpr int cudaSuccess=0;
struct cudaFuncAttributes{int numRegs=0;size_t localSizeBytes=0,sharedSizeBytes=0;int maxDynamicSharedSizeBytes=0;};
namespace okc {constexpr int NTL=576;template<bool B>void moe_fused(){}}
int mode=0,queries=0;
template<class F>int cudaFuncGetAttributes(cudaFuncAttributes*a,F f){
 queries++;bool b=f==okc::moe_fused<true>;a->numRegs=96;a->localSizeBytes=mode==7?128:0;a->sharedSizeBytes=0;a->maxDynamicSharedSizeBytes=194304;
 return mode==1+int(b)?17:0;}
template<class F>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*o,F f,int nt,size_t sm){
 queries++;assert(nt==576&&sm==194304);bool b=f==okc::moe_fused<true>;*o=(mode==5+int(b)?0:1);return mode==3+int(b)?18:0;}
'''
    suffix=r'''
int main(){int passed=0;for(mode=0;mode<10;mode++){
 queries=0;bool ok=true;try{okc_shared_a_resources(194304,mode==8?19:0,mode==9?20:0);}catch(const std::runtime_error&){ok=false;}
 assert(queries==4);assert(ok==(mode==0||mode==7));passed++;
}printf("%d\n",passed);}
'''
    out,err=compile_run(prefix+host+suffix)
    lines=[x for x in err.splitlines() if x.startswith('SSAR ')]
    assert len(lines)==20 and all('dyn=194304' in x for x in lines)
    # Compile the real first-call block as well: a thrown resource check must
    # leave attr false, skip the launch continuation, and be retried next call.
    start=cs.index('  static bool attr = false;',cs.index('void okc_run('))
    stop=cs.index('  cudaStream_t st =',start)
    first=cs[start:stop]
    pre2=prefix+r'''
#define OKC_WGS 1
#define OKC_LAUNCHCHK 0
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize=0;
int setters=0,launches=0;
size_t okc_smem_bytes(){return 194304;}
template<class F>int cudaFuncSetAttribute(F f,int,int sm){assert(sm==194304);setters++;return 0;}
'''
    first_function='void first_call(){\n'+first+'  launches++;\n}\n'
    suffix2=r'''
int main(){
 mode=1;bool rejected=false;try{first_call();}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&launches==0&&setters==2&&queries==4);
 mode=0;first_call();assert(launches==1&&setters==4&&queries==8);
 first_call();assert(launches==2&&setters==4&&queries==8);
 printf("failure_propagates retry_admits cached_success\n");
}
'''
    first_out,first_err=compile_run(pre2+host+first_function+suffix2)
    assert first_out.strip()=='failure_propagates retry_admits cached_success'
    assert len([x for x in first_err.splitlines() if x.startswith('SSAR ')])==4
    return dict(modes=int(out),records=len(lines),max_record_bytes=max(len(x.encode())+1 for x in lines),local_allocation_does_not_reject=True,
        actual_first_call_failure_propagates=True,failed_call_does_not_cache_success=True,successful_attributes_cached=True)


def packing(candidate):
    import test_wgs_kperm3_cpu as p
    old=p.CANDIDATE;p.CANDIDATE=candidate
    log=io.StringIO()
    try:
        with contextlib.redirect_stdout(log):p.main()
    finally:p.CANDIDATE=old
    return dict(actual_candidate_packing=True,output=log.getvalue().splitlines())


def negative_native(cs):
    rejected=[]
    cases=(
        ('wrong_register_pair',lambda: native_layout(cs.replace('"r"(tile + off), "r"(r0)', '"r"(tile + off), "r"(r1)',1))),
        ('wrong_A_descriptor_step',lambda: native_trace(cs.replace('ad + (uint64_t)(jj * 128)','ad + (uint64_t)(jj * 64)'),True)),
        ('one_A_bank_wait1',lambda: native_trace(cs.replace('(ci & 1) * WGS_A_BANK_BYTES','(ci & 0) * WGS_A_BANK_BYTES'),True)),
    )
    for name,check in cases:
        try:check()
        except NativeCheckFailed:rejected.append(name)
        else:raise AssertionError('negative control accepted: '+name)
    return rejected


def run(candidate=CANDIDATE,expected_sha=None,independent=True):
    candidate=Path(candidate);raw=candidate.read_bytes();digest=sha(raw)
    if expected_sha:assert digest==expected_sha
    parent=builder.PARENT.read_bytes();assert sha(parent)==builder.PARENT_SHA
    assert builder.build().encode()==raw,"candidate is not exactly reproducible"
    scope,ps,cs=conformance(parent.decode(),raw.decode())
    layout=native_layout(cs)
    pt=native_trace(ps,False);ct=native_trace(cs,True)
    assert ct['barriers']==ct['commit_groups']*2 and pt['barriers']==0
    assert {k:v for k,v in pt.items() if k!='barriers'}=={k:v for k,v in ct.items() if k!='barriers'}
    result=dict(status='PASS',source_sha256=digest,parent_sha256=builder.PARENT_SHA,cuda_sha256=sha(cs.encode()),
        conformance=scope,native_layout=layout,native_parent_operand_trace=pt,native_shared_operand_trace=ct,
        host_resources=host_mock(cs),packing=packing(candidate),native_negative_controls=negative_native(cs),
        limits='CPU execution uses actual source control flow/scatter/descriptor with mocked CUDA and decode arithmetic; it proves operand transport/order and source preservation, not tensor numerical results, CUDA compilation, GPU ordering, compiler scheduling or speed.')
    if independent:
        import test_wgs_shared_a_lifetimes_critic as critic
        result['independent_lifetimes']=critic.run(candidate,digest)
    assert sha(candidate.read_bytes())==digest
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,default=CANDIDATE);ap.add_argument('--expected-sha');ap.add_argument('--without-independent',action='store_true');ap.add_argument('--output',type=Path,default=OUT)
    a=ap.parse_args();r=run(a.candidate,a.expected_sha,not a.without_independent)
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(status=r['status'],source_sha256=r['source_sha256'],report_sha256=sha(a.output.read_bytes()),output=str(a.output)),indent=2))
