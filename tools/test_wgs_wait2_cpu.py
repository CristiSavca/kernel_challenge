#!/usr/bin/env python3
"""Scope and actual-segment lifetime checks; native CPU stubs, never CUDA."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "muse_moe_w4a16/candidates/native_wgs_kperm3_v9.py"
CANDIDATE = ROOT / "muse_moe_w4a16/candidates/native_wgs_wait2_v12.py"
OUT = ROOT / "experiments/iterations/20261104-wgs-wait2-source-cpu.json"


def literal(tree, name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def segment(src):
    a = src.index("template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(")
    b = src.index("\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_phase(", a)
    return src[a:b]


def comments(s):
    return re.sub(r"//[^\n]*", "", s)


def preprocess(src, defs):
    src = re.sub(r"^#include[^\n]*", "", src, flags=re.M)
    r = subprocess.run(["clang++", "-E", "-P", "-x", "c++", "-D__CUDA_ARCH__=900", *defs, "-"],
                       input=src, text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def native_trace(seg):
    # Compile the actual wgs_seg body through its terminal drain/x release.
    # The unchanged terminal epilogue/fixup is checked textually below.
    body = seg[:seg.index("  if (OKC_WGSX & 8)")] + "}\n"
    prefix = r'''
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
constexpr int WGS_NS=3,WGS_NX=3,WGS_NT=8,UNIT_BYTES=36864,WGS_PIECE=9216,WGS_XS=32768;
constexpr int BW_F=0,BW_E=3,BX_E=9;
struct uint4 {uint32_t x,y,z,w;}; struct uint2 {uint32_t x,y;};
struct Params {uint32_t zero=0;}; struct Smem {uint64_t bars[24]{};};
struct WgsPend {int kind,u,slot,nst;}; struct {int x;} threadIdx;
alignas(128) unsigned char ring_mem[3*UNIT_BYTES+3*WGS_XS];
Smem* global_s; int seq_now,group_now,call_in_group;float* primary;
uint64_t trace_hash=1469598103934665603ull,mmacalls=0,groups=0,flushes=0,releases=0;
struct Ref {uint32_t*p;uint32_t value;};
struct Group {int seq;std::vector<Ref> refs;};
std::vector<Group> pending;std::vector<Ref> current;
void mix(uint64_t v){trace_hash=(trace_hash^v)*1099511628211ull;}
void check(){for(auto &g:pending)for(auto r:g.refs)assert(*r.p==r.value);}
uint32_t smem_u32(const void*p){return uint32_t((const unsigned char*)p-ring_mem);}
uint64_t wg_desc(uint32_t p,int,int){return p;}
uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t c){return a^b^c;}
void dq4(uint32_t q,uint32_t&a,uint32_t&b,uint32_t&c,uint32_t&d){a=q^1;b=q^2;c=q^3;d=q^4;}
uint32_t bmul2l(uint32_t a,uint32_t b){return a^b;}
uint32_t bmul2h(uint32_t a,uint32_t b){return a^(b+12345);}
void wg_fence(){check();}
void wg_commit(){assert(call_in_group==2);check();pending.push_back({seq_now,current});current.clear();call_in_group=0;groups++;group_now++;}
template<int N>void wg_wait(){check();while(pending.size()>N)pending.erase(pending.begin());}
void wgs_wait(uint64_t*,uint32_t parity,int tag){
 check();seq_now=tag-3000000;assert(parity==((seq_now/3)&1));group_now=0;
 for(auto &g:pending)assert(g.seq%3!=seq_now%3);
 auto*p=(uint32_t*)(ring_mem+(seq_now%3)*UNIT_BYTES);
 for(int i=0;i<UNIT_BYTES/4;i++)p[i]=uint32_t(seq_now*100003+i*17+7);
}
void mbar_arrive1(uint64_t*p){
 check();auto i=p-global_s->bars;
 if(i>=BX_E&&i<BX_E+3){for(auto &g:pending)assert(g.seq%3!=i-BX_E);releases++;}
 else assert(i>=BW_E&&i<BW_E+3);
}
template<bool PB>void wgs_flush(const Params&,Smem&,WgsPend&pd){assert(group_now==4&&pd.kind);check();pd.kind=0;flushes++;}
template<int NTT>struct WGSMMA {
 static void op(bool zero,float(&d)[8][4],uint32_t&a,uint32_t&b,uint32_t&c,uint32_t&e,uint64_t desc){
  check();if(!primary)primary=&d[0][0];bool odd=&d[0][0]!=primary;
  if(NTT==1)assert(odd==bool(group_now&1));else assert(!odd);
  // Compare zero-D with the parent's complete actual-source trace; no
  // numerical tensor arithmetic is emulated here.
  for(auto*p:{&a,&b,&c,&e}){current.push_back({p,*p});mix(*p);}
  mix(desc);mix(NTT);mix(odd);mix(zero);mmacalls++;call_in_group++;
 }
 static void mma0(float(&d)[8][4],uint32_t&a,uint32_t&b,uint32_t&c,uint32_t&e,uint64_t desc){op(true,d,a,b,c,e,desc);}
 static void mma(float(&d)[8][4],uint32_t&a,uint32_t&b,uint32_t&c,uint32_t&e,uint64_t desc){op(false,d,a,b,c,e,desc);}
};
'''
    # The real function initializes d via tensor MMA; initialize stub accumulators
    # only so its retained final NTT1 generic addition does not read indeterminate CPU floats.
    body = body.replace("float d[WGS_NT][4];", "float d[WGS_NT][4] = {};")
    body = body.replace("float d_odd[WGS_NT][4];", "float d_odd[WGS_NT][4] = {};")
    suffix = r'''
template<bool PB,int N>void run(){
 Params p;Smem s;global_s=&s;
 for(int start=0;start<6;start++)for(int n=1;n<=16;n++)for(int tid:{0,1,31,32,127,128,511}){
  threadIdx.x=tid;int seq=start;WgsPend pd{1,0,0,0};primary=nullptr;pending.clear();current.clear();call_in_group=0;
  wgs_seg<PB,N>(p,s,ring_mem,0,0,n,-1,seq,pd);assert(seq==start+n&&pending.empty()&&current.empty());
 }
}
int main(){
 run<false,1>();run<false,2>();run<false,3>();run<false,4>();run<false,5>();run<false,6>();run<false,7>();run<false,8>();
 run<true,1>();run<true,2>();run<true,3>();run<true,4>();run<true,5>();run<true,6>();run<true,7>();run<true,8>();
 printf("%llu %llu %llu %llu %llu\n",(unsigned long long)trace_hash,(unsigned long long)mmacalls,
 (unsigned long long)groups,(unsigned long long)flushes,(unsigned long long)releases);
}
'''
    with tempfile.TemporaryDirectory(prefix="okc-wait2-native-") as td:
        exe = str(Path(td) / "trace")
        r = subprocess.run(["clang++", "-x", "c++", "-std=c++17", "-O2", "-o", exe, "-"],
                           input=prefix+body+suffix, text=True, capture_output=True)
        assert r.returncode == 0, r.stderr
        r = subprocess.run([exe], text=True, capture_output=True)
        assert r.returncode == 0, r.stderr
        return dict(zip(("trace_hash", "mma_calls", "commit_groups", "flushes", "x_releases"), map(int, r.stdout.split())))


def main():
    pt = ast.parse(PARENT.read_bytes()); ct = ast.parse(CANDIDATE.read_bytes())
    ps = literal(pt,"CUDA_SRC"); cs = literal(ct,"CUDA_SRC")
    def key(n):
        return n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
    pn = {key(n): n for n in pt.body if key(n)}; cn = {key(n): n for n in ct.body if key(n)}
    assert pn.keys() == cn.keys()
    for k in pn:
        if k not in ("CUDA_SRC","_ext"): assert ast.dump(pn[k]) == ast.dump(cn[k]),k
    oldext = ast.get_source_segment(PARENT.read_text(),pn["_ext"])
    newext = ast.get_source_segment(CANDIDATE.read_text(),cn["_ext"])
    newext = newext.replace("okc_moe_wgs_wait2_v12_","okc_moe_kperm3_v9_")
    newext = newext.replace('verbose="-DOKC_WGS=1" in defs,  # expose this experiment\'s WGS ptxas diagnostics',"verbose=False,")
    assert newext == oldext
    pseg = segment(ps); cseg = segment(cs)
    assert cseg.count('af[ci & 3]') == 20 and 'af[ci & 1]' not in cseg
    assert cseg.count('wg_wait<2>();') == 1 and cseg.count('wg_wait<0>();') == 1
    assert cseg.count('if (ci & 1)') == pseg.count('if (ci & 1)') == 1
    norm = cseg.replace('af[4][GS][4]','af[2][GS][4]').replace('af[ci & 3]','af[ci & 1]').replace('wg_wait<2>();','wg_wait<1>();')
    norm = norm.replace('if (ci == 1) {\n          if (prev >= 0)', 'if (ci == 0) {\n          if (prev >= 0)')
    norm = re.sub(r'  static_assert\(GS == 2[^\n]*\n','',norm)
    assert comments(norm) == comments(pseg)
    host_start = cs.index('#if OKC_WGS\n    TORCH_CHECK(e0 == cudaSuccess && e1 == cudaSuccess, "wait2')
    host_end = cs.index('\n#endif',host_start)+len('\n#endif')
    host = cs[host_start:host_end]
    assert 'occ0 >= 1 && occ1 >= 1' in host
    assert host.count('cudaOccupancyMaxActiveBlocksPerMultiprocessor(') == 2
    assert cs.index('occ0 >= 1 && occ1 >= 1') < cs.index('if (g_okc_pdl)')
    restored = cs[:host_start]+cs[host_end:]
    restored = restored.replace(cseg,pseg).replace('\n\n    attr = true;','\n    attr = true;')
    assert restored == ps
    builds = 0
    for name in ('TINY_DEFINES','TINY2_DEFINES','MID_DEFINES','MAIN_FALLBACK_DEFINES'):
        for mul in (0,1):
            for prmt in (0,1):
                defs = [*literal(pt,name),f'-DOKC_MULDEC={mul}',f'-DOKC_PRMTSC={prmt}','-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']
                assert preprocess(ps,defs) == preprocess(cs,defs),(name,mul,prmt)
                builds += 1
    parent_native = native_trace(pseg); candidate_native = native_trace(cseg)
    assert parent_native == candidate_native
    result = dict(source_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        cuda_sha256=hashlib.sha256(cs.encode()).hexdigest(),parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        non_wgs_preprocessed_controls=builds,actual_segment_native= candidate_native,
        unchanged_nonsegment_cuda_after_removing_host_diagnostics=True,
        limits='CPU instrumented stubs execute actual segment control flow and deferred operand-pointer reads, not WGMMA numerical arithmetic, CUDA compilation, full-grid ordering or speed.')
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
