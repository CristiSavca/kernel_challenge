"""V28 actual-source CPU reduction proof, adapted from reviewed V26.

No CUDA/candidate import. Source equality preserves all unrelated computation;
CPU snippets model payload addressing, contributor ownership and FP32 order only.
"""
import argparse
import ast
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
import build

HERE = Path(__file__).resolve().parent

def sha(b): return hashlib.sha256(b).hexdigest()
def literal(s, name): return build.literal(s, name)
def body(src,needle):
    i=src.index(needle)
    while True:
        a=src.index('{',i)
        if ';' not in src[i:a]:break
        i=src.index(needle,i+len(needle))
    j=a+1;depth=1
    while depth:depth+=(src[j]=='{')-(src[j]=='}');j+=1
    return src[i:j]


def preprocess(src, defs):
    src = re.sub(r"^#include[^\n]*", "", src, flags=re.M)
    proc = subprocess.run(["/usr/bin/clang++", "-E", "-P", "-x", "c++",
                           "-D__CUDA_ARCH__=900", *defs, "-"], input=src,
                          text=True, capture_output=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def source_checks(expected_sha=None):
    source = (HERE / "submission.py").read_text()
    parent = build.PARENT.read_text()
    manifest = json.loads((HERE / "manifest.json").read_text())
    source_sha = sha(source.encode())
    assert source_sha == manifest["source_sha256"]
    if expected_sha:
        assert source_sha == expected_sha
    generated, edits = build.make()
    assert generated == source
    assert edits == json.loads((HERE / "edits.json").read_text())
    restored = source
    for old, new in reversed(edits):
        assert restored.count(new) == 1
        restored = restored.replace(new, old)
    assert restored == parent
    pt, ct = ast.parse(parent), ast.parse(source)
    assert len(pt.body) == len(ct.body)
    changed = []
    for a, b in zip(pt.body, ct.body):
        if ast.dump(a) != ast.dump(b):
            name = a.name if isinstance(a, ast.FunctionDef) else a.targets[0].id
            changed.append(name)
    assert changed == ["MOE_CUDA_SRC", "_ma_ext", "WGS_FIXF"], changed
    oldext = next(ast.get_source_segment(parent, n) for n in pt.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_ma_ext")
    newext = next(ast.get_source_segment(source, n) for n in ct.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_ma_ext")
    assert newext.replace("dl_okc_moe_private_partials_v28_", "dl_okc_moe_kperm3_v12_") == oldext
    p, c = literal(parent, "MOE_CUDA_SRC"), literal(source, "MOE_CUDA_SRC")
    exact = ("void dq4(", "void wgs_epi_tile(", "void wgs_wproducer(",
             "void wgs_xproducer(", "void wgs_gather(", "void route(",
             "int wgs_split(", "void wgs_prefix(", "int wgs_stage_at(",
             "void wgs_range(", "void wgs_ranges(", "int wgs_cta_of(",
             "void wgs_phase(", "moe_fused(Params")
    for name in exact:
        assert body(c, name) == body(p, name), name
    sg, sp = body(c, "void wgs_seg("), body(p, "void wgs_seg(")
    assert sg[:sg.index("  if (slot >= 0)")] == sp[:sp.index("  if (slot >= 0)")]
    assert sg[sg.index("    // kind 3:"):] == sp[sp.index("    // kind 3:"):]
    cf, pf = body(c, "void wgs_flush("), body(p, "void wgs_flush(")
    anchor = '  asm volatile("fence.acq_rel.gpu;'
    assert cf[:cf.index(anchor)] == pf[:pf.index(anchor)]
    anchor = "  if (wt == 0) *fc = 0;"
    assert cf[cf.index(anchor):] == pf[pf.index(anchor):]
    assert "red.global.add.v4.f32" in sp and "red.global.add.v4.f32" not in sg
    assert 'add.rn.ftz.f32 %0, %1, %2;' in body(c, "float v28_sum(")
    assert "make_float4(0.f, 0.f, 0.f, 0.f)" not in cf
    assert "p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)4 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW)" in c
    assert "WGS_FIXF = 4 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16" in source
    assert "--use_fast_math" in newext  # unchanged public build; sum explicitly retains global red.add's FTZ.
    builds = 0
    for name in ("MOE_TINY_DEFINES", "MOE_TINY2_DEFINES", "MOE_MID_DEFINES"):
        for prmt in (0, 1):
            defs = list(literal(parent, name)) + ["-DOKC_MULDEC=1", "-DOKC_DQ7=1",
                    "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3", f"-DOKC_PRMTSC={prmt}"]
            assert preprocess(p, defs) == preprocess(c, defs), name
            builds += 1
    # Explicit non-WGS MAIN/fallback form: every edit is inside OKC_WGS.
    defs = ["-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=0",
            "-DOKC_MULDEC=1", "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3"]
    for prmt in (0, 1):
        assert preprocess(p, defs + [f"-DOKC_PRMTSC={prmt}"]) == preprocess(c, defs + [f"-DOKC_PRMTSC={prmt}"])
        builds += 1
    payload = 4 * 160 * 4 * 4096
    cnt = 2 * 160 * 4
    done = 136 * 16
    assert (payload * 4, cnt * 4, done * 4) == (41943040, 5120, 8704)
    assert manifest["added_scratch_bytes"] == 20971520
    # Native proof below executes actual helpers with actual ybuf prefix.
    return source, c, dict(complete_parent_restoration=True, build_reproducible=True,
        changed_python_nodes=changed, unchanged_device_functions=list(exact),
        per_partial_MMA_and_epilogue_unchanged=True,
        publication_owner_election_and_reset_source_unchanged=True,
        native_public_vector_atomic_replaced=True, non_wgs_preprocessed_builds=builds,
        weights_scales_pdl_build_flags_routing_and_finalizer_unchanged=True,
        payload_bytes=payload*4, counter_bytes=cnt*4, doneAs_bytes=done*4)

def native_test(c,mutant=None):
    definitions='\n'.join(body(c,n) for n in ('int wgs_cost(', 'void wgs_prefix(', 'int wgs_stage_at(',
                  'void wgs_range(', 'int wgs_cta_of(', 'int wgs_split(', 'void wgs_ranges(', 'float* wgs_partial('))
    costs=c[c.index('#ifndef OKC_WGSA'):c.index('#ifndef OKC_WGSPROF')]
    # wgs_cost is already part of this exact constants block.
    definitions=definitions.replace(body(c,'int wgs_cost('),'',1)
    phase='template<bool PB>\n'+body(c,'void wgs_phase(')
    flush=body(c,'void wgs_flush(')
    owner=flush[flush.index('  const int ntt ='):flush.index('  if (wt == 0) *fc = 0;')]
    seg=body(c,'void wgs_seg(')
    store=seg[seg.index('    const int boundary ='):seg.index('    // kind 3:')]
    if mutant=='wrong_boundary':store=store.replace('u == (S.wgr[2 * PB] >> 4) ? 0 : 1','0')
    elif mutant=='lost_range_carry':owner=owner.replace('g0 = g1;','/* lost carry */')
    elif mutant=='read_empty':owner=owner.replace('g0 < end && g1 > begin && g1 > g0','true')
    else:assert mutant is None
    code=r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <map>
#include <set>
#include <vector>
#include <random>
#include <limits>
using std::min; using std::max;
#define __device__
#define __forceinline__ inline
#define __noinline__
#define OKC_DEBUG 0
#define OKC_WGSFIT 0
#define OKC_WGSREORD 1
#define OKC_FT_T 64
constexpr int WGS_SLOTS=160,WGS_FIXW=4096,EMAX=3072,HD=4096;
struct Params{float* ybuf;};
struct Smem{int njF=0,njS=0,jC[128]{},wcp[129]{},wcpB[129]{},wgr[4]{};uint32_t wtot=0,wtotB=0;};
struct{int x;}blockIdx,gridDim;
struct WgsPend{int kind,u,slot,nst;};
struct float4{float x,y,z,w;};
float4 make_float4(float a,float b,float c,float d){return{a,b,c,d};}
uint32_t bits(float x){uint32_t u;memcpy(&u,&x,4);return u;}
float frombits(uint32_t u){float x;memcpy(&x,&u,4);return x;}
float flush_subnormal(float x){uint32_t u=bits(x);return (u&0x7f800000u)==0?frombits(u&0x80000000u):x;}
float v28_sum(float a,float b){volatile float x=flush_subnormal(a)+flush_subnormal(b);return flush_subnormal(x);}
std::set<const float*> readable;uint64_t reads=0;
float4 __ldcg(const float4*p){
 auto*f=reinterpret_cast<const float*>(p);for(int i=0;i<4;i++)assert(readable.count(f+i));reads++;
 return *p;
}
std::vector<float4> outputs;
template<bool PB>void wgs_epi_tile(const Params&,Smem&,int,int,int nt,const float*d){
 assert(nt==(int)outputs.size());outputs.push_back({d[0],d[1],d[2],d[3]});
}
struct Segment{int c,pb,u,s0,s1,slot,nt;};std::vector<Segment> segments;
template<bool PB,int NTT>void wgs_seg(const Params&,Smem&S,unsigned char*,int u,int s0,int s1,int slot,int&seq,WgsPend&){
 assert(NTT==(S.jC[u>>(PB?4:5)]+7)/8);assert(0<=s0&&s0<s1&&s1<=16);
 segments.push_back({blockIdx.x,PB,u,s0,s1,slot,NTT});seq+=s1-s0;
}
template<bool PB>void wgs_flush(const Params&,Smem&,WgsPend&){}
'''+costs+definitions+'\n'+phase+r'''
template<bool PB>void actual_owner(const Params&p,Smem&S,WgsPend pd){
 const int wg=0,wp=0,wt=0,UBL=PB?4:5,j=pd.u>>UBL,ub=pd.u&((1<<UBL)-1);
'''+owner+r'''
}
template<bool PB,int NTT>void actual_store(const Params&p,Smem&S,int u,const float (&d)[8][4]){
 const int wg=0,wt=0;
'''+store+r'''
}
void store_dispatch(int pb,int n,const Params&p,Smem&S,int u,const float (&d)[8][4]){
#define CASE(N) case N:if(pb)actual_store<true,N>(p,S,u,d);else actual_store<false,N>(p,S,u,d);break;
 switch(n){CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) default:assert(false);}
#undef CASE
}
uint64_t cases=0,partials=0,units=0,empty=0,rotated=0,twoedges=0,readvalues=0;
void run_case(const std::vector<int>&counts,int G,bool host_domain){
 gridDim.x=G;Smem S;S.njS=counts.size();for(int j=0;j<(int)counts.size();j++)S.jC[j]=counts[j];
 // Oversubscribed helper-only tests use extra CPU allocation. They do not
 // expand the production G<160 host contract or its forty-MiB bound.
 std::vector<float> memory((size_t)EMAX*HD+(size_t)(2*160+G)*2*4*4096,frombits(0x7fc00000));Params p{memory.data()};
 segments.clear();std::vector<std::pair<int,int>> ranges[2];
 for(int b=0;b<2;b++)ranges[b].resize(G);
 for(int c=0;c<G;c++){
  blockIdx.x=c;wgs_ranges(p,S);int seq=0;
  for(int b=0;b<2;b++){
   int a=S.wgr[2*b],z=S.wgr[2*b+1];ranges[b][c]={a,z};empty+=a==z;
   int split=wgs_split(b,a,z);rotated+=split>a;
   if(b)wgs_phase<true>(p,S,nullptr,a,z,seq);else wgs_phase<false>(p,S,nullptr,a,z,seq);
  }
 }
 for(int pb=0;pb<2;pb++){
  std::map<int,std::vector<Segment>> byunit;
  std::map<int,std::set<int>> slots;
  std::set<int> whole;
  for(auto s:segments)if(s.pb==pb){
   auto [a,z]=ranges[pb][s.c];assert(s.u*16+s.s0>=a&&s.u*16+s.s1<=z);
   if(s.s1-s.s0==16){assert(s.slot==-1);whole.insert(s.u);continue;}
   int edge=s.u==(a>>4)?0:1;
   assert(slots[s.c].insert(edge).second);assert(slots[s.c].size()<=2);
   byunit[s.u].push_back(s);partials++;
  }
  for(auto [c,es]:slots)twoedges+=es.size()==2;
  for(auto &[u,contrib]:byunit){
   units++;assert(!whole.count(u));std::sort(contrib.begin(),contrib.end(),[](auto a,auto b){return a.c<b.c;});
   int target=0,prev=0;std::vector<int> authors;
   for(auto s:contrib){assert(s.s0==prev);prev=s.s1;target+=s.s1-s.s0;authors.push_back(s.c);}
   assert(target==16&&prev==16);
   int first=wgs_cta_of(S,pb,u*16,G),last=wgs_cta_of(S,pb,u*16+15,G);
   assert(first==contrib.front().c&&last==contrib.back().c);
   for(auto s:contrib)assert(s.slot==first);
   int n=contrib[0].nt;readable.clear();float expected[8][4]{};
   for(auto s:contrib){
    blockIdx.x=s.c;S.wgr[2*pb]=ranges[pb][s.c].first;
    float d[8][4]{};const float options[]={1e20f,-1e20f,1.f,frombits(0x00400000),-0.f};
    for(int nt=0;nt<n;nt++)for(int i=0;i<4;i++){
     d[nt][i]=options[(s.c+nt+i)%5];expected[nt][i]=v28_sum(expected[nt][i],d[nt][i]);
    }
    store_dispatch(pb,n,p,S,u,d);
    int edge=u==(ranges[pb][s.c].first>>4)?0:1;
    float*ptr=wgs_partial(p,pb,s.c,edge,0);
    for(int nt=0;nt<n;nt++)for(int i=0;i<4;i++)readable.insert(ptr+nt*512+i);
    if(host_domain)assert(ptr>=(p.ybuf+(size_t)EMAX*HD)&&ptr+n*512<=(p.ybuf+(size_t)EMAX*HD)+4*160*4*4096);
   }
   WgsPend pd{3,u,first,contrib[0].s1-contrib[0].s0};outputs.clear();
   if(pb)actual_owner<true>(p,S,pd);else actual_owner<false>(p,S,pd);
   assert(outputs.size()==(size_t)n);
   for(int nt=0;nt<n;nt++)for(int i=0;i<4;i++)assert(bits((&outputs[nt].x)[i])==bits(expected[nt][i]));
   readvalues+=(uint64_t)contrib.size()*n*4;
   // Publication model: arbitrary A arrival order elects only the final arrival;
   // B's designated first-stage owner waits for all other stage counts.
   for(int order=0;order<3;order++){
    auto arrivals=contrib;if(order==1)std::reverse(arrivals.begin(),arrivals.end());
    if(order==2)std::rotate(arrivals.begin(),arrivals.begin()+arrivals.size()/2,arrivals.end());
    int count=0,owners=0;for(auto s:arrivals){int old=count;count+=s.s1-s.s0;if(old+s.s1-s.s0==16)owners++;}
    assert(count==16&&owners==1);
   }
   int other=16-(contrib[0].s1-contrib[0].s0);int seen=0;
   for(size_t k=1;k<contrib.size();k++){assert(seen<other);seen+=contrib[k].s1-contrib[k].s0;}
   assert(seen==other); // payload remains immutable; only the count resets.
  }
 }
 cases++;
}
int main(){
 {
  std::vector<float>allocation((size_t)EMAX*HD+4*160*4*4096);Params p{allocation.data()};size_t next=0;
  for(int pb=0;pb<2;pb++)for(int c=0;c<160;c++)for(int edge=0;edge<2;edge++)for(int wg=0;wg<4;wg++){
   assert(wgs_partial(p,pb,c,edge,wg)==(p.ybuf+(size_t)EMAX*HD)+next);next+=4096;
  }
  assert(next*4==41943040); // every complete payload slice is disjoint and bounded
 }
 for(int G:{1,7,32,132,159}){
  run_case({},G,true);
  for(int n:{1,8,9,16,17,32,33,64})run_case({n},G,true);
  for(int J:{2,3,8,31,32,33,64,65,88}){
   std::vector<int>v(J);for(int j=0;j<J;j++)v[j]=1+(j*17)%64;run_case(v,G,true);
  }
 }
 // Empty intermediate CTAs are unreachable in normal positive-job grids with
 // these costs and G<160, but the owner guard is tested directly beyond that
 // host bound, in isolated CPU storage, without a GPU admissibility claim.
 run_case({1},513,false);run_case({64},528,false);
 assert(empty>0&&rotated>0&&twoedges>0&&units>0&&reads>0);
 assert(bits(v28_sum(frombits(0x00400000),0.f))==0);
 assert(bits(v28_sum(frombits(0x80400000),frombits(0x80000000)))==0x80000000u);
 assert(bits(v28_sum(frombits(0x00800001),-frombits(0x00800000)))==0);
 assert(v28_sum(v28_sum(1e20f,-1e20f),1.f)!=v28_sum(1e20f,v28_sum(-1e20f,1.f)));
 printf("{\"cases\":%llu,\"partial_contributions\":%llu,\"split_units\":%llu,\"empty_ranges\":%llu,\"rotated_ranges\":%llu,\"two_boundary_ctas\":%llu,\"float4_loads\":%llu,\"verified_float_values\":%llu}\n",
 (unsigned long long)cases,(unsigned long long)partials,(unsigned long long)units,(unsigned long long)empty,
 (unsigned long long)rotated,(unsigned long long)twoedges,(unsigned long long)reads,(unsigned long long)readvalues);
}
'''
    with tempfile.TemporaryDirectory(prefix='v28-cpu-') as td:
        p=Path(td);(p/'test.cpp').write_text(code)
        cp=subprocess.run(['/usr/bin/clang++','-std=c++17','-O1','-fno-fast-math',str(p/'test.cpp'),'-o',str(p/'test')],capture_output=True,text=True,timeout=40)
        assert cp.returncode==0,cp.stderr
        r=subprocess.run([str(p/'test')],capture_output=True,text=True,timeout=90)
        if mutant:
            assert r.returncode!=0,('negative control unexpectedly passed',mutant)
            return {'mutant':mutant,'caught':True,'returncode':r.returncode}
        assert r.returncode==0,(r.returncode,r.stderr)
    result=json.loads(r.stdout)
    result['native_source_sha256']=sha(code.encode())
    result['limits']='Exact range/phase/store/owner source executed with CPU memory and FP32-add shim; GPU publication, progress, atomics and WGMMA are not emulated.'
    return result


def run(expected_sha=None):
    source, cuda, checks = source_checks(expected_sha)
    return dict(status="PASS", source_sha256=sha(source.encode()),
                manifest_sha256=sha((HERE / "manifest.json").read_bytes()),
                verifier_sha256=sha(Path(__file__).read_bytes()),
                source_checks=checks, native_schedule_and_payload=native_test(cuda),
                negative_controls=[native_test(cuda, m) for m in
                                   ("wrong_boundary", "lost_range_carry", "read_empty")],
                memory=dict(private_payload_bytes=41943040, added_scratch_bytes=20971520,
                            initialization="Original scratch allocation, larger by20MiB; no per-replay private-payload clearing."),
                limits=["No CUDA compilation, GPU memory-order/progress, numerical-tolerance or speed proof.",
                        "Exact actual range/phase/store/owner snippets execute onCPU forWG0/WT0; affine complete slices check otherwarpgroups/lanes.",
                        "Explicit RN/FTZ FP32 addition uses a CPU scalar shim matching the public global reduction policy; changed CTA-order association is not bit-equivalent to native vector-atomic arrival order.",
                        "The zero-range helper tests do not establish arbitrary all-remote global routing support; inherited dispatch remains unchanged."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha")
    parser.add_argument("--output", type=Path, default=HERE / "cpu_result.json")
    args = parser.parse_args()
    result = run(args.expected_sha)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

