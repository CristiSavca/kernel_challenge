#!/usr/bin/env python3
"""Actual-source scope/role/guard checks for one WGS redistribution candidate.

Native CPU stubs are not CUDA compilation, tensor math or scheduling evidence.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import test_wgs_wait2_cpu as wait2

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "muse_moe_w4a16/candidates/native_wgs_wait2_v12.py"
CANDIDATE = ROOT / "muse_moe_w4a16/candidates/native_wgs_wait2_reg112_v13.py"
OUT = ROOT / "experiments/iterations/20261104-wgs-reg112-source-cpu.json"
EXPECTED_PARENT = "97ffccc30a90ce8400cddef800d932889822eaf876fbdedaec18b1013e8f66b6"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(cpp):
    with tempfile.TemporaryDirectory(prefix="okc-wgs-reg112-") as td:
        exe = str(Path(td) / "audit")
        r = subprocess.run(["clang++", "-x", "c++", "-std=c++17", "-O2", "-o", exe, "-"],
                           input=cpp, text=True, capture_output=True)
        assert r.returncode == 0, r.stderr
        r = subprocess.run([exe], text=True, capture_output=True)
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)


def between(src, start, end):
    a = src.index(start)
    b = src.index(end, a)
    return src[a:b]


def role_native(src, preprocessed):
    role = between(src, "#if OKC_WGS\n  if (tid >= NT) {", "#endif\n#if FAST1_ON")
    role = role.removeprefix("#if OKC_WGS\n")
    actions = re.findall(r'asm volatile\("setmaxnreg\.(dec|inc)\.sync\.aligned\.u32 (\d+);\\n" ::: "memory"\);', role)
    assert actions == [("dec", "32"), ("inc", "112")]
    role = re.sub(r'asm volatile\("setmaxnreg\.(dec|inc)\.sync\.aligned\.u32 (\d+);\\n" ::: "memory"\);',
                  lambda m: f'quota({-1 if m[1] == "dec" else 1}, {m[2]});', role)
    assert "asm" not in wait2.comments(role)
    nw = re.search(r"constexpr int NW = .*?;", preprocessed).group()
    ntl = re.search(r"constexpr int NTL = .*?;", preprocessed).group()
    cpp = r'''
#include <cassert>
#include <cstdio>
#include <vector>
struct Params{};struct Smem{};
''' + nw + "\n" + ntl + r'''
struct Event {int action,arg;};
std::vector<Event> events;int current_tid;
void quota(int action,int n){assert(events.empty());events.push_back({action,n});}
void bar(int n){events.push_back({2,n});}
#define WS_JOIN() bar(NTL)
void wgs_wproducer(const Params&,Smem&,unsigned char*){events.push_back({3,0});WS_JOIN();}
void wgs_xproducer(const Params&,Smem&,unsigned char*){events.push_back({4,0});bar(NT+32);WS_JOIN();}
void consumer_body(){events.push_back({5,0});bar(NT+32);WS_JOIN();}
void run(int tid){Params p;Smem S;unsigned char raw[128];void*xbuf=raw;
''' + role + r'''
    consumer_body();
  }
}
int main(){
  static_assert(NT==512&&NTL==640);
  unsigned long long cases=0;int decr=0,incr=0,weight=0,xprod=0,idle=0,bar2=0,bar7=0;
  // Role decisions have no M/job inputs. Repeat across representative absent,
  // one-job and multi-job labels to expose accidental state retention in stubs.
  for(int jobs: {0,1,2,32,79,88}){
    (void)jobs;decr=incr=weight=xprod=idle=bar2=bar7=0;
    for(int tid=0;tid<NTL;tid++){
      events.clear();current_tid=tid;run(tid);cases++;
      assert(!events.empty());int b2=0,b7=0;
      for(auto e:events)if(e.action==2){if(e.arg==640)b2++;else if(e.arg==544)b7++;else assert(false);}
      assert(b2==1);bar2+=b2;bar7+=b7;
      if(tid<512){assert(events[0].action==1&&events[0].arg==112);assert(events[1].action==5);assert(b7==1);incr++;}
      else{
        assert(events[0].action==-1&&events[0].arg==32);decr++;
        if(tid<544){assert(events[1].action==3&&b7==0);weight++;}
        else if(tid<576){assert(events[1].action==4&&b7==1);xprod++;}
        else{assert(events.size()==2&&b7==0);idle++;}
      }
    }
    assert(incr==512&&decr==128&&weight==32&&xprod==32&&idle==64);
    assert(bar2==640&&bar7==544);
  }
  printf("{\"thread_cases\":%llu,\"consumers\":%d,\"producer_dec_threads\":%d,\"weight_threads\":%d,\"x_threads\":%d,\"idle_threads\":%d,\"barrier2\":%d,\"barrier7\":%d}\n",cases,incr,decr,weight,xprod,idle,bar2,bar7);
}
'''
    return native(cpp)


def guard_native(host):
    exprs = [re.search(r"TORCH_CHECK\(("+prefix+r"[^,]+),", host).group(1)
             for prefix in (r"a0\.numRegs", r"a0\.localSizeBytes", r"occ0 >=")]
    cpp = r'''
#include <cstdio>
#include <cassert>
#include <initializer_list>
struct Attr{int numRegs;unsigned long localSizeBytes;};
int main(){int cases=0,passed=0;
for(int r0:{80,88,89,95,96,97,104})for(int r1:{80,88,89,95,96,97,104})
for(int l0:{0,4,32})for(int l1:{0,4,32})for(int occ0:{0,1,2})for(int occ1:{0,1,2}){
Attr a0{r0,(unsigned long)l0},a1{r1,(unsigned long)l1};
bool actual = ''' + " && ".join("("+x+")" for x in exprs) + r''';
bool expected=r0==96&&r1==96&&l0==0&&l1==0&&occ0>=1&&occ1>=1;
assert(actual==expected);cases++;passed+=actual;
}
printf("{\"resource_cases\":%d,\"admitted_combinations\":%d}\n",cases,passed);
}
'''
    return native(cpp)


def main():
    initial_sha = sha(CANDIDATE)
    assert sha(PARENT) == EXPECTED_PARENT
    pt = ast.parse(PARENT.read_bytes());ct = ast.parse(CANDIDATE.read_bytes())
    ps = wait2.literal(pt,"CUDA_SRC");cs = wait2.literal(ct,"CUDA_SRC")
    def key(n):
        return n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
    pn = {key(n):n for n in pt.body if key(n)};cn = {key(n):n for n in ct.body if key(n)}
    assert pn.keys()==cn.keys()
    for k in pn:
        if k not in ("CUDA_SRC","_ext"):assert ast.dump(pn[k])==ast.dump(cn[k]),k
    assert ast.get_source_segment(CANDIDATE.read_text(),cn["_ext"]).replace("okc_moe_wgs_wait2_reg112_v13_","okc_moe_wgs_wait2_v12_")==ast.get_source_segment(PARENT.read_text(),pn["_ext"])
    assert wait2.segment(cs)==wait2.segment(ps)

    old_ntl=next(line for line in ps.splitlines() if line.startswith("constexpr int NTL ="))
    new_ntl=between(cs,"#if OKC_WGS\nconstexpr int NTL =", "\nconstexpr int NRB =")
    assert new_ntl.endswith(old_ntl+"\n#endif")
    assert 'constexpr int NTL = NT + 128;' in new_ntl
    old_role=between(ps,"#if OKC_WGS\n  if (tid >= NT) {","#endif\n#if FAST1_ON")+"#endif"
    new_role=between(cs,"#if OKC_WGS\n  if (tid >= NT) {","#endif\n#if FAST1_ON")+"#endif"
    close='\n#if OKC_WGS\n  }  // consumer112 region; producers have already returned\n#endif'
    assert cs.count(close)==1
    assert cs.index(close)>cs.index("DBG(5)")
    old_host=between(ps,'#if OKC_WGS\n    TORCH_CHECK(e0 == cudaSuccess && e1 == cudaSuccess, "wait2', '\n#endif')+'\n#endif'
    new_host=between(cs,'#if OKC_WGS\n    TORCH_CHECK(e0 == cudaSuccess && e1 == cudaSuccess, "wait2', '\n#endif')+'\n#endif'
    normalized_host=new_host.replace('"wait2 reg112 ', '"wait2 ')
    normalized_host=normalized_host.replace('OKC_WGS_WAIT2_REG112 resources intended_initial=96 producer=32 consumer=112 ', 'OKC_WGS_WAIT2 resources ')
    for extra in (
        '    TORCH_CHECK(a0.numRegs == 96 && a1.numRegs == 96, "wait2 requires initial96 for both CTA-local register pools");\n',
        '    TORCH_CHECK(a0.localSizeBytes == 0 && a1.localSizeBytes == 0, "wait2 rejects local storage in either specialization");\n'):
        assert normalized_host.count(extra)==1
        normalized_host=normalized_host.replace(extra,'')
    assert normalized_host==old_host, 'Unexpected host query/launch alteration'
    assert new_host.count('cudaOccupancyMaxActiveBlocksPerMultiprocessor(')==2
    assert new_host.count('cudaFuncGetAttributes(')==2
    for guard in ('a0.numRegs == 96 && a1.numRegs == 96','a0.localSizeBytes == 0 && a1.localSizeBytes == 0','occ0 >= 1 && occ1 >= 1'):
        assert cs.index(guard)<cs.index('if (g_okc_pdl)')<cs.index('cudaLaunchKernelEx(')
        assert cs.index(guard)<cs.index('<<<(int)grid, okc::NTL, sm, st>>>')
    restored=cs.replace(new_ntl,old_ntl).replace(new_role,old_role).replace(close,'').replace(new_host,old_host)
    parent_comment='// The unchanged segment below retains its parent WAIT2 resource comment (576 threads,96 registers).\n// This revision launches640 threads with initial96 and requested consumer112; compiler overlap remains unproved.\n'
    assert restored.count(parent_comment)==1
    restored=restored.replace(parent_comment,'')
    assert restored==ps, 'Unexpected CUDA change outside approved regions'
    assert cs.count('setmaxnreg.')==2

    controls=0
    for name in ('TINY_DEFINES','TINY2_DEFINES','MID_DEFINES','MAIN_FALLBACK_DEFINES'):
        for mul in (0,1):
            for prmt in (0,1):
                defs=[*wait2.literal(pt,name),f'-DOKC_MULDEC={mul}',f'-DOKC_PRMTSC={prmt}','-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']
                assert wait2.preprocess(ps,defs)==wait2.preprocess(cs,defs),(name,mul,prmt)
                controls+=1
    defs=[*wait2.literal(pt,'MAIN_DEFINES'),'-DOKC_MULDEC=1','-DOKC_PRMTSC=1','-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']
    pre=wait2.preprocess(cs,defs)
    roles=role_native(cs,pre)
    guards=guard_native(new_host)
    trace=wait2.native_trace(wait2.segment(cs))
    previous=json.loads((ROOT/'experiments/iterations/20261104-wgs-wait2-source-cpu.json').read_text())
    assert previous['source_sha256']==EXPECTED_PARENT
    assert trace==previous['actual_segment_native']
    assert sha(CANDIDATE)==initial_sha
    result=dict(source_sha256=initial_sha,parent_sha256=EXPECTED_PARENT,
                verifier_sha256=sha(Path(__file__)),cuda_sha256=hashlib.sha256(cs.encode()).hexdigest(),
                non_wgs_preprocessed_controls=controls,actual_role_prefix_native=roles,
                actual_host_predicates_native=guards,actual_segment_native=trace,
                unchanged_cuda_after_exact_scope_restoration=True,
                unchanged_python_except_cuda_and_namespace=True,
                full_segment_byte_identity=True,gpu_used=False,
                limits='Native stubs verify actual role conditions, quota/barrier call order, host admission predicates and inherited deferred-AF pointer lifetime. They do not compile CUDA/PTX, emulate WGMMA numerics, validate actual allocations/hints, simulate CUDA progress or measure speed.')
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
