#!/usr/bin/env python3
"""Actual producer trace/host source proof for the fixed distance3 hint."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re

import build_wgs_l2_prefetch_candidate as builder
import test_wgs_shared_a_cpu as common
from test_wgs_wait2_cpu import literal,preprocess

ROOT=builder.ROOT
OUT=ROOT/'experiments/iterations/20261104-wgs-l2-prefetch-cpu.json'
def sha(x):return hashlib.sha256(x).hexdigest()

def conformance(parent,candidate):
    pt,ct=ast.parse(parent),ast.parse(candidate)
    def key(n):return n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
    pn={key(n):n for n in pt.body if key(n)};cn={key(n):n for n in ct.body if key(n)}
    assert pn.keys()==cn.keys()
    for k in pn:
        if k not in ('CUDA_SRC','_ext'):assert ast.dump(pn[k])==ast.dump(cn[k]),k
    ext=ast.get_source_segment(candidate,cn['_ext']).replace('okc_moe_wgs_l2_prefetch_v17_','okc_moe_kperm3_v9_').replace('verbose="-DOKC_WGS=1" in defs,  # retain WGS compiler evidence','verbose=False,')
    assert ext==ast.get_source_segment(parent,pn['_ext'])
    ps,cs=literal(pt,'CUDA_SRC'),literal(ct,'CUDA_SRC')
    assert cs.count(builder.PREFETCH)==1
    restored=cs.replace(builder.PREFETCH,'').replace(builder.HOST+'\n','').replace('#if OKC_WGS\n    okc_l2_prefetch_resources(sm, e0, e1);\n#endif\n','')
    assert restored==ps,'source outside declared hint/resource delta changed'
    assert common.function(cs,'wgs_wproducer').replace(builder.PREFETCH,'')==common.function(ps,'wgs_wproducer')
    builds=0
    for name in ('TINY_DEFINES','TINY2_DEFINES','MID_DEFINES','MAIN_FALLBACK_DEFINES'):
        for mul in (0,1):
            for prmt in (0,1):
                flags=[*literal(pt,name),f'-DOKC_MULDEC={mul}',f'-DOKC_PRMTSC={prmt}','-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']
                assert preprocess(ps,flags)==preprocess(cs,flags)
                builds+=1
    return dict(restored_parent_cuda_sha256=sha(restored.encode()),unchanged_python_nodes=len(pn)-2,non_wgs_preprocessed_builds=builds,packing_dispatch_MMA_math_and_all_other_device_functions_preserved=True),ps,cs

def translated_producer(src,name):
    text='void '+common.function(src,'wgs_wproducer')
    text=text.replace('wgs_wproducer(',name+'(',1)
    # Byte-address pointers are represented as uint64_t. No data is dereferenced:
    # this preserves the actual offset expressions without allocating a 864MiB bank.
    text=text.replace('const unsigned char* base','const uint64_t base').replace('const unsigned char* src','const uint64_t src').replace('const unsigned char* psrc','const uint64_t psrc')
    ar=r'asm volatile\("mbarrier\.arrive\.expect_tx[^\n]*\n?\s*::"r"\((.*?)\), "r"\((.*?)\) : "memory"\);'
    text,n=re.subn(ar,lambda m:f'cpu_arrive({m[1]}, {m[2]}, pb, i, gs);',text)
    assert n==1,'arrival translation'
    cp=r'asm volatile\("cp\.async\.bulk\.shared[^\n]*\n\s*::"r"\((.*?)\), "l"\((.*?)\), "r"\((.*?)\), "r"\((.*?)\) L2H_ARG : "memory"\);'
    text,n=re.subn(cp,lambda m:f'cpu_copy({m[1]}, {m[2]}, {m[3]}, {m[4]}, pb, i, gs);',text)
    assert n==1,'copy translation'
    pf=r'asm volatile\("cp\.async\.bulk\.prefetch\.L2\.global[^\n]*\n\s*:: "l"\((.*?)\), "r"\((.*?)\) : "memory"\);'
    text,n=re.subn(pf,lambda m:f'cpu_hint({m[1]}, {m[2]}, pb, i, gs);',text)
    assert n==int('cp.async.bulk.prefetch.L2.global' in src)
    assert 'asm volatile' not in text
    return text

def native_trace(ps,cs):
    split='int '+common.function(ps,'wgs_split')
    code=r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#define OKC_WGSREORD 1
constexpr int UNIT_BYTES=36864,WGS_NS=3,BW_F=0,BW_E=3;
struct Params{uint64_t wAs=0x100000000ull,wBs=0x200000000ull;};
struct Smem{int wgr[4];int jE[256];uint64_t bars[24];};
struct{int x;}threadIdx;
struct Event{int kind,pb,i,gs;uint64_t a,b,c,d;bool operator==(const Event&o)const{return kind==o.kind&&pb==o.pb&&i==o.i&&gs==o.gs&&a==o.a&&b==o.b&&c==o.c&&d==o.d;}};
std::vector<Event>events;Smem*S0;unsigned char*R0;
uint64_t smem_u32(const void*p){uintptr_t u=(uintptr_t)p,b=(uintptr_t)S0->bars,r=(uintptr_t)R0;
 if(u>=b&&u<b+sizeof(S0->bars))return 0x1000+(u-b);assert(u>=r&&u<r+3*UNIT_BYTES);return 0x10000+(u-r);}
void join(){events.push_back({0,0,0,0,0,0,0,0});}
#define WS_JOIN() join()
void wgs_wait(uint64_t*b,uint32_t phase,int tag){events.push_back({1,0,0,0,smem_u32(b),phase,(uint64_t)tag,0});}
void cpu_arrive(uint64_t bar,uint32_t bytes,int pb,int i,int gs){events.push_back({2,pb,i,gs,bar,bytes,0,0});}
void cpu_copy(uint64_t dst,uint64_t src,uint32_t bytes,uint64_t bar,int pb,int i,int gs){events.push_back({3,pb,i,gs,dst,src,bytes,bar});}
void cpu_hint(uint64_t src,uint32_t bytes,int pb,int i,int gs){events.push_back({4,pb,i,gs,src,bytes,0,0});}
'''+split+'\n'+translated_producer(ps,'parent_producer')+'\n'+translated_producer(cs,'candidate_producer')+r'''
uint64_t cases=0,copies=0,hints=0,waits=0,empty=0,wraps=0,crossjobs=0,nonzero=0;
void check(int a,int n,int b,int m,int perm){
 Params p;Smem s{};alignas(128)unsigned char ring[3*UNIT_BYTES];S0=&s;R0=ring;
 s.wgr[0]=a;s.wgr[1]=a+n;s.wgr[2]=b;s.wgr[3]=b+m;
 for(int j=0;j<256;j++)s.jE[j]=perm==0?j%32:perm==1?(31-j%32):((j*17+7)%32);
 threadIdx.x=0;events.clear();parent_producer(p,s,ring);auto parent=events;
 events.clear();candidate_producer(p,s,ring);auto child=events;
 std::vector<Event>actual;for(auto&e:child)if(e.kind!=4)actual.push_back(e);
 assert(parent==actual);std::vector<Event>stage[2];for(auto&e:parent){if(e.kind==3)stage[e.pb].push_back(e);if(e.kind==1)waits++;}
 assert(stage[0].size()==(size_t)n&&stage[1].size()==(size_t)m);
 int hinted[2]={};size_t seenCopies=0;
 for(size_t z=0;z<child.size();z++){auto&e=child[z];if(e.kind==3){seenCopies++;copies++;continue;}if(e.kind!=4)continue;
  assert(z>0&&child[z-1].kind==3&&child[z-1].pb==e.pb&&child[z-1].i==e.i);
  assert(e.i+3<(int)stage[e.pb].size());auto&future=stage[e.pb][e.i+3];
  assert(e.a==future.b&&e.b==future.c&&e.b==UNIT_BYTES&&!(e.a&127));
  uint64_t base=e.pb?p.wBs:p.wAs,sz=uint64_t(32)*(e.pb?16:32)*16*UNIT_BYTES;
  assert(e.a>=base&&e.a+e.b<=base+sz);
  int ubits=e.pb?4:5;if((e.gs>>(4+ubits))!=(future.gs>>(4+ubits)))crossjobs++;
  if(future.gs<e.gs)wraps++;hinted[e.pb]++;hints++;
 }
 assert(hinted[0]==std::max(0,n-3)&&hinted[1]==std::max(0,m-3));
 empty+=(n==0)+(m==0);cases++;
 // Inactive lanes still join, then return without adding copies/waits/hints.
 for(int lane:{1,31}){threadIdx.x=lane;events.clear();candidate_producer(p,s,ring);assert(events.size()==1&&events[0].kind==0);nonzero++;}
}
int main(){
 for(int a=0;a<64;a++)for(int n=0;n<=128;n++)check(a,n,(a*7)%128,(n*13)%129,n%3);
 // Multiple job boundaries, greatest legal job-array region, expert IDs0/31,
 // empty phase transitions and many wraps of unchanged shared ring parity.
 for(int perm=0;perm<3;perm++)for(int a:{0,1,15,16,127,255,511,512,513,1023,4095,65519})
 for(int n:{0,1,2,3,4,7,15,16,17,63,127,511,1025}){
  check(a,n,(a/2),n,perm);check(a,n,(a/2),0,perm);check(a,0,(a/2),n,perm);
 }
 assert(wraps>0&&crossjobs>0);
 printf("%llu %llu %llu %llu %llu %llu %llu %llu\n",(unsigned long long)cases,(unsigned long long)copies,(unsigned long long)hints,(unsigned long long)waits,(unsigned long long)empty,(unsigned long long)wraps,(unsigned long long)crossjobs,(unsigned long long)nonzero);
}
'''
    out,_=common.compile_run(code)
    names=['range_pair_cases','identical_real_copies','correct_future_hints','identical_waits','empty_phase_ranges','rotated_wrap_hints','cross_job_hints','nonproducer_lane_returns']
    return dict(zip(names,map(int,out.split())))

def host_mock(cs):
    code=inspect.getsource(common.host_mock).replace('def host_mock(cs):','def adapted(cs):').replace('okc_shared_a_resources','okc_l2_prefetch_resources').replace('194304','231168').replace('SSAR ','LPFR ')
    ns=dict(common.__dict__);exec(code,ns);return ns['adapted'](cs)

def run(candidate=builder.CANDIDATE,expected_sha=None):
    candidate=Path(candidate);raw=candidate.read_bytes();digest=sha(raw)
    if expected_sha:assert digest==expected_sha
    parent=builder.PARENT.read_bytes();assert sha(parent)==builder.PARENT_SHA
    assert builder.build().encode()==raw
    scope,ps,cs=conformance(parent.decode(),raw.decode());trace=native_trace(ps,cs)
    negatives=[]
    mutations=[('missing_tail_guard',lambda s:s.replace('if (i + 3 < g1 - g0)','if (true)')),
      ('wrong_rotated_wrap',lambda s:s.replace('pgs = g0 + (pgs - g1)','pgs = g0')),
      ('current_expert_for_future_job',lambda s:s.replace('S.jE[pj]','S.jE[j]')),
      ('changed_expected_bytes',lambda s:s.replace('"r"(UNIT_BYTES) : "memory");','"r"(UNIT_BYTES / 2) : "memory");',1)),
      ('changed_real_copy_bytes',lambda s:s.replace('"l"(src), "r"(UNIT_BYTES),','"l"(src), "r"(UNIT_BYTES / 2),')),
      ('wrong_future_distance',lambda s:s.replace('int pgs = gs + 3','int pgs = gs + 2'))]
    for name,mutate in mutations:
        producer=common.function(cs,'wgs_wproducer');mutated=mutate(producer);assert mutated!=producer
        try:native_trace(ps,cs.replace(producer,mutated))
        except common.NativeCheckFailed:negatives.append(name)
        else:raise AssertionError('negative accepted: '+name)
    result=dict(status='PASS',source_sha256=digest,parent_sha256=builder.PARENT_SHA,cuda_sha256=sha(cs.encode()),conformance=scope,native_producer_trace=trace,host_resources=host_mock(cs),negative_controls=negatives,
      limits='Native execution of actual producer/split control flow and address expressions uses uint64 byte-address and CUDA-operation shims. It proves emitted requests, bounds and ordering in these finite cases; not CUDA compilation, hardware asynchrony, hint execution/effectiveness, cache behavior, GPU tensor correctness or speed. Full source restoration proves the arithmetic/packing and other device paths unchanged; no new MMA or lifecycle simulator is used.')
    assert sha(candidate.read_bytes())==digest
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,default=builder.CANDIDATE);ap.add_argument('--expected-sha');ap.add_argument('--output',type=Path,default=OUT)
    a=ap.parse_args();r=run(a.candidate,a.expected_sha);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(status=r['status'],source_sha256=r['source_sha256'],report_sha256=sha(a.output.read_bytes()),output=str(a.output)),indent=2))
