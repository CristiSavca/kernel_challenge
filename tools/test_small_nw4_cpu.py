#!/usr/bin/env python3
"""Exact-source CPU checks for the isolated small NW4 candidate; no CUDA/Torch."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from audit_small_moe_residency import namespace, packed_addresses, resources, schedule
from test_wgs_kperm3_cpu import Tensor, literal

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/"muse_moe_w4a16/candidates/native_wgs_kperm3_v9.py"
CANDIDATE=ROOT/"muse_moe_w4a16/candidates/native_small_nw4_v11.py"
OUT=ROOT/"experiments/iterations/20261104-small-nw4-source-cpu.json"


def extract(source, start):
    a=source.index(start); brace=source.index("{",a); depth=1; b=brace+1
    while depth:
        depth+=(source[b]=="{")-(source[b]=="}");b+=1
    return source[a:b]


def cpp_preprocess(source, defines):
    source=re.sub(r"^#include.*$", "",source,flags=re.M)
    return subprocess.run(["clang++","-E","-P","-x","c++","-D__CUDA_ARCH__=900",*defines,"-"],
        input=source,text=True,capture_output=True,check=True).stdout


def native_inner(source, tree):
    dq=extract(source,"__device__ __forceinline__ void dq4(")
    load=extract(source,"template <bool PB>\n__device__ __forceinline__ void load_x_regs(")
    # The two #if branches have different loop counts; raw brace counting
    # cannot delimit this function before preprocessing selects a branch.
    wa=source.index("template <bool PB>\n__device__ void skinny_ring_ws(")
    wb=source.index("\n__device__ __forceinline__ void ws_issue(",wa)
    ws=source[wa:wb]
    a=ws.index("#if OKC_NW == 16"); b=ws.index("\n#if OKC_WSB\n",a)
    inner=ws[a:b]
    scdk=next(x for x in source.splitlines() if x.startswith("#define SCDK("))
    prefix=r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#define __device__
#define __forceinline__ inline
#define OKC_NW 4
#define OKC_DQ7 1
#define OKC_NACC 4
#define OKC_XNOLUT 0
#define OKC_XNODQ 0
#define OKC_XNOMMA 0
constexpr int NCH=16,UNIT_CODES=32768,HD=4096,ID=4096;
using __nv_bfloat16=uint16_t;
struct uint4 {uint32_t x,y,z,w;}; struct uint2 {uint32_t x,y;};
uint4 make_uint4(uint32_t a,uint32_t b,uint32_t c,uint32_t d){return {a,b,c,d};}
template<class T>T __ldg(const T*p){return *p;} template<class T>T __ldcg(const T*p){return *p;}
uint32_t __funnelshift_r(uint32_t a,uint32_t b,unsigned s){s&=31;return s?(a>>s)|(b<<(32-s)):a;}
uint32_t __funnelshift_l(uint32_t a,uint32_t b,unsigned s){s&=31;return s?(a<<s)|(b>>(32-s)):a;}
uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t ctrl){uint64_t ab=a|(uint64_t(b)<<32);uint32_t r=0;
 for(int i=0;i<4;i++){unsigned q=(ctrl>>(4*i))&15,v=(ab>>(8*(q&7)))&255;
 if(q&8)v=(v&128)?255:0;r|=v<<(8*i);}return r;}
float from_bits(uint32_t u){float x;memcpy(&x,&u,4);return x;}
uint32_t bits(float x){uint32_t u;memcpy(&u,&x,4);return u;}
float fp8(unsigned x){int e=(x>>3)&15,m=x&7;float v=e==15&&m==7?NAN:e?std::ldexp(float(8+m),e-10):std::ldexp(float(m),-9);return x&128?-v:v;}
uint32_t bmul2(uint32_t a,uint32_t b){uint32_t r=0;for(int i=0;i<2;i++){
 float x=from_bits(((a>>(16*i))&65535)<<16)*from_bits(((b>>(16*i))&65535)<<16);
 r|=(bits(x)>>16)<<(16*i);}return r;}
struct Params {const uint16_t*x;const uint16_t*inter;};
struct Smem {int jC[88]{},jS[88]{};short ent[128]{};uint32_t lut[256]{};};
alignas(16) unsigned char packed[36864],expected_codes[65536],expected_scales[4096];
alignas(16) uint16_t input_x[16*4096],inter_x[16*4096];
int global_warp,global_lane,global_count,global_pb,mmacalls;uint64_t comparisons=0;
void mma_bf16(float*,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
 int c=mmacalls/4,st=mmacalls%4,g=global_lane/4,t=global_lane%4;uint32_t aa[4]={a0,a1,a2,a3};
 float mags[8]={0,.5f,1,1.5f,2,3,4,6};
 for(int reg=0;reg<4;reg++)for(int upper=0;upper<2;upper++){
 int h=reg&1,j0=reg>=2?2:0,row=g+8*h,col=global_warp*1024+c*64+t*16+st*4+j0+upper;
 unsigned code=expected_codes[row*4096+col],sf=expected_scales[row*256+col/16];
 float q=(code&8)?-mags[code&7]:mags[code&7];uint16_t want=bits(std::ldexp(q*fp8(sf),-62))>>16;
 uint16_t got=aa[reg]>>(16*upper);
 assert(got==want||(std::isnan(from_bits(uint32_t(got)<<16))&&std::isnan(from_bits(uint32_t(want)<<16))));comparisons++;
 }
 uint32_t bb[2]={b0,b1};for(int pair=0;pair<2;pair++){
 int col=global_warp*1024+c*64+t*16+st*4+pair*2;uint32_t want=0;
 if(g<global_count){int row=global_pb?3+g:7-g;auto*xs=global_pb?inter_x:input_x;want=xs[row*4096+col]|(uint32_t(xs[row*4096+col+1])<<16);}
 assert(bb[pair]==want);comparisons++;
 }mmacalls++;
}
'''
    body="\nvoid compute(const Params&p,Smem&S,int warp,int lane,bool pb){uint4 x0[2*NCH];if(pb)load_x_regs<true>(p,S,0,lane,warp,x0);else load_x_regs<false>(p,S,0,lane,warp,x0);const unsigned char*slot=packed;const uint32_t k16=16;\n"+inner+"\n}\n"
    main=r'''
int main(){Params p{input_x,inter_x};Smem s;s.jS[0]=3;for(int r=0;r<8;r++)s.ent[3+r]=(7-r)*8;
 for(int i=0;i<16*4096;i++){input_x[i]=uint16_t(i);inter_x[i]=uint16_t(i^0x1234);}
 for(int i=0;i<256;i++){uint32_t q=bits(std::ldexp(fp8(i),64))>>16;s.lut[i]=q|(q<<16);}
 int patterns=0;while(fread(packed,1,sizeof packed,stdin)==sizeof packed){
 assert(fread(expected_codes,1,sizeof expected_codes,stdin)==sizeof expected_codes);
 assert(fread(expected_scales,1,sizeof expected_scales,stdin)==sizeof expected_scales);
 for(int pb=0;pb<2;pb++)for(int cnt=0;cnt<=8;cnt++)for(int w=0;w<4;w++)for(int l=0;l<32;l++){
 global_pb=pb;global_count=cnt;global_warp=w;global_lane=l;mmacalls=0;s.jC[0]=cnt;compute(p,s,w,l,pb);assert(mmacalls==64);
 }patterns++;}assert(patterns==4);printf("%llu\n",(unsigned long long)comparisons);}
'''
    ns=namespace(tree);rows=Tensor(np.arange(16)); coords=np.arange(65536,dtype=np.uint32).reshape(16,4096)
    sfcoords=np.arange(4096,dtype=np.uint32).reshape(16,256)
    runs=[]
    with tempfile.TemporaryDirectory(prefix="okc-nw4-native-") as td:
        for mul in (0,1):
            pos=ns["_stored_pos_mul"]() if mul else ns["_stored_pos"]()
            for prmt in (0,1):
                data=[]
                for pattern in range(4):
                    code=((coords>>(4*pattern))&15).astype(np.uint8)
                    sf=((sfcoords>>(8*(pattern%2)))&255).astype(np.uint8)
                    if prmt:sf=(8+sfcoords%119).astype(np.uint8)
                    original=Tensor(code[:,::2]|(code[:,1::2]<<4));scale=Tensor(sf)
                    unit=ns["_unit_order"](ns["_pack_codes"](original,rows,pos),ns["_pack_scales"](scale,rows)).a
                    data.extend((unit.tobytes(),code.tobytes(),sf.tobytes()))
                exe=str(Path(td)/f"inner-{mul}-{prmt}")
                cpp=prefix+f"\n#define OKC_MULDEC {mul}\n#define OKC_PRMTSC {prmt}\n"+scdk+"\n"+dq+"\n"+load+body+main
                comp=subprocess.run(["clang++","-x","c++","-std=c++17","-O2","-o",exe,"-"],input=cpp,text=True,capture_output=True)
                assert comp.returncode==0,comp.stderr
                r=subprocess.run([exe],input=b"".join(data),capture_output=True)
                assert r.returncode==0,r.stderr.decode()
                runs.append(dict(MULDEC=mul,PRMTSC=prmt,operand_comparisons=int(r.stdout)))
    return runs


def main():
    parent=ast.parse(PARENT.read_bytes());content=CANDIDATE.read_bytes();candidate=ast.parse(content)
    psrc=literal(parent,"CUDA_SRC");src=literal(candidate,"CUDA_SRC")
    changed={"CUDA_SRC","TINY_DEFINES","TINY2_DEFINES","MID_DEFINES","_ext"}
    def key(n):return n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
    pn={key(n):n for n in parent.body if key(n)};cn={key(n):n for n in candidate.body if key(n)}
    assert pn.keys()==cn.keys()
    for name in pn:
        if name not in changed:assert ast.dump(pn[name])==ast.dump(cn[name]),name
    for name in ("TINY_DEFINES","TINY2_DEFINES","MID_DEFINES"):
        expect=tuple("-DOKC_NW=4" if v=="-DOKC_NW=8" else v for v in literal(parent,name))
        assert literal(candidate,name)==expect[:1]+("-DOKC_NS=2",)+expect[1:]
    extp=ast.dump(pn["_ext"]);extc=ast.dump(cn["_ext"])
    assert extc.replace("okc_moe_small_nw4_v11_","okc_moe_kperm3_v9_")==extp
    builds=0
    for name in ("MAIN_DEFINES","TINY_DEFINES","TINY2_DEFINES","MID_DEFINES","MAIN_FALLBACK_DEFINES"):
        for mul in (0,1):
            for prmt in (0,1):
                if name=="MAIN_DEFINES" and not prmt:continue
                defs=[*literal(parent,name),f"-DOKC_MULDEC={mul}",f"-DOKC_PRMTSC={prmt}","-DOKC_DQ7=1","-DOKC_NS=3","-DOKC_NSF=3"]
                assert cpp_preprocess(psrc,defs)==cpp_preprocess(src,defs),(name,mul,prmt)
                builds+=1
    # Actual unchanged source portions are the lifecycle proof's conformance link.
    for start in ("__device__ void ws_helper(","__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {",
                  "template <bool PB>\n__device__ __forceinline__ void load_x_regs(","__device__ void route("):
        assert extract(src,start)==extract(psrc,start),start
    tailstart="  // programmatic dependent launch: the next kernel may start"
    tailend="\n}  // namespace okc"
    assert src[src.index(tailstart):src.index(tailend)].replace("NT*4-column","1024-column")==psrc[psrc.index(tailstart):psrc.index(tailend)]
    assert src.index("occ0 >= 2 && occ1 >= 2")<src.index("grid *= 2")<src.index("if (g_okc_pdl)")
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ0, okc::moe_fused<false>, okc::NTL, sm)" in src
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ1, okc::moe_fused<true>, okc::NTL, sm)" in src
    result=dict(source_sha256=hashlib.sha256(content).hexdigest(),cuda_sha256=hashlib.sha256(src.encode()).hexdigest(),
        parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),unchanged_preprocessed_builds=builds,
        actual_native_inner=native_inner(src,candidate),packed_addresses=packed_addresses(candidate),
        resources=resources(src),schedule=schedule(),
        limits="CPU executes actual inner loader/unpack/scale/MMA operand construction; it does not emulate tensor math, asynchronous GPU ordering, ptxas resources or performance.")
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("schedule",)},indent=2))


if __name__=="__main__":main()
