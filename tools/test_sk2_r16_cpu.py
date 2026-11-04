#!/usr/bin/env python3
"""Fixed R16 source/packing/native host proof. No CUDA compilation or GPU proof."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

import build_sk2_r16_candidate as builder
from test_dense_stage_directory_cpu import Tensor, Torch, block, key, literal
from test_dense_escape_global_cpu import ring_model
from test_sk2_r16_router_native_cpu import run as router_run

ROOT = builder.ROOT
Tensor.permute = lambda self, *axes: Tensor(self.a.transpose(axes))
Tensor.reshape = lambda self, *shape: Tensor(self.a.reshape(*shape))


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def env_from(source):
    tree = ast.parse(source)
    names = {"bfp_SK2_CONFIGS", "bfp_SK2_PICK", "bfp_SK2_K", "bfp_SK2_SRC", "SK2B", "SK2B_UN_TOKR",
             "SK2B1", "SK2B_O_TOKR", "_bfp__sk2_src_unfused", "bfp__sk2_src", "bfp__sk2_geom",
             "bfp_sk2_tileunits", "bfp_sk2_stream", "bfp_sk2_table", "bfp_sk2_r16_preflight"}
    env = {"torch": Torch}
    exec(compile(ast.Module(body=[n for n in tree.body if key(n) in names], type_ignores=[]), "<actual-helpers>", "exec"), env)
    return env


def conformance(source):
    assert source == builder.build(write=False)
    restored = source
    for old, new, count in reversed(builder.REPLACEMENTS):
        assert restored.count(new) == count, new[:80]
        restored = restored.replace(new, old)
    assert sha(restored) == builder.PARENT_SHA and restored == builder.PARENT.read_text()
    parent_tree, tree = ast.parse(restored), ast.parse(source)
    changed = {"bfp_SK2_SRC", "bfp_CPP_SRC", "bfp__get_ext", "bfp_SK2_CONFIGS", "bfp__sk2_geom", "bfp__sk2_src",
               "bfp_fused_pooler_router", "bfp_prepare", "prepare", "_uv_pooler"}
    added = {"bfp_sk2_r16_preflight"}
    dump = lambda n: ast.dump(n, include_attributes=False)
    assert [dump(n) for n in parent_tree.body if key(n) not in changed] == [dump(n) for n in tree.body if key(n) not in changed | added]
    untouched = {}
    for n in parent_tree.body:
        name = key(n)
        if isinstance(n, ast.Assign) and name and name.endswith("SRC") and name not in changed:
            try:
                value = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, str):
                assert literal(tree, name) == value
                untouched[name] = sha(value)
    old, new = env_from(restored), env_from(source)
    oldsrc, newsrc = old["bfp__sk2_src"](True), new["bfp__sk2_src"](True)
    assert "@@" not in newsrc
    # Non-target dispatch and complete UV-O/BND generated wrappers must remain byte-identical.
    line_map = lambda s: {re.search(r"cfg == (\d+) && tokr == (\d+)", x).groups(): x for x in s.splitlines()
                          if "cfg == " in x and "launch_sk2<" in x and "&& tokr ==" in x}
    before, after = line_map(oldsrc), line_map(newsrc)
    assert before.keys() == after.keys()
    for identity in before:
        if int(identity[0]) not in (2, 3, 5, 6, 7):
            assert before[identity] == after[identity], identity
    for start in ("int64_t sk2b(", "void sk2t("):
        assert block(oldsrc, oldsrc.index(start)) == block(newsrc, newsrc.index(start))
    # Same ring/barrier/escape/MMA code apart from declared geometry and router finalizer.
    oldcuda, newcuda = old["bfp_SK2_SRC"], new["bfp_SK2_SRC"]
    for start, end in (("  if (warp >= NW + 1)", "  // ---------------- consumers"),
                       ("  // ---------------- consumers", "  if constexpr (FUSE) {"),
                       ("  // ---------------- epilogue", "// R16_ADMISSION_BEGIN")):
        if start == "  // ---------------- epilogue":
            po = oldcuda[oldcuda.index(start):oldcuda.index("template <int MT", oldcuda.index(start))]
            pn = newcuda[newcuda.index(start):newcuda.index(end)]
            assert po.strip() == pn.strip()
        else:
            assert oldcuda[oldcuda.index(start):oldcuda.index(end, oldcuda.index(start))] == newcuda[newcuda.index(start):newcuda.index(end, newcuda.index(start))]
    assert re.findall(r'asm volatile\("griddepcontrol\.(?:wait|launch_dependents);[^\n]+', oldcuda) == re.findall(r'asm volatile\("griddepcontrol\.(?:wait|launch_dependents);[^\n]+', newcuda)
    # Build flag syntax is unchanged; cache name and export list are the only extension builder changes.
    ext_old = next(n for n in parent_tree.body if key(n) == "bfp__get_ext")
    ext_new = next(n for n in tree.body if key(n) == "bfp__get_ext")
    flags = lambda node: [ast.dump(k.value) for n in ast.walk(node) if isinstance(n, ast.Call) for k in n.keywords if k.arg == "extra_cuda_cflags"]
    assert flags(ext_old) == flags(ext_new)
    return {"restored_parent_sha256": sha(restored), "patch_count": len(builder.REPLACEMENTS),
            "changed_top_level_nodes": sorted(changed), "added_top_level_nodes": sorted(added),
            "unaffected_embedded_sources": untouched}, new, newsrc


def native_admission(source, generated):
    cuda = env_from(source)["bfp_SK2_SRC"]
    host = cuda[cuda.index("// R16_ADMISSION_BEGIN"):cuda.index("// R16_ADMISSION_END") + len("// R16_ADMISSION_END")]
    start = cuda.index("template <int MT", cuda.index("// R16_ADMISSION_END"))
    launch = cuda[start:cuda.index("    CUtensorMap xm", start)]
    launch += "  };\n" + next(x for x in cuda.splitlines() if "if (TOKR == 8 || M < TOKR) go" in x) + "\n}\n"
    wrapper = block(generated, generated.index("int64_t sk2_r16_preflight("))
    preamble = r'''
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
namespace at {struct Tensor{};}
using __nv_bfloat16=uint16_t; struct Sk2Bnd{}; struct TArgs{};
#define TORCH_CHECK(c,...) do {if(!(c))throw std::runtime_error("check");}while(0)
#define C10_CUDA_CHECK(c) TORCH_CHECK((c)==0)
struct cudaFuncAttributes {int numRegs=64;size_t localSizeBytes=0,sharedSizeBytes=384;int maxDynamicSharedSizeBytes=233088,maxThreadsPerBlock=1024;};
enum {cudaDevAttrMaxSharedMemoryPerBlockOptin=1,cudaFuncAttributeMaxDynamicSharedMemorySize=2,cudaDevAttrMultiProcessorCount=3};
int failure=0, query=0, blocks_value=2, local_value=0, maxthreads_value=1024, maxdyn_value=233088;
std::vector<int> sets; std::vector<std::pair<int,size_t>> occupancies;
int status(){++query;return query==failure?7:0;}
int cudaGetDevice(int*p){*p=0;return status();}
int cudaDeviceGetAttribute(int*p,int kind,int){*p=kind==1?233472:132;return status();}
template<class K>int cudaFuncGetAttributes(cudaFuncAttributes*p,K){*p=cudaFuncAttributes{};p->localSizeBytes=local_value;p->maxThreadsPerBlock=maxthreads_value;p->maxDynamicSharedSizeBytes=maxdyn_value;return status();}
template<class K>int cudaFuncSetAttribute(K,int,int value){sets.push_back(value);return status();}
template<class K>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*p,K,int th,size_t sm){*p=blocks_value;occupancies.push_back({th,sm});return status();}
template<int MT,int TOKR,int NT,int NWR,int WSK,int UPW,int NS,bool COMP,bool HL,int NACC,int CL,typename OT,bool PK,bool FUSE,bool BND>void sk2_kernel(){}
'''
    main = r'''
int main(){
  // Exactly14 generated records (12 distinct pointers), with each family's actual pmax.
  int total=0;for(int cfg:std::vector<int>{2,3,5,6,7})total+=sk2_r16_preflight(cfg,cfg<5?56:88);
  assert(total==14&&occupancies.size()==14&&sets.size()==14);
  for(int x:sets)assert(x==233088);
  int failed=0;
  for(int e=1;e<=6;++e){failure=e;query=0;try{sk2_r16_preflight(7,88);}catch(const std::runtime_error&){++failed;}}
  assert(failed==6);failure=0;
  local_value=4;try{sk2_r16_preflight(7,88);}catch(const std::runtime_error&){++failed;}local_value=0;
  blocks_value=1;try{sk2_r16_preflight(7,88);}catch(const std::runtime_error&){++failed;}blocks_value=2;
  maxthreads_value=32;try{sk2_r16_preflight(7,88);}catch(const std::runtime_error&){++failed;}maxthreads_value=1024;
  maxdyn_value=1;try{sk2_r16_preflight(7,88);}catch(const std::runtime_error&){++failed;}maxdyn_value=233088;
  assert(failed==10);
  // Exhausted logging never disables resource queries or checking; storage remains bounded.
  for(int i=0;i<200;++i)sk2_r16_preflight(7,88+i*2);
  assert(sk2_r16_admission::stopped&&sk2_r16_admission::bytes<=8192);
  size_t lastbytes=sk2_r16_admission::bytes;int oldquery=query;local_value=4;
  try{sk2_r16_preflight(7,10000);}catch(const std::runtime_error&){++failed;}
  assert(failed==11&&query==oldquery+6&&sk2_r16_admission::bytes==lastbytes);
  std::cout<<"{\"generated_resource_records\":"<<total<<",\"negative_admission_cases\":"<<failed<<",\"log_bytes\":"<<lastbytes<<"}\n";
}
'''
    with tempfile.TemporaryDirectory(prefix="okc-r16-host-") as td:
        cpp, exe = Path(td) / "proof.cpp", Path(td) / "proof"
        cpp.write_text(preamble + host + "\n" + launch + wrapper + main)
        built = subprocess.run(["clang++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)], capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        ran = subprocess.run([str(exe)], capture_output=True, text=True)
        assert ran.returncode == 0, ran.stderr
    assert len(ran.stderr.encode()) <= 8192 and ran.stderr.count("R16R_STOP") == 1 and ran.stderr.count("R16R_SCHEMA") == 1
    lines = [line for line in ran.stderr.splitlines() if line.startswith("R16R ")]
    records = []
    for row in lines[:14]:
        fields = dict(re.findall(r"(\w+)=([^ ]+)", row))
        records.append({k: [int(v) for v in value.split(",")] for k, value in fields.items()})
    identities = [tuple(r["t"]) for r in records]
    assert len(set(identities)) == 12
    expected = []
    # Eight runtime families: ordinary pooler16 compatibility + seven fixed layouts.
    for tokr,wsk,upw,ns,nacc,fuse,n,minimum in ((16,8,1,3,2,0,4096,1),
            (8,8,1,4,2,1,4096,2),(16,8,1,3,2,1,4096,2),(32,8,1,2,2,1,4096,1),
            (32,4,2,2,1,0,4096,2),(16,4,2,3,1,0,2304,2),(32,4,2,2,1,0,2304,2),(8,4,2,5,1,0,2304,2)):
        for pk in ((1,) if tokr==8 else (1,0)):
            t=(2 if tokr==32 else 1,tokr,2,1,wsk,upw,ns,1,0,nacc,1,2,pk,fuse,0)
            expected.append((t,n,56 if n==4096 else 88,minimum))
    assert sorted((tuple(r["t"]),r["x"][1],r["x"][3],r["min"][0]) for r in records)==sorted(expected)
    assert sum(r["min"] == [1] for r in records) == 4
    assert all(r["err"] == [0]*6 for r in records)
    return {**json.loads(ran.stdout), "unique_compiled_pointers":12, "pointers": records, "actual_resources": "UNKNOWN: CUDA APIs mocked on CPU"}


def packing(env):
    # Execute unchanged actual pack-stream/table helpers with NumPy tensor operations.
    units = 0
    configurations = []
    for cfg in env["bfp_SK2_CONFIGS"]:
        name, layer, n, R, nt, nwr, wsk, upw, nacc, cl = cfg
        if name not in {"pooler", "pooler32_r144", "qkv8_r176", "qkv16_r176", "qkv32_r208"}: continue
        k = env["bfp_SK2_K"][layer]
        tub = Tensor(np.arange((n//8)*(k//64), dtype=np.int64).reshape(n//8,k//64,1))
        stream = env["bfp_sk2_stream"](tub,n,R,nt,nwr,wsk,upw).a
        assert stream.size == tub.a.size and np.array_equal(np.sort(stream.reshape(-1)), tub.a.reshape(-1))
        for st in range(stream.shape[0]):
            for cta in range(n//R):
                for warp in range(nwr*wsk):
                    for u in range(upw):
                        for tile in range(nt):
                            rowtile=cta*(R//8)+(warp//wsk)*nt+tile
                            ku=st*wsk*upw+(warp%wsk)*upw+u
                            assert stream[st,cta,warp,u,tile,0] == rowtile*(k//64)+ku
                            units += 1
        configurations.append({"name":name,"units":int(stream.size),"shape":list(stream.shape)})
    # Actual tileunits then actual compressed stream on every sign/code/low-byte pattern.
    n,k=96,512
    rows=np.arange(n)[:,None];cols=np.arange(k)[None,:]
    bits=(((54+(cols%8)) | (((rows+cols)&1)*128))<<8 | ((rows*73+cols*37)&255)).astype(np.uint16)
    tu=env["bfp_sk2_tileunits"](Tensor(bits)).a
    low=(tu&255).astype(np.uint8); code=(((tu>>15)<<3)|(((tu>>8)&127)-54)).astype(np.uint8)
    cbytes=np.concatenate([code[...,0:4]|(code[...,4:8]<<4),code[...,8:12]|(code[...,12:16]<<4)],axis=-1)
    packed=np.concatenate([low.reshape(n//8,k//64,512),cbytes.reshape(n//8,k//64,256)],axis=-1)
    decoded_words=0
    for wsk,upw in ((8,1),(4,2)):
        stream=env["bfp_sk2_stream"](Tensor(packed),n,16,2,1,wsk,upw).a
        for st in range(stream.shape[0]):
            for cta in range(n//16):
                for warp in range(wsk):
                    for u in range(upw):
                        for tile in range(2):
                            data=stream[st,cta,warp,u,tile]
                            lo=data[:512].reshape(32,16);cb=data[512:].reshape(32,8)
                            cc=np.concatenate([cb[:,:4]&15,cb[:,:4]>>4,cb[:,4:]&15,cb[:,4:]>>4],axis=-1)
                            got=((((cc.astype(np.uint16)>>3)<<7)|(54+(cc&7).astype(np.uint16)))<<8)|lo
                            rowtile=cta*2+tile;ku=st*8+warp*upw+u
                            assert np.array_equal(got,tu[rowtile,ku]);decoded_words+=got.size
    # New R16 tables retain every original row/K/delta. Empty, sparse/ties, dense, and boundary rows.
    table_cases=0;entries=0
    for mode in ("empty","sparse","dense"):
        if mode=="empty": rr=np.zeros(0,np.int64);kk=rr.copy()
        elif mode=="sparse": rr=np.arange(96,dtype=np.int64).repeat(5);kk=np.tile(np.array([0,63,64,255,511]),96)
        else: rr=np.arange(32,dtype=np.int64).repeat(512);kk=np.tile(np.arange(512,dtype=np.int64),32)
        delta=((rr+1)*0.125+(kk%31)*0.00390625).astype(np.float32)
        comp={"rows":Tensor(rr),"cols":Tensor(kk),"delta":Tensor(delta)}
        for nrows in (32,96):
            tab,pmax=env["bfp_sk2_table"](comp,nrows,16)
            actual=[]
            for cta,t in enumerate(tab.a):
                count=int(t[0,0]); assert count<pmax and pmax%2==0
                assert np.all(np.diff(t[1:count+1,0]&65535)>=0)
                for item in t[1:count+1]:actual.append((cta*16+(int(item[0])>>16),int(item[0])&65535,int(item[1])))
            expected=list(zip(rr[rr<nrows].tolist(),kk[rr<nrows].tolist(),delta[rr<nrows].view(np.int32).tolist()))
            assert sorted(actual)==sorted(expected);table_cases+=1;entries+=len(actual)
    return {"actual_helper_stream_units":units,"configurations":configurations,"decoded_bf16_words":decoded_words,
            "table_cases":table_cases,"table_entries":entries,"codec_source": "byte-identical to97af; actual stream/table helpers executed via NumPy shim"}


def decode_native(parent):
    cuda=literal(ast.parse(parent),"bfp_CUDA_SRC")
    start=cuda.index("__device__ __forceinline__ void decode8(")
    decode=block(cuda,start).replace("__device__ __forceinline__ ","")
    body=r'''
#include <cstdint>
#include <cassert>
#include <iostream>
uint32_t lop3_sel(uint32_t a,uint32_t b,uint32_t c){return (a&~c)|(b&c);}
uint32_t prmt(uint32_t a,uint32_t b,uint32_t s){uint64_t x=((uint64_t)b<<32)|a;uint32_t out=0;for(int i=0;i<4;++i)out|=((x>>(8*((s>>(4*i))&7)))&255)<<(8*i);return out;}
'''+decode+r'''
int main(){uint64_t n=0;for(unsigned base=0;base<=120;++base)for(unsigned code=0;code<16;++code)for(unsigned low=0;low<256;++low){
uint32_t lw=low*0x01010101u,nw=(code|(code<<4))*0x01010101u,r[4];decode8(lw,lw,nw,base*0x01010101u,r);
unsigned word=(((code>>3)<<7)|(base+(code&7)))<<8|low;
for(int j=0;j<4;++j){assert((r[j]&65535)==word&&(r[j]>>16)==word);n+=2;}}
std::cout<<n<<"\n";}
'''
    with tempfile.TemporaryDirectory(prefix="okc-r16-codec-") as td:
        exe=Path(td)/"proof"
        built=subprocess.run(["clang++","-std=c++17","-O2","-x","c++","-o",str(exe),"-"],input=body,text=True,capture_output=True)
        assert built.returncode==0,built.stderr
        ran=subprocess.run([str(exe)],capture_output=True,text=True)
        assert ran.returncode==0,ran.stderr
    return {"actual_decode8_bf16_words":int(ran.stdout),"intrinsics":"CPU bitwise models of unchanged lop3_sel and prmt"}


def lifecycle(source,env):
    class Table:
        shape=(256,56,2)
    class Ext:
        calls=[]
        def sk2_r16_preflight(self,ci,pmax):
            self.calls.append((ci,pmax));return {2:7,3:2,5:2,6:2,7:1}[ci]
    ext=Ext();env["bfp__get_ext"]=lambda:ext
    configs={n:(i,None,Table(),56,None,r) for i,(n,l,r,*rest) in enumerate(env["bfp_SK2_CONFIGS"]) if i in (2,3,5,6,7)}
    st={"sm90":True,"sk2":configs}
    env["bfp_sk2_r16_preflight"](st);assert st["r16_preflight_complete"] and len(ext.calls)==5
    failures=0
    for mode in ("missing","sm80","bad_pmax","bad_shape","api"):
        s={"sm90":True,"sk2":dict(configs)}
        if mode=="missing":del s["sk2"]["pooler"]
        if mode=="sm80":s["sm90"]=False
        if mode=="bad_pmax":s["sk2"]["pooler"]=(2,None,Table(),55,None,4096)
        if mode=="bad_shape":s["sk2"]["pooler"]=(2,None,Table(),58,None,4096)
        if mode=="api":
            def fail(*args):raise RuntimeError("mock API failure")
            ext.sk2_r16_preflight=fail
        try:env["bfp_sk2_r16_preflight"](s)
        except RuntimeError:failures+=1
        else:raise AssertionError(mode)
        assert "r16_preflight_complete" not in s
    tree=ast.parse(source)
    bfp=next(n for n in tree.body if key(n)=="bfp_prepare")
    text=ast.get_source_segment(source,bfp)
    assert text.index("bfp_sk2_prepare(weights, state)")<text.index("bfp_sk2_r16_preflight(state)")<text.index("bfp__selftest(weights, state)")<text.index("bfp__autotune(weights, state)")
    prepare=next(n for n in tree.body if key(n)=="prepare")
    outer=next(n for n in prepare.body if isinstance(n,ast.If) and "r16_preflight_complete" in ast.unparse(n))
    pre_call=next(i for i,n in enumerate(prepare.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="_bfp_prepare")
    assert prepare.body[pre_call+1] is outer
    outercode=compile(ast.Module(body=[outer],type_ignores=[]),"<actual-outer-failure-gate>","exec")
    for state in ({"bfp":None},{"bfp":{}},{"bfp":{"r16_preflight_complete":False}}):
        try:exec(outercode,{"st":state})
        except RuntimeError:failures+=1
        else:raise AssertionError("swallowed backend failure")
    exec(outercode,{"st":{"bfp":{"r16_preflight_complete":True}}})
    # Execute the actual catch wrapper, then the actual outer failure gate.
    wrapper=next(n for n in tree.body if key(n)=="_bfp_prepare")
    def failed_backend(weights):raise RuntimeError("mock resource preflight failure")
    class Weight:
        def contiguous(self):return self
    caught_env={"bfp_prepare":failed_backend}
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]),"<actual-catch-wrapper>","exec"),caught_env)
    caught={}
    with contextlib.redirect_stdout(io.StringIO()):
        caught_env["_bfp_prepare"]({k:Weight() for k in ("qkv_gate","o","pooler","unpooler","router")},caught)
    assert caught["bfp"] is None
    try:exec(outercode,{"st":caught})
    except RuntimeError:failures+=1
    else:raise AssertionError("actual wrapper swallowed preflight failure")
    return {"positive_queries":5,"negative_lifecycle_cases":failures,"placement":"after immutable packing, before selftest/tuner; outer failure gate outside catch"}


def rings(records):
    checked=0
    for record in records:
        mt,tokr,nt,nwr,wsk,upw,ns,comp,hl,nacc,cl,ob,pk,fuse,bnd=record["t"]
        m,n,k,pmax,threads,grid,dyn=record["x"]
        nw=nwr*wsk;nesc=min(ns,4);xby=wsk*upw*tokr*128;wby=nw*upw*nt*768;wbyr=nw*2048
        sst=(xby+(max(wby,wbyr) if fuse else wby)+1023)//1024*1024
        expected=1024+ns*sst+16*ns+32+nesc*((16*mt*(16+1)+3)//4*4)*4+pmax*8+128
        assert dyn==expected and threads==(nw+1+nesc)*32 and grid==n//16
        assert ns*sst>=wsk*(16*mt)*(16+((40-16)%32))*4
        for physical_m in range(1,tokr+1):
            xbytes=wsk*upw*physical_m*128
            assert xbytes<=xby and xby+wby<=sst
            if fuse:assert xby+wbyr<=sst
        stages=k//(wsk*upw*64)+(2 if fuse else 0)
        for seed in range(16):assert ring_model(nw+nesc,ns,stages,seed);checked+=1
    assert any(not ring_model(12,4,18,seed,expected_delta=-1) for seed in range(100))
    return {"interleaving_models":checked,"early_reuse_negative_control":"detected","all_physical_M_checked":True,
            "scope":"source-identical handbacks with changed constant geometry; CPU model assumes GPU barrier semantics"}


def run():
    source=builder.CANDIDATE.read_text();digest=sha(source)
    conf,env,generated=conformance(source)
    native=native_admission(source,generated)
    result={"candidate":str(builder.CANDIDATE.relative_to(ROOT)),"candidate_sha256":digest,"parent_sha256":builder.PARENT_SHA,
            "conformance":conf,"host_admission":native,"packing":packing(env),"codec":decode_native(builder.PARENT.read_text()),
            "lifecycle":lifecycle(source,env),"ring":rings(native["pointers"]),"router":router_run(builder.CANDIDATE,digest),
            "status":"PASS","scope":"CPU source/helper/native mock proof only; CUDA compilation, actual resources, numerical GPU correctness and speed UNKNOWN"}
    assert sha(builder.CANDIDATE.read_text())==digest
    return result


if __name__=="__main__":
    output=ROOT/"experiments/iterations/20261104-sk2-r16-cpu.json"
    result=run();output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"candidate_sha256":result["candidate_sha256"],"output":str(output.relative_to(ROOT))}))
