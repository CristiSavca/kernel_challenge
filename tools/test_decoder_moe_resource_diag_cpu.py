#!/usr/bin/env python3
"""Compile the actual MoE diagnostic host snippets with CPU CUDA API mocks."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import decoder_moe_resource_diag_fragment as fragment

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/candidates/muse_decode_layer/97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2.py"
OUT = ROOT / "experiments/iterations/20261104-decoder-moe-resource-host-cpu.json"
FIELDS = "NW NS WS TINY SPLIT MULDEC DQ7 PRMTSC FINPF WSBLK SKSC2 FTSC2 L2H CPF".split()

PREAMBLE = r'''
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <climits>
#include <cstring>
#include <thread>
#include <vector>
namespace okc { constexpr int NTL=288; template<bool B> void moe_fused() {} }
struct cudaFuncAttributes {
  int numRegs=0; size_t localSizeBytes=0, sharedSizeBytes=0;
  int maxDynamicSharedSizeBytes=0, maxThreadsPerBlock=0;
};
enum { cudaDevAttrMultiProcessorCount=1 };
static int calls[4]={}, errors[4]={};
static bool extreme=false, expected_big=false;
int cudaGetDevice(int* p) { ++calls[0]; *p=extreme?INT_MAX:0; return errors[0]; }
int cudaDeviceGetAttribute(int* p,int,int) { ++calls[1]; *p=extreme?INT_MAX:132; return errors[1]; }
template<class K> int cudaFuncGetAttributes(cudaFuncAttributes* p,K kern) {
  ++calls[2]; assert(kern==(expected_big?okc::moe_fused<true>:okc::moe_fused<false>));
  p->numRegs=extreme?INT_MAX:149; p->localSizeBytes=extreme?SIZE_MAX:0;
  p->sharedSizeBytes=extreme?SIZE_MAX:1024;
  p->maxDynamicSharedSizeBytes=extreme?INT_MAX:231424;
  p->maxThreadsPerBlock=extreme?INT_MAX:288; return errors[2];
}
template<class K> int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* p,K kern,int threads,size_t) {
  ++calls[3]; assert(threads==okc::NTL);
  assert(kern==(expected_big?okc::moe_fused<true>:okc::moe_fused<false>));
  *p=extreme?INT_MAX:1; return errors[3];
}
static int g_okc_pdl=0;
'''

CALL_PREFIX = r'''
void exact_call(int M=16,size_t sm=193792,long long grid=132,int tensor_device=0) {
  struct {int M;} p{M};
  struct X {int device;int get_device()const{return device;}} x{tensor_device};
'''

MAIN = r'''
int main(int argc,char**argv) {
  assert(argc==2); const char* mode=argv[1];
  using namespace moe_resource_diag;
  exact_call(); assert(calls[0]==0 && count==0);
  if (!std::strcmp(mode,"disabled")) {
    okc_set_diag(1,1,0); exact_call(); assert(calls[0]==0 && !enabled);
  } else {
    if (!std::strcmp(mode,"extreme")) extreme=true;
    if (!std::strncmp(mode,"error",5)) errors[mode[5]-'0']=7;
    if (!std::strcmp(mode,"all_error")) for(int&i:errors)i=7;
    okc_set_diag(1,extreme?LLONG_MAX:1,320);
    if (extreme) {
      expected_big=true; exact_call(INT_MAX,SIZE_MAX,LLONG_MAX,INT_MAX);
      assert(exhausted && count==0);
    } else if (!std::strcmp(mode,"threads")) {
      std::vector<std::thread> ts;
      for(int i=0;i<32;++i) ts.emplace_back([]{exact_call();});
      for(auto&t:ts)t.join();
      assert(calls[0]==1 && calls[2]==1 && count==1);
    } else {
      exact_call(); assert(count==1);
      int saved[4]; std::memcpy(saved,calls,sizeof calls);
      exact_call(1); exact_call(8,193792,264); g_okc_pdl=1; exact_call();
      assert(!std::memcmp(saved,calls,sizeof calls));
      okc_set_diag(0,2,320); exact_call(32); assert(!std::memcmp(saved,calls,sizeof calls));
      okc_set_diag(1,2,320); exact_call(); assert(!std::memcmp(saved,calls,sizeof calls));
      if (!std::strcmp(mode,"big")) exact_call(32);
      if (!std::strcmp(mode,"dynamic")) exact_call(16,193793);
      if (!std::strcmp(mode,"device")) exact_call(16,193792,132,1);
      assert(!std::memcmp(saved,calls,sizeof calls));
      if (!std::strcmp(mode,"big") || !std::strcmp(mode,"dynamic") || !std::strcmp(mode,"device")) assert(exhausted);
    }
    const size_t saved_bytes=used; const int saved_count=count;
    int saved[4]; std::memcpy(saved,calls,sizeof calls);
    if(exhausted) {
      okc_set_diag(0,3,320); okc_set_diag(1,3,100000);
      exact_call(); assert(used==saved_bytes && count==saved_count && !std::memcmp(saved,calls,sizeof calls));
    }
    assert(limit==320 && used<=limit && count<=1);
  }
  std::printf("{\"bytes\":%zu,\"records\":%d,\"exhausted\":%d,\"queries\":[%d,%d,%d,%d]}\n",used,count,(int)exhausted.load(),calls[0],calls[1],calls[2],calls[3]);
}
'''


def run():
    parent = PARENT.read_text()
    parsed = ast.parse(parent)
    values = {}
    wanted = {"MOE_CUDA_SRC", "MOE_TINY_DEFINES", "MOE_TINY2_DEFINES", "MOE_MID_DEFINES", "_MA_SFX_IF_PRMT", "DQ7", "MOE_MULDEC", "_MA_MAIN_NS"}
    for n in parsed.body:
        if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in wanted:
            values[n.targets[0].id] = ast.literal_eval(n.value)
    assert values["DQ7"] and values["MOE_MULDEC"] and values["_MA_MAIN_NS"]==3
    changed = fragment.apply_moe_fragment(parent)
    assert fragment.reverse_moe_fragment(changed)==parent
    compile(changed, "<diagnostic-fragment>", "exec")
    preprocessor_source = re.sub(r"^\s*#include[^\n]*$", "", values["MOE_CUDA_SRC"], flags=re.M)
    reports=[]
    with tempfile.TemporaryDirectory(prefix="okc-moe-resource-host-") as temp:
        temp=Path(temp)
        for name in ("MOE_TINY_DEFINES", "MOE_TINY2_DEFINES", "MOE_MID_DEFINES"):
            for prmt in (False,True):
                flags=list(values[name])+list(values["_MA_SFX_IF_PRMT"] if prmt else ())
                flags += ["-DOKC_DQ7=1", "-DOKC_MULDEC=1", "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3"]
                pp=subprocess.run(["clang++","-E","-dM","-x","c++",*flags,"-"],input=preprocessor_source,text=True,capture_output=True)
                assert pp.returncode==0,pp.stderr
                macros=dict(re.findall(r"^#define (OKC_\w+) (.+)$",pp.stdout,re.M))
                effective={key:int(macros["OKC_"+key]) for key in FIELDS}
                defines="\n".join("#define OKC_"+k+" "+str(v) for k,v in effective.items())+"\n#define OKC_BIGM 32\n"
                cpp=temp/"host.cpp"; binary=temp/"host"
                cpp.write_text(defines+PREAMBLE+fragment.MOE_HOST+CALL_PREFIX+fragment.MOE_CALL+"\n}\n"+MAIN)
                compiled=subprocess.run(["clang++","-std=c++17","-Wall","-Wextra","-Werror","-Wno-unused-const-variable",str(cpp),"-o",str(binary)],capture_output=True,text=True)
                assert compiled.returncode==0,compiled.stderr
                for mode in ("disabled","same","big","dynamic","device","error0","error1","error2","error3","all_error","extreme","threads"):
                    result=subprocess.run([str(binary),mode],capture_output=True,text=True)
                    assert result.returncode==0,(name,prmt,mode,result.stderr)
                    obj=json.loads(result.stdout)
                    assert obj["bytes"]==len(result.stderr.encode())<=320
                    lines=result.stderr.splitlines()
                    rows=[line for line in lines if line.startswith("MOER g=")]
                    assert len(rows)==obj["records"]<=1
                    assert result.stderr.count("MOER_SCHEMA")==obj["records"]
                    assert result.stderr.count("MOER_STOP")==obj["exhausted"]
                    if mode.startswith("error") or mode=="all_error":
                        row=rows[0]
                        statuses={"error0":"7,-1,0,0","error1":"0,7,0,0","error2":"0,0,7,-1","error3":"0,0,0,7","all_error":"7,-1,7,-1"}
                        assert row.endswith("e="+statuses[mode]),row
                        if mode in ("error0","all_error"): assert " d=0,-1 " in row,row
                        if mode in ("error2","all_error"): assert "a=-1,-1,-1,-1,-1,-1," in row,row
                    if rows:
                        flag_text=rows[0].split(" f=")[1].split(" x=")[0]
                        assert list(map(int,flag_text.split(",")))==list(effective.values())
                    obj.update(build=name,prmt=prmt,mode=mode,row_bytes=max((len(row)+1 for row in rows),default=0),effective_flags=effective)
                    reports.append(obj)
        # Compile the same helper/call with non-TINY/non-WS compile-time controls.
        # No query is compiled into these configurations, and Python assigns budget0.
        for control in ("MAIN","FALLBACK"):
            cpp=temp/"control.cpp"; binary=temp/"control"
            defs="#define OKC_TINY 0\n#define OKC_WS 0\n"
            code=defs+PREAMBLE+fragment.MOE_HOST+CALL_PREFIX+fragment.MOE_CALL+"\n}\nint main(){okc_set_diag(1,1,0);exact_call();assert(!moe_resource_diag::enabled&&calls[0]==0);}\n"
            cpp.write_text(code)
            compiled=subprocess.run(["clang++","-std=c++17",str(cpp),"-o",str(binary)],capture_output=True,text=True)
            assert compiled.returncode==0,compiled.stderr
            result=subprocess.run([str(binary)],capture_output=True,text=True)
            assert result.returncode==0 and not result.stderr,result.stderr
    return {
        "verdict":"PASS", "kind":"actual_MoE_host_helper_and_call_compiled_with_mock_CUDA",
        "parent_sha256":hashlib.sha256(parent.encode()).hexdigest(),
        "fragment_sha256":hashlib.sha256(Path(fragment.__file__).read_bytes()).hexdigest(),
        "full_fragment_reverse_identity":True,"effective_flag_builds":6,"host_modes_per_build":12,
        "nonreporting_preprocessor_controls":2,
        "schema_bytes":131,"maximum_observed_resource_row_bytes":max(r["row_bytes"] for r in reports),
        "maximum_observed_module_bytes":max(r["bytes"] for r in reports),
        "maximum_accepted_first_row_bytes":320-131-48,
        "formal_bound":"A record writes only if used+schema+snprintf_length+48<=320; stop adds27B at most once. Six module budgets plus one33B Python notice <=1953B.",
        "limitations":["CPU mocks do not establish actual CUDA status, compiled resources, occupancy, compiler semantics or GPU correctness.","One immutable-resource key per module; any second distinct BIG/device/threads/dynamic key emits partial coverage before CUDA queries.","M/grid/PDL/generation are first-sample context, not resource dedup keys or weight identity.","Extreme formatting yields only explicit partial coverage, never an unbounded line."],
        "runs":reports,
    }


if __name__=="__main__":
    result=run()
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="runs"},indent=2))
