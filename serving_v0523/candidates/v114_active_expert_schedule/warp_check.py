"""Execute the production active-expert scheduler with a 32-lane C++ coroutine ballot."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREFIX = r"""
#include <array>
#include <cassert>
#include <coroutine>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using std::uint32_t;
struct Index { uint32_t x, y; } threadIdx, blockIdx, gridDim;
struct Params { uint32_t experts; const std::vector<int>* data;
  struct Offsets { const std::vector<int>* data; int operator[](size_t i)const{return data->at(i);} } offsets;
};
std::array<std::vector<std::pair<unsigned,unsigned>>,32> trace;
constexpr uint32_t kWidth=64, kWarps=4;
void project_tile(const Params&, uint32_t expert, uint32_t column) {
  trace[threadIdx.x%32].emplace_back(expert,column);
}
struct Routine {
  struct promise_type {
    bool predicate; uint32_t result;
    Routine get_return_object(){return {std::coroutine_handle<promise_type>::from_promise(*this)};}
    std::suspend_always initial_suspend(){return {};}
    std::suspend_always final_suspend()noexcept{return {};}
    void return_void(){}
    void unhandled_exception(){std::abort();}
  };
  std::coroutine_handle<promise_type> h;
};
struct Ballot {
  bool predicate; Routine::promise_type* p{};
  bool await_ready(){return false;}
  void await_suspend(std::coroutine_handle<Routine::promise_type> h){p=&h.promise();p->predicate=predicate;}
  uint32_t await_resume(){return p->result;}
};
#define __ballot_sync(mask,pred) (co_await Ballot{bool(pred)})
#define __popc(v) __builtin_popcount(v)
#define __ffs(v) __builtin_ffs(v)
"""
SUFFIX = r"""
static int cases{},cta{},tiles{},repeat{},empty{},fallback{};
std::vector<std::pair<unsigned,unsigned>> execute(const Params& p) {
  std::array<Routine,32> warp{};
  for(unsigned i=0;i<32;i++){trace[i].clear();warp[i]=run_active_kernel(p);}
  for(;;){
    unsigned done=0,mask=0;
    for(unsigned i=0;i<32;i++){
      threadIdx.x=i;warp[i].h.resume();
      done+=warp[i].h.done();
      if(!warp[i].h.done() && warp[i].h.promise().predicate)mask|=1u<<i;
    }
    if(done){assert(done==32);break;}
    for(auto& r:warp)r.h.promise().result=mask;
  }
  for(unsigned i=0;i<32;i++){
    assert(trace[0]==trace[i]);warp[i].h.destroy();
  }
  return trace[0];
}
void check(const std::vector<unsigned>& counts,unsigned expected,unsigned bands){
  const uint32_t experts=counts.size();std::vector<int> offsets{19};
  for(auto n:counts)offsets.push_back(offsets.back()+n);
  Params p{experts,&offsets,{&offsets}};
  unsigned y=expected*2;
  if(!(experts<=32 && expected>0 && expected<=(experts-1)/2)){++fallback;return;}
  std::vector<std::vector<unsigned>> seen(experts,std::vector<unsigned>(bands));
  for (unsigned band=0;band<bands;band++)for(unsigned group=0;group<y;group++){
    blockIdx={band,group};gridDim={bands,y};
    auto got=execute(p);++cta;empty+=got.empty();repeat+=got.size()>1;
    for(auto [expert,col]:got){assert(col==64*band);++seen.at(expert).at(col/64);++tiles;}
  }
  for(unsigned expert=0;expert<experts;expert++)for(auto visits:seen[expert])assert(visits==bool(counts[expert]));
  ++cases;
}
int main(){
  for(unsigned n:{3u,4u,5u,6u,7u,8u,9u})for(unsigned mask=0;mask<(1u<<n);mask++){
    std::vector<unsigned> v(n);
    for(unsigned i=0;i<n;i++)v[i]=((mask>>i)&1)*(1+i*3);
    check(v,1,2);
  }
  std::mt19937 rng(79641);
  for(unsigned step=0;step<720;step++){
    unsigned experts=std::array<unsigned,7>{2,7,16,22,31,32,33}[step%7];
    std::vector<unsigned> v(experts);
    for(unsigned i=0;i<experts;i++)v[i]=(rng()%100<(step%5)*25)?1+rng()%24:0;
    if(step%6==0)v.back()=17;
    check(v,std::array<unsigned,6>{0,1,2,4,8,16}[step%6],1+step%3);
  }
  std::printf("{\"tested_active_plans\":%d,\"fallback_plans\":%d,\"simulated_ctas\":%d,\"live_tiles\":%d,\"early_exits\":%d,\"multi_expert_ctas\":%d}\n",cases,fallback,cta,tiles,empty,repeat);
}
"""


def function_body(code, signature):
    begin=code.index("{",code.index(signature))
    depth=1
    for i in range(begin+1,len(code)):
        depth += (code[i]=="{") - (code[i]=="}")
        if depth==0:
            return code[begin+1:i]
    raise ValueError("incomplete function")


def check(root):
    name="python/sglang/kernels/jit/csrc/muse_spark/expert_skinny_gate_up.cuh"
    code=(root/name).read_text()
    body=function_body(code,"void run_active_kernel")
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)
        src=tmp/"test.cpp"
        src.write_text(PREFIX+"\nRoutine run_active_kernel(const Params p){\n"+body+"\n}\n"+SUFFIX)
        subprocess.run(["c++","-O2","-std=c++20",src,"-o",tmp/"warp"],check=True)
        result=json.loads(subprocess.check_output([tmp/"warp"]))
    parent_patch=next(Path(__file__).resolve().parents[1].glob("v89_*/patch.diff")).read_text()
    head=parent_patch.index("diff --git a/"+name+" ")
    section=parent_patch[head:].split("\ndiff --git ",1)[0]
    parent="\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))+"\n"
    old=function_body(parent,"void run_kernel")
    old=old.replace("  const uint32_t expert = blockIdx.y;\n","").replace("  const uint32_t column = blockIdx.x * kWidth;\n","")
    assert old==function_body(code,"void project_tile")
    for fn in ("broadcast_scale","operand_pair","accumulate"):
        assert function_body(parent,fn+"(")==function_body(code,fn+"(")
    assert "expected_local_routes <= (params.experts - 1) / 2" in code
    result["projection_arithmetic_body_matches_v89"]=True
    return result


if __name__=="__main__":
    print(json.dumps(check(Path(sys.argv[1])),indent=2))
