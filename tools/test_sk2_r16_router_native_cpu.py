#!/usr/bin/env python3
"""Native CPU checks of extracted R16 router publication/reduction snippets.

No submission import, CUDA execution, GPU memory-order proof, or MMA emulation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2"
PARENT = ROOT / "experiments/candidates/muse_decode_layer" / (PARENT_SHA + ".py")
CANDIDATE = ROOT / "muse_decode_layer/candidates/native_sk2_r16_v14.py"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def embedded(source, name):
    return ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))


def block_at(source, start):
    opening = source.index("{", start)
    depth = 1
    for end in range(opening+1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start:end+1]
    raise AssertionError("Unbalanced extracted source block")


def statement(source, pattern):
    matches = re.findall(pattern, source)
    assert len(matches) == 1, (pattern, len(matches))
    return matches[0]


def normalized(source):
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    return re.sub(r"\s+", "", source)


def integer_expression(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return integer_expression(node.left) * integer_expression(node.right)
    raise AssertionError("Unexpected allocation expression")


def snippets(source):
    cuda = embedded(source, "bfp_SK2_SRC")
    kernel = cuda[cuda.index("sk2_kernel("):cuda.index("static void launch_sk2")]
    router = block_at(kernel, kernel.index("  if constexpr (FUSE) {"))
    scatter_start = router.index("    float* redr = reinterpret_cast<float*>(gbase);")
    scatter_start = router.index("#pragma unroll", scatter_start)
    scatter_end = router.index("    named_bar(1, NC);", scatter_start)
    scatter = router[scatter_start:scatter_end]
    writer_start = router.index("    float* pp = part")
    writer_end = router.index("    named_bar(1, NC);", writer_start)
    writer = router[writer_start:writer_end]
    reader = block_at(router, router.index("    if (s_rlast) {"))
    # Parenthesization/order is prescribed even though synthetic exact-valued
    # operands would not distinguish a reordered floating-point sum.
    reductions = re.findall(r"const float (vu|vv|v) = ([^;]+);", reader)
    assert [name for name,_ in reductions] == ["vu","vv","v"], reductions
    for name,rhs in reductions:
        index={"vu":"iu","vv":"iv","v":"i"}[name]
        expected=" + ".join("__ldcg(pr + "+(str(piece*256)+" + " if piece else "")+index+")" for piece in range(8))
        assert normalized(rhs)==normalized(expected),(name,"reduction is not fixed left-associated piece0..7")
    predicate = statement(router, r"s_rlast\s*=\s*([^;]+);")
    geometry_start = kernel.index("  constexpr int NW =")
    geometry_end = kernel.index("  // partial sums", geometry_start)
    geometry = kernel[geometry_start:geometry_end]
    nstr = statement(kernel, r"(constexpr int NSTR\s*=\s*[^;]+;)")
    route = statement(kernel, r"(const int rt\s*=\s*[^;]+;)")
    gsr = statement(kernel, r"(const int gsr\s*=\s*[^;]+;)")
    wsr = statement(kernel, r"(const uint8_t\* wsr\s*=\s*[^;]+;)")
    a0 = statement(router, r"(const uint32_t a0\s*=\s*[^;]+;)")
    load_offsets = re.findall(r'"r"\(a0(?: \+ (\d+))?\)\);', router)
    assert len(load_offsets) == 4, load_offsets
    producer = statement(kernel, r"bulk_g2s\(s0 \+ XBY, (wsr[^,]+), WBYR, full0 \+ 8 \* slot\);")
    # The unchanged native MMA stream has two explicit calls, hi then lo, per k16.
    arithmetic = router[router.index("    float accr["):router.index("    // k-warp reduction")]
    parent_cuda = embedded(PARENT.read_text(), "bfp_SK2_SRC")
    parent_router = parent_cuda[parent_cuda.index("  if constexpr (FUSE) {"):]
    parent_arithmetic = parent_router[parent_router.index("    float accr["):parent_router.index("    // k-warp reduction")]
    assert normalized(arithmetic) == normalized(parent_arithmetic), "Router MMA/operand source changed"
    assert "mma_bf16(accr[0][mt]" in arithmetic and "mma_bf16(accr[1][mt]" in arithmetic
    allocations = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) and
            isinstance(t.slice, ast.Constant) and t.slice.value == "fused_part" for t in node.targets):
            assert isinstance(node.value, ast.Call) and node.value.args
            allocations.append(integer_expression(node.value.args[0]))
    assert allocations == [32*8*32*8]*2, allocations
    return dict(geometry=geometry, nstr=nstr, route=route, gsr=gsr, wsr=wsr, a0=a0,
                producer=producer, load_offsets=[int(x or 0) for x in load_offsets],
                scatter=scatter, writer=writer, reader=reader, predicate=predicate,
                allocations=allocations, inherited_mma_source_sha256=sha(normalized(arithmetic)))


NATIVE = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
static void need(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
static float* pb=nullptr; static size_t pcount=0; static std::vector<uint32_t> epochs;
static uint32_t epoch=0; static uint64_t stores=0,loads=0,calls=0,all_orders=0,coordinate_cases=0;
static void __stcg(float*p,float x){
  const ptrdiff_t i=p-pb; need(i>=0 && (size_t)i<pcount,"part store bounds");
  need(epochs[i]!=epoch,"part store overlap"); epochs[i]=epoch;*p=x;++stores;
}
static float __ldcg(const float*p){
  const ptrdiff_t i=p-pb;need(i>=0&&(size_t)i<pcount,"part read bounds");
  need(epochs[i]==epoch,"stale/unpublished partial");++loads;return *p;
}
struct TArgs{int mode;};
template<int TOKR,int MT,int NSTAGE> struct Geometry {
  static constexpr int NT=2,NWR=1,WSK=8,UPW=1;
  static constexpr bool FUSE=true,COMP=true,HL=false;
@@GEOMETRY@@
};
template<int MT> void scatter(float*redr,int wk,int lane,int piece,int tile){
  constexpr int TOKS=16*MT;const int g=lane>>2,t4=lane&3;
  float accr[2][MT][4];
  for(int h=0;h<2;++h)for(int m=0;m<MT;++m)for(int e=0;e<4;++e){
    const int token=16*m+lane/4+(e>=2?8:0),row=2*(lane%4)+(e&1);
    accr[h][m][e]=(float)((piece+1)*64+wk*2+token+row+tile)+h*0.25f;
  }
@@SCATTER@@
}
template<int MT> void publish(float*part,float*redr,int rt,int rq){
  constexpr int NC=256,TOKS=16*MT,WSK=8;
  for(int tid=0;tid<NC;++tid){
@@WRITER@@
  }
}
template<int MT> void finish(float*part,float*outr,const float*uv_ra,const float*uv_rs,int M,int rt,int mode){
  constexpr int NC=256;const TArgs ta{mode};const bool s_rlast=true;
  for(int tid=0;tid<NC;++tid){
@@READER@@
  }
}
static bool last(uint32_t old_bits){
  int old;static_assert(sizeof(old)==sizeof(old_bits));std::memcpy(&old,&old_bits,sizeof old);
  return @@PREDICATE@@;
}
struct State{
  std::vector<float> part=std::vector<float>(32*8*32*8,NAN);
  std::array<uint32_t,32> counter{};
  State(){pb=part.data();pcount=part.size();epochs.assign(pcount,0);}
};
template<int MT> struct Fixture{
  std::array<std::vector<float>,8> redr;
  int rt;
  explicit Fixture(int tile):rt(tile){
    for(int p=0;p<8;++p){redr[p].assign(8*16*MT*8,NAN);
      for(int w=0;w<8;++w)for(int l=0;l<32;++l)scatter<MT>(redr[p].data(),w,l,p,rt);
      for(int w=0;w<8;++w)for(int t=0;t<16*MT;++t)for(int r=0;r<8;++r){
        const float a=(float)((p+1)*64+w*2+t+r+rt);
        need(redr[p][(w*16*MT+t)*8+r]==a+(a+0.25f),"scatter coordinate or hi/lo mismatch");
      }
    }
  }
};
static float reference(int tile,int token,int row){
  double value=0;
  for(int p=0;p<8;++p)for(int w=0;w<8;++w)for(int h=0;h<2;++h)
    value+=(p+1)*64+w*2+token+row+tile+h*0.25;
  return(float)value;
}
template<int MT> void trial(State&s,Fixture<MT>&f,int M,int mode,const std::array<int,8>&order){
  ++epoch;need(epoch!=0,"test epoch wrap");++calls;
  std::vector<float> out(32*256,NAN);float ra[32],rs[32];
  for(int t=0;t<32;++t){ra[t]=0.5f+t/32.f;rs[t]=1.f+t/16.f;}
  int finishers=0;
  for(int arrival=0;arrival<8;++arrival){
    publish<MT>(s.part.data(),f.redr[order[arrival]].data(),f.rt,order[arrival]);
    const uint32_t old=s.counter[f.rt]++; // intentional modulo2^32 model
    if(last(old)){
      ++finishers;need(arrival==7,"premature/four-way last arrival");
      finish<MT>(s.part.data(),out.data(),ra,rs,M,f.rt,mode);
    }
  }
  need(finishers==1 && !(s.counter[f.rt]&7),"counter alignment/finisher count");
  const int tokens=mode==2?M/2:M;
  for(int t=0;t<32;++t)for(int r=0;r<256;++r){
    const bool written=t<tokens && r/8==f.rt;const float v=out[t*256+r];
    if(!written){need(std::isnan(v),"unexpected router output");continue;}
    const int j=r%8;
    const float expect=mode==2?rs[t]*(reference(f.rt,2*t,j)+ra[t]*reference(f.rt,2*t+1,j)):reference(f.rt,t,j);
    need(v==expect,"router sum/UV pairing/stride mismatch");
  }
}
template<int TOKR,int MT,int NSTAGE> void coordinates(){
  using G=Geometry<TOKR,MT,NSTAGE>;
  constexpr bool FUSE=true;constexpr int WBYR=G::WBYR,XBY=G::XBY,UPS=G::UPS;
  @@NSTR@@
  need(NSTR==2 && G::NW==8 && UPS==8,"router geometry");
  need(G::SST>=XBY+G::WBY && G::SST>=XBY+WBYR,"undersized router ring slot");
  need(G::SST==((XBY+std::max(G::WBY,WBYR)+1023)&~1023),"ring alignment");
  std::vector<uint8_t> stream(16*32*8*2048),seen(stream.size()/2,0);
  const uint8_t*Wsr=stream.data();
  constexpr int offsets[4]={@@LOAD_OFFSETS@@};
  for(int cta=0;cta<256;++cta){
    @@ROUTE@@
    @@WSR@@
    need(rt==cta/8 && rq==cta%8,"router CTA decomposition");
    const int nst=16;
    for(int j=0;j<NSTR;++j){const int st=nst+j;
      @@GSR@@
      need(gsr==2*(cta%8)+j,"router K-stage coverage");
      const uint8_t* global_piece=@@PRODUCER@@;
      for(int warp=0;warp<8;++warp)for(int lane=0;lane<32;++lane){
        const uint32_t s0=0;
        @@A0@@
        for(int group=0;group<4;++group)for(int word=0;word<4;++word)for(int half=0;half<2;++half){
          const size_t byte=(global_piece-Wsr)+(a0-XBY)+offsets[group]+word*4+half*2;
          need(!(byte&1) && byte+2<=stream.size(),"hi/lo load extent");
          need(seen[byte/2]++==0,"overlapping hi/lo weight read");
        }
      }
    }
  }
  need(std::all_of(seen.begin(),seen.end(),[](uint8_t x){return x==1;}),"missing hi/lo K weight");
  ++coordinate_cases;
}
int main(int argc,char**argv){
  try{
    coordinates<8,1,4>();coordinates<16,1,3>();coordinates<32,2,2>();
    State s;Fixture<1> f0(0),f31(31);Fixture<2> f032(0),f3132(31);
    std::array<int,8> order{0,1,2,3,4,5,6,7};
    const bool quick=argc>1 && std::string(argv[1])=="quick";
    do{trial(s,f0,1,0,order);++all_orders;if(quick)break;}while(std::next_permutation(order.begin(),order.end()));
    std::iota(order.begin(),order.end(),0);
    // Counter starts aligned to eight, including signed-boundary and full wrap.
    for(uint32_t start:{0u,8u,0x7ffffff8u,0xfffffff8u}){
      for(int tile:{0,31})for(int orientation=0;orientation<2;++orientation){
        s.counter[tile]=start;
        if(orientation)std::reverse(order.begin(),order.end());
        for(int M=1;M<=32;++M){
          if(M<=8)trial(s,tile?f31:f0,M,0,order); // TOKR8, MT1
          if(M<=16)trial(s,tile?f31:f0,M,0,order); // TOKR16, MT1
          trial(s,tile?f3132:f032,M,0,order);      // TOKR32, MT2
          if(!(M&1)){
            if(M<=8)trial(s,tile?f31:f0,M,2,order);
            if(M<=16)trial(s,tile?f31:f0,M,2,order);
            trial(s,tile?f3132:f032,M,2,order);
          }
        }
      }
    }
    std::printf("{\"arrival_permutations\":%llu,\"completed_calls\":%llu,\"part_stores\":%llu,\"part_loads\":%llu,\"geometry_cases\":%llu,\"unique_BF16_hi_lo_coordinates_per_geometry\":4194304}\n",(unsigned long long)all_orders,(unsigned long long)calls,(unsigned long long)stores,(unsigned long long)loads,(unsigned long long)coordinate_cases);
  }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}
'''


def native_source(parts):
    text=NATIVE
    for key,value in parts.items():
        if key=="load_offsets":value=",".join(map(str,value))
        # Lift unchanged local constant expressions into a native template type.
        if key=="geometry":value=value.replace("constexpr int", "static constexpr int")
        if isinstance(value,str):text=text.replace("@@"+key.upper()+"@@",value)
    assert "@@" not in text
    return text


def run(candidate=CANDIDATE, expected_sha=None):
    candidate=Path(candidate)
    source=candidate.read_text()
    digest=sha(source)
    assert sha(PARENT.read_text())==PARENT_SHA
    if expected_sha is not None:assert digest==expected_sha,"Candidate source hash mismatch"
    parts=snippets(source)
    code=native_source(parts)
    negatives={
        "old_four_way_counter":native_source({**parts,"predicate":"(old & 3) == 3"}),
        "old_partial_writer_stride":native_source({**parts,"writer":parts["writer"].replace("rt * 8 + rq","rt * 4 + rq")}),
        "undersized_old_ring":native_source({**parts,"geometry":re.sub(r"constexpr int SST =[^;]+;","constexpr int SST = ((XBY + WBY) + 1023) & ~1023;",parts["geometry"])}),
    }
    for name,mutant in negatives.items():assert mutant!=code,(name,"negative mutation did not change extracted source")
    results={}
    with tempfile.TemporaryDirectory(prefix="okc-r16-router-native-") as temp:
        temp=Path(temp)
        for name,cpp in [("actual",code),*negatives.items()]:
            src=temp/(name+".cpp");binary=temp/name;src.write_text(cpp)
            built=subprocess.run(["clang++","-std=c++17","-O2","-ffp-contract=off",str(src),"-o",str(binary)],capture_output=True,text=True,timeout=30)
            assert built.returncode==0,(name,built.stderr)
            executed=subprocess.run([str(binary),*(["quick"] if name!="actual" else [])],capture_output=True,text=True,timeout=60)
            if name=="actual":
                assert executed.returncode==0,executed.stderr
                results[name]=json.loads(executed.stdout)
                assert results[name]["arrival_permutations"]==40320
            else:
                assert executed.returncode!=0,(name,"negative control incorrectly passed")
                results[name]={"caught":True,"message":executed.stderr.strip()}
    assert sha(candidate.read_text())==digest,"Candidate changed during proof"
    return {"verdict":"PASS","candidate_sha256":digest,"parent_sha256":PARENT_SHA,
        "test_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_translation_sha256":sha(code),"snippet_sha256":{k:sha(v) for k,v in parts.items() if isinstance(v,str) and k!="inherited_mma_source_sha256"},
        "inherited_mma_source_sha256":parts["inherited_mma_source_sha256"],
        "explicit_reduction_order":"Source-parsed left-associated sums read pieces0..7 in order, separately forordinary/u/v.",
        "partial_allocation_float_counts":parts["allocations"],"results":results,
        "limits":["CPU publication is sequential and assumes all writes become visible before the modeled release counter; no CUDA release/acquire proof.",
                  "32-bit unsigned wrap models counter bits; this does not execute GPU atomics or test overlapping kernel invocations.",
                  "Actual scatter/writer/reader and addressing snippets execute natively; synthetic accumulator operands are used, not tensor-core MMA emulation.",
                  "All40320 arrival permutations cover canonical M1/TOKR8/tile0. Directed mixed-TOKR/ragged/UV/tile31/wrap tests are separate coverage, not the full Cartesian product.",
                  "Unchanged router MMA source and complete hi/lo packed-address coverage do not establish GPU numerical correctness or performance."]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",type=Path,default=CANDIDATE)
    parser.add_argument("--expected-sha")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=run(args.candidate,args.expected_sha)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
