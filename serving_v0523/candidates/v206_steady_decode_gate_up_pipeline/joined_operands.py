"""Feed identically generated immediate/late MMA partials through the actual BF16 SiLU writer."""
import hashlib
import json
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1])
d=stage/"python/sglang/kernels/jit/csrc/muse_spark"
g={"__file__":str(here/"cpu_check.py")}
script=(here/"cpu_check.py").read_text()
exec(compile(script[:script.index("results={}")],str(here/"cpu_check.py"),"exec"),g)
mm=g["src"].split("int main() {")[0]+r"""
int main(){
  auto cast=[](float f){uint32_t b=std::bit_cast<uint32_t>(f);return uint16_t((b+0x7fff+((b>>16)&1))>>16);};
  for(int seed=0;seed<256;++seed){
    int n=seed%3==0?1:(seed%3==1?7:32);
    auto ref0=compute<Base>(n,2*seed),ref1=compute<Base>(n,2*seed+1);
    auto new0=compute<MuseOverlapMma>(n,2*seed),new1=compute<MuseOverlapMma>(n,2*seed+1);
    for(int j=0;j<MUSE_OVERLAP_M;++j){
      uint16_t pair[2];
      float native=ref0[j]+ref1[j], late=new0[j]+new1[j];
      assert(std::bit_cast<uint32_t>(native)==std::bit_cast<uint32_t>(late));
      pair[0]=cast(4.0f*std::bit_cast<float>(uint32_t(cast(native))<<16));
      pair[1]=cast(4.0f*std::bit_cast<float>(uint32_t(cast(late))<<16));
      assert(fwrite(pair,sizeof(pair),1,stdout)==1);
    }
  }
}
"""
ref=runpy.run_path(str(here.parent/"v191_decode_gate_up_epilogue/independent_epilogue.py"))
writer=(d/"decode_gate_up.cuh").read_text().split("template <> class EpilogueGmemWriter",1)[1]
writer="template <> class EpilogueGmemWriter"+writer
writer,n=re.subn(r'asm\("mul\.rn\.f32[^;]+;[^\n]+\);',"product = __bfloat162float(value) * scale;",writer)
assert n==1
main=ref["MAIN"].replace("assert(bank_map(","assert(bank_map<false>(")
a=main.index("uint16_t value(");b=main.index("float rscale(",a)
main=main[:a]+r"""
std::vector<uint16_t> source_pairs;
uint16_t operand(unsigned r,unsigned col,unsigned half,unsigned band,bool late){
  size_t i=(band*13*128*MUSE_GATE_M+r*128+col*37+half*193)%(MUSE_GATE_M*256);
  return source_pairs[2*i+unsigned(late)];
}
"""+main[b:]
main=main.replace("int main(){",r"""
int main(){
  source_pairs.resize(2*MUSE_GATE_M*256);
  assert(fread(source_pairs.data(),sizeof(uint16_t),source_pairs.size(),stdin)==source_pairs.size());
""")
main=main.replace("=value(r,c%128,c/128,nt);","=operand(r,c%128,c/128,nt,true);")
main=main.replace("value(r,c%128,0,c/128)","operand(r,c%128,0,c/128,false)").replace("value(r,c%128,1,c/128)","operand(r,c%128,1,c/128,false)")
assert 'value(r,' not in main
cpp=ref["CPP"]+"template<bool Prompt>\n"+ref["function"]((d/"decode_gate_up_launch.cuh").read_text(),"static bool bank_map(")+"\n"+writer+main
rows=[]
with tempfile.TemporaryDirectory() as temp:
    t=Path(temp);(t/"mm.cpp").write_text(mm);(t/"ep.cpp").write_text(cpp)
    for w in (8,16,24):
        opt=["c++","-std=c++20","-O2","-fno-strict-aliasing",f"-DMUSE_OVERLAP_M={w}",f"-DMUSE_GATE_M={w}"]
        header=d/"grouped_decode_mma_pair.cuh"
        subprocess.run([*opt,f'-DHEADER="{header}"',"-DMUSE_PAIR_LAST_K_SLICES",t/"mm.cpp","-o",t/"mm"],check=True)
        subprocess.run([*opt,"-DSTART_ROW=3",f"-DSLOT_COUNT={w+5}","-DHUMMING_SHAPE_N=8192","-DPAD=5",t/"ep.cpp","-o",t/"ep"],check=True)
        inputs=subprocess.check_output([t/"mm"])
        assert len(inputs)==4*256*w
        values=list(map(int,subprocess.check_output([t/"ep"],input=inputs).split()))
        rows.append(dict(width=w,original_and_candidate_fp32_values=256*w,
                         input_digest=hashlib.sha256(inputs).hexdigest(),final_bf16_silu_words=values[0],
                         untouched_redzone_values=values[1],distinct_gate_words=values[3],distinct_up_words=values[4]))
print(json.dumps(dict(physical_gpu=False,driver="CPU delayed matrix execution of the production two-pair run, two K-warp FP32 partials added, original BF16 cast/exponent recovery, native packed-row writer and BF16 SiLU",cases=rows),indent=2))
