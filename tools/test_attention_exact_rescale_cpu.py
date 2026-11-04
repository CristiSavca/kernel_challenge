"""Source restoration and actual-loop host FP32 checks; no CUDA/Torch import."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c"
CANDIDATE_SHA = "c6b90f168bfdc21a8583b6a4bd03cebc0e6c4e03f994f74759c54ab5b677335c"
PARENT = ROOT / f"experiments/candidates/muse_attention_decode/{PARENT_SHA}.py"
CANDIDATE = ROOT / "muse_attention_decode/candidates/exact_rescale_v23.py"
OLD_HEADER = "# CUDA source and normal compile/launch configuration remain unchanged.\n"
NEW_HEADER = (
    "# The parent preserved the upstream CUDA and normal compile/launch configuration.\n"
    f"# V23 parent SHA-256: {PARENT_SHA}\n"
    "# V23 changes only Q1 non-LAZY output rescaling: skip its sixteen FP32\n"
    "# accumulator products when the existing alpha is exactly positive one.\n"
    "# No warp vote; row-sum math, Q8, ONE, precision and scheduling are unchanged.\n"
    "# Mechanism: PipeLink SM90 role_core.cuh @ 4d8ebab9f1527589e1cfd892a8a77a3b6211991a\n"
    "# (Apache-2.0); implementation here is a local scalar conditional.\n"
)
OLD_BODY = """        l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
"""
NEW_BODY = """        l_r[0] = l_r[0] * alpha + ls;
        if (alpha != 1.f) {
#pragma unroll
          for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
        }
"""
REPLACEMENTS = [(OLD_HEADER, NEW_HEADER), (OLD_BODY, NEW_BODY),
    ('name=f"okc_attn_v298{tag}_{arch}"', 'name=f"okc_attn_exact_rescale_v23{tag}_{arch}"')]


def conformance(parent, candidate):
    ast.parse(candidate)
    restored = candidate
    for old, new in REPLACEMENTS:
        assert parent.count(old) == 1 and restored.count(new) == 1
        restored = restored.replace(new, old, 1)
    assert restored == parent, "Unapproved source change"
    assert candidate.splitlines()[0] == parent.splitlines()[0]
    # Actual source bytes supplied to the host compiler; parent and candidate
    # braces/macro placement outside this excerpt are covered by full restoration.
    return parent[parent.index(OLD_BODY):parent.index(OLD_BODY) + len(OLD_BODY)], \
        candidate[candidate.index(NEW_BODY):candidate.index(NEW_BODY) + len(NEW_BODY)]


NATIVE = r'''
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
static float fp(uint32_t b) { float f; std::memcpy(&f,&b,4); return f; }
static uint32_t bits(float f) { uint32_t b; std::memcpy(&b,&f,4); return b; }
__attribute__((noinline)) void old_loop(float alpha, float ls, float l_r[1], float o[1][8][4]) {
@@OLD@@
}
__attribute__((noinline)) void new_loop(float alpha, float ls, float l_r[1], float o[1][8][4]) {
@@NEW@@
}
int main() {
  std::vector<uint32_t> values={0,0x80000000u,1,0x80000001u,0x007fffffu,0x807fffffu,
    0x00800000u,0x80800000u,0x3f800000u,0xbf800000u,0x7f7fffffu,0xff7fffffu};
  uint32_t rng=0x9e3779b9u;
  for(int i=0;i<4096;++i) {
    rng=rng*1664525u+1013904223u;
    if((rng&0x7f800000u)!=0x7f800000u) values.push_back(rng);
  }
  const uint32_t alphas[]={0,0x80000000u,0x3e800000u,0x3f7fffffu,0x3f800000u,
    0x3f800001u,0x40000000u,0x7f800000u,0x7fc01234u};
  uint64_t checks=0, unit_checks=0;
  for(uint32_t ab:alphas) for(uint32_t vb:values) {
    float a[1][8][4],b[1][8][4];
    for(int t=0;t<8;++t) for(int e=0;e<4;++e) a[0][t][e]=b[0][t][e]=fp(vb);
    float la[1]={fp(0x3f012345u)},lb[1]={la[0]};
    old_loop(fp(ab),fp(0x3d765432u),la,a); new_loop(fp(ab),fp(0x3d765432u),lb,b);
    if(bits(la[0])!=bits(lb[0])) return 2;
    for(int t=0;t<8;++t) for(int e=0;e<4;++e) {
      ++checks;
      if(bits(a[0][t][e])!=bits(b[0][t][e])) return 3;
      if(ab==0x3f800000u) {++unit_checks; if(bits(b[0][t][e])!=vb) return 4;}
      if(e>=2 && bits(b[0][t][e])!=vb) return 5;
    }
  }
  // Nonfinite O at alpha==1 is reported separately, never promoted to a
  // universal IEEE/GPU payload-preservation guarantee.
  const uint32_t special[]={0x7f800000u,0xff800000u,0x7fc12345u,0xffc12345u,0x7f812345u};
  std::vector<uint32_t> special_differences;
  for(uint32_t vb:special) {
    float a[1][8][4]={},b[1][8][4]={},la[1]={0},lb[1]={0};
    a[0][0][0]=b[0][0][0]=fp(vb);
    old_loop(1.f,0.f,la,a);new_loop(1.f,0.f,lb,b);
    if(bits(a[0][0][0])!=bits(b[0][0][0])) special_differences.push_back(vb);
  }
  std::cout << "{\"finite_inputs\":"<<values.size()<<",\"alphas\":9,\"output_comparisons\":"<<checks
    <<",\"unit_alpha_copy_checks\":"<<unit_checks<<",\"nonfinite_O_directed\":5,\"nonfinite_difference_input_bits\":[";
  for(size_t i=0;i<special_differences.size();++i) {if(i) std::cout<<",";std::cout<<special_differences[i];}
  std::cout << "]}\n";
}
'''


def run():
    parent, candidate = PARENT.read_text(), CANDIDATE.read_text()
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    assert hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() == CANDIDATE_SHA
    old, new = conformance(parent, candidate)
    mutants = [candidate.replace("if (alpha != 1.f) {", "if (alpha > 0.99f) {", 1),
               candidate.replace("l_r[0] = l_r[0] * alpha + ls;", "l_r[0] += ls;", 1),
               candidate.replace("if (alpha != 1.f) {", "if (__any_sync(0xffffffffu, alpha != 1.f)) {", 1)]
    for mutant in mutants:
        try: conformance(parent, mutant)
        except AssertionError: pass
        else: raise AssertionError("Source-scope negative control was accepted")
    compiler = shutil.which("clang++")
    assert compiler, "A host C++ compiler is required for the actual-loop check"
    with tempfile.TemporaryDirectory(prefix="okc-rescale-cpu-") as tmp:
        source, binary = Path(tmp)/"test.cpp", Path(tmp)/"test"
        source.write_text(NATIVE.replace("@@OLD@@", old).replace("@@NEW@@", new))
        subprocess.run([compiler,"-std=c++17","-O2","-fno-fast-math","-ffp-contract=off",
                        str(source),"-o",str(binary)],check=True,capture_output=True,timeout=30)
        native = subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=10)
    return {"status":"PASS", "parent_sha256":PARENT_SHA,"candidate_sha256":CANDIDATE_SHA,
            "full_byte_restoration":True,"source_scope_negative_controls":3,
            "actual_loop_host_result":json.loads(native.stdout),
            "limits":["Host FP32 semantics only; no CUDA compilation, FTZ, scheduling or GPU numerical proof.",
                      "Finite O inputs include signed zeros, subnormals and extremes; nonfinite O differences reported separately.",
                      "Host row-sum fusion is disabled equally in both wrappers; GPU FMA lowering is unproven.",
                      "No performance measurement, provider call or kernel allocation."]}


if __name__ == "__main__":
    result = run()
    out = ROOT/"experiments/iterations/20261104-attention-exact-rescale-cpu.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
