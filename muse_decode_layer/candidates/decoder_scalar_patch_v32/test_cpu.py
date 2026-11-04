"""Focused actual-source value/array equivalence; no CUDA or performance emulation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v32_build", HERE / "build.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def lit(s, name):
    n = next(n for n in ast.parse(s).body if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(n.value)


def identity():
    parent = (HERE / "parent.py").read_text()
    source = (HERE / "submission.py").read_text()
    built, edits, nobs = b.build()
    assert built == source
    restored = source
    for e in reversed(edits):
        assert restored.count(e["new"]) == e.get("count", 1)
        restored = restored.replace(e["new"], e["old"])
    assert restored == parent
    # Device edits are confined to this literal. All other CUDA, packing and
    # geometry functions restore through the already-reviewed diagnostic delta.
    diag = parent
    for e in json.loads((HERE / "observations.json").read_text()):
        diag = diag.replace(e["old"], e["new"])
    assert b.sha(diag) == b.OBSERVED_SHA
    ps, cs = lit(parent, "bfp_SK2_SRC"), lit(source, "bfp_SK2_SRC")
    assert "&b[nt]" not in cs
    assert "sk2_patch_v29(" not in cs
    assert "uint32_t*" not in b.VALUE_HELPERS and "(&" not in b.VALUE_HELPERS
    for line in ["const uint4* rp = reinterpret_cast<const uint4*>(ptab) +",
                 "(size_t)(repair[nt] & 0x3fffffffu) * 64 + lane * 2;",
                 "const uint4 r0 = __ldg(rp), r1 = __ldg(rp + 1);",
                 "if (UPW == 1 && uu == 0)", "if (UPW > 1)",
                 "mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);"]:
        assert ps.count(line) > 0 and ps.count(line) == cs.count(line), line
    # The original non-COMP loads and low-half MMA remain literal source.
    for line in ps.splitlines():
        if 'asm volatile("ld.shared.v4.u32' in line or 'if constexpr (HL)' in line:
            assert line in cs
    for node in ast.parse(parent).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
            if name and name != "bfp_SK2_SRC":
                assert lit(source, name) == node.value.value, name
    return {"parent_sha256": b.sha(parent), "candidate_sha256": b.sha(source),
            "full_restoration": True, "observation_edits": nobs,
            "total_edits": len(edits), "other_cuda_literals_identical": True,
            "raw_load_and_ring_handback_expressions_unchanged": True}


def native():
    parent = (HERE / "parent.py").read_text()
    source = (HERE / "submission.py").read_text()
    ps, cs = lit(parent, "bfp_SK2_SRC"), lit(source, "bfp_SK2_SRC")
    dec = b.block(parent, parent.index("__device__ __forceinline__ void decode8("))
    oldpatch = b.block(ps, ps.index("__device__ __forceinline__ void sk2_patch_v29("))
    helpers = cs[cs.index("// V32: COMP weight words"):cs.index("// ============================================================================ sk2")]
    decodecall_start = cs.index("            bv0[nt] = sk2_decode_patch_v32<0>")
    decodecall_end = cs.index("\n          }", decodecall_start)
    decodecall = cs[decodecall_start:decodecall_end]
    rawstart = cs.index("            const uint4* rp = reinterpret_cast<const uint4*>(ptab)")
    rawend = cs.index("\n          } else {", rawstart)
    rawcall = cs[rawstart:rawend].replace("__ldg(rp)", "checked(rp)").replace("__ldg(rp + 1)", "checked(rp + 1)")
    mma_start = cs.index("            if constexpr (COMP) {", cs.index("constexpr int HM = HL ? 2 : 1;"))
    mma_branch = b.block(cs, mma_start)
    # Actual branch compiled with record-only mma_bf16; no arithmetic model.
    prelude = r'''
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#define __device__
#define __forceinline__ inline
struct uint4 { uint32_t x,y,z,w; }; struct uint2 { uint32_t x,y; };
uint4 make_uint4(uint32_t x,uint32_t y,uint32_t z,uint32_t w){return{x,y,z,w};}
void ck(bool x){if(!x)std::exit(2);}
uint32_t lop3_sel(uint32_t a,uint32_t b,uint32_t c){return(a&~c)|(b&c);}
uint32_t prmt(uint32_t a,uint32_t b,unsigned s){uint64_t z=uint64_t(a)|(uint64_t(b)<<32);uint32_t r=0;for(int i=0;i<4;i++)r|=uint32_t((z>>(8*((s>>(4*i))&7)))&255)<<(8*i);return r;}
uint32_t rng=20261104;uint32_t next(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
std::vector<uint64_t> operands;
void mma_bf16(float*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t b0,uint32_t b1){operands.push_back(uint64_t(b0)|(uint64_t(b1)<<32));}
const uint4* rb;const uint4* re;uint4 checked(const uint4*p){ck(p>=rb&&p<re);return*p;}
'''
    body = prelude + dec + "\n" + oldpatch + "\n" + helpers + r'''
uint64_t words=0,rawloads=0,mmapairs=0;
void lane_case(uint32_t d,int lane,uint4 low,uint2 codes,uint32_t base4){
 uint32_t old[8];decode8(low.x,low.y,codes.x,base4,old);decode8(low.z,low.w,codes.y,base4,old+4);sk2_patch_v29(d,lane,old);
 const int nt=0;uint4 lw[1]={low};uint2 cw[1]={codes};uint32_t repair[1]={d};uint4 bv0[1],bv1[1];
''' + decodecall + r'''
 const uint32_t actual[8]={bv0[0].x,bv0[0].y,bv0[0].z,bv0[0].w,bv1[0].x,bv1[0].y,bv1[0].z,bv1[0].w};
 for(int k=0;k<8;k++){ck(actual[k]==old[k]);words++;}
}
int main(){
 // Each possible BF16 bit pattern is decoded in an eight-exponent window.
 for(unsigned base=0;base<=120;base+=8)for(unsigned code=0;code<16;code++)for(unsigned lo=0;lo<256;lo++){
  uint32_t l=lo*0x01010101u,n=code*0x11111111u;
  lane_case(0,0,{l,l,l,l},{n,n},base*0x01010101u);
 }
 // Every replacement bit pattern, all32 destination lanes, all512 positions across cases.
 for(unsigned bits=0;bits<65536;bits++){
  unsigned pos=(bits*37u)&511;uint32_t d=(1u<<30)|(pos<<16)|bits;
  uint4 l{next(),next(),next(),next()};uint2 n{next(),next()};
  for(int lane=0;lane<32;lane++)lane_case(d,lane,l,n,54u*0x01010101u);
 }
 const uint32_t special[]={0,0x8000,1,0x8001,0x7f80,0xff80,0x7fc1,0x7f81};
 for(unsigned pos=0;pos<512;pos++)for(uint32_t bits:special)for(int lane=0;lane<32;lane++)
  lane_case((1u<<30)|(pos<<16)|bits,lane,{next(),next(),next(),next()},{next(),next()},54u*0x01010101u);
 // Actual raw-sidecar address and value assignment, including nonzero unit indices.
 std::vector<uint4> raw(129*64);for(unsigned i=0;i<raw.size();i++)raw[i]={4*i,4*i+1,4*i+2,4*i+3};
 rb=raw.data();re=rb+raw.size();const void*ptab=raw.data();
 for(unsigned u=0;u<129;u++)for(int lane=0;lane<32;lane++){
  const int nt=0;uint32_t repair[1]={(2u<<30)|u};uint4 bv0[1],bv1[1];
''' + rawcall + r'''
  const uint32_t a[8]={bv0[0].x,bv0[0].y,bv0[0].z,bv0[0].w,bv1[0].x,bv1[0].y,bv1[0].z,bv1[0].w};
  for(unsigned w=0;w<8;w++){ck(a[w]==u*256+lane*8+w);rawloads++;}
 }
 // Actual COMP MMA argument branch: preserve j/mt/nt order, both halves and both pairs.
 constexpr bool COMP=true;constexpr int HM=1;
 uint4 bv0[4],bv1[4];float acc[4][2][4][4]{};uint32_t a[4][2][4]{};
 for(int nt=0;nt<4;nt++){bv0[nt]={unsigned(nt*8),unsigned(nt*8+1),unsigned(nt*8+2),unsigned(nt*8+3)};bv1[nt]={unsigned(nt*8+4),unsigned(nt*8+5),unsigned(nt*8+6),unsigned(nt*8+7)};}
 for(int NT=1;NT<=4;NT++)for(int MT=1;MT<=2;MT++)for(int NACC=1;NACC<=4;NACC++)for(int UPW=1;UPW<=4;UPW++){
  operands.clear();std::vector<uint64_t> expected;
  for(int uu=0;uu<UPW;uu++)for(int j=0;j<4;j++)for(int mt=0;mt<MT;mt++)for(int nt=0;nt<NT;nt++){
''' + mma_branch + r'''
   expected.push_back(uint64_t(nt*8+2*j)|(uint64_t(nt*8+2*j+1)<<32));
  }
  ck(operands==expected);mmapairs+=operands.size();
 }
 std::cout<<"{\"weight_word_comparisons\":"<<words<<",\"raw_word_comparisons\":"<<rawloads<<",\"actual_mma_operand_pairs\":"<<mmapairs<<"}";
}
'''
    mutations = {
        "wrong_owner": ("(d >> 20) & 31u", "(d >> 21) & 31u", 2),
        "wrong_value_half": ("const unsigned shift = (v & 1u) * 16u;", "const unsigned shift = ((v ^ 1u) & 1u) * 16u;", 2),
        "wrong_static_word": ("4 * HALF + 1>(q.y", "4 * HALF + 0>(q.y", 1),
        "wrong_raw_stride": ("* 64 + lane * 2", "* 63 + lane * 2", 1),
        "wrong_mma_pair": ("(j & 1) ? qv.w : qv.y", "(j & 1) ? qv.y : qv.w", 1),
    }
    with tempfile.TemporaryDirectory(prefix="v32-value-") as td:
        td = Path(td)
        def execute(src, label):
            exe = td / label
            c = subprocess.run(["clang++", "-std=c++17", "-O2", "-x", "c++", "-o", str(exe), "-"],
                               input=src, text=True, capture_output=True)
            assert c.returncode == 0, c.stderr
            return subprocess.run([str(exe)], text=True, capture_output=True)
        p = execute(body, "actual")
        assert p.returncode == 0, p.stderr
        result = json.loads(p.stdout)
        for name, (old, new, count) in mutations.items():
            assert body.count(old) == count, (name, body.count(old))
            # Mutate only the last occurrence: old reference stays unchanged.
            pos = body.rindex(old)
            changed = body[:pos] + body[pos:].replace(old, new, 1)
            p = execute(changed, name)
            assert p.returncode != 0, name
    return dict(result, native_source_sha256=b.sha(body), negative_controls=list(mutations),
                actual_source_helpers_and_consumer_blocks=True)


def run():
    start = b.sha((HERE / "submission.py").read_bytes())
    r = {"status": "CPU_PASS", "identity": identity(), "native": native(),
         "limits": ["No CUDA compiler or GPU used", "No proof of register allocation or zero local memory",
                    "CPU PRMT/LOP3 shims test integer byte semantics, not device execution",
                    "Packing, geometry and publication reuse frozen V29 source and prior proof",
                    "GPU correctness, activation, resources and fixed performance gates remain required"]}
    assert start == b.sha((HERE / "submission.py").read_bytes())
    return r


if __name__ == "__main__":
    r = run()
    text = json.dumps(r, indent=2) + "\n"
    (HERE / "cpu_result.json").write_text(text)
    print(text)
