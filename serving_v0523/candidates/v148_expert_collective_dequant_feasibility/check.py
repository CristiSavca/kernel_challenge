"""Replay the compiled dequant expressions for both signs and all supported raw scales."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
H = "python/sglang/kernels/jit/csrc/muse_spark/expert_route_pair.cuh"
M = H.replace("expert_route_pair", "expert_medium_route_pair")
TAGS = ("base", "test", "swar", "table", "marlin", "imad")


def source_from_patch(patch, path):
    marker = f"diff --git a/{path} b/{path}\n"
    body = patch.split(marker, 1)[1].split("\ndiff --git ", 1)[0]
    assert "new file mode" in body
    return "\n".join(line[1:] for line in body.splitlines() if line.startswith("+") and not line.startswith("+++")) + "\n"


CPP = r"""
#include <cstdint>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#define SGL_DEVICE inline
uint32_t fpbits(float f) {uint32_t v;std::memcpy(&v,&f,4);return v;}
float bf(uint32_t x){uint32_t raw=x<<16;float f;std::memcpy(&f,&raw,4);return f;}
uint32_t pair_mul(uint32_t x,uint32_t y) {
    uint32_t lo=fpbits(bf(x)*bf(y)),hi=fpbits(bf(x>>16)*bf(y>>16));
    return (hi & 0xffff0000u) | (lo >> 16);
}
uint32_t cpu_prmt(uint32_t lo,uint32_t hi,uint32_t sel) {
    uint32_t v=0;uint64_t bytes=lo|(uint64_t(hi)<<32);
    for(unsigned p=0;p<4;++p){unsigned pick=(sel>>(4*p))&15, ch=(bytes>>(8*(pick&7)))&255;
        if(pick&8)ch=(ch&128)?255:0; v|=ch<<(8*p);}
    return v;
}
"""
DRIVER = r"""
uint32_t expected(uint32_t c,uint32_t raw) {
    const float mag[]={0,.5f,1,1.5f,2,3,4,6};
    float scale=raw<8?0.0f:std::ldexp(1.0f+float(raw&7)/8,int(raw>>3)-7);
    float x=(c&8)?-mag[c&7]:mag[c&7];
    uint32_t f=fpbits(x*scale);assert((f&65535)==0);return f>>16;
}
int main(){
    uint32_t rng=148; uint64_t pairs=0,negzero=0;
    for(unsigned low=0;low<=126;++low) for(unsigned high=0;high<=126;++high)
    for(unsigned pair=0;pair<256;++pair) for(unsigned f=0;f<4;++f) {
        rng=rng*1664525+1013904223;
        unsigned n=4*f, a=pair&15,b=pair>>4;
        uint32_t w=(rng&~(0x000f000fu<<n)) | (a<<n) | (b<<(16+n));
        unsigned packed=(low<8?0:low) | ((high<8?0:high)<<8);
        unsigned raw=(f&1)?high:low;
        uint32_t result=production(w,packed,f);
        uint32_t ea=expected(a,raw),eb=expected(b,raw);
        assert(result==(ea|(eb<<16)));++pairs;negzero+=(ea==32768)+(eb==32768);
    }
    std::printf("%llu %llu\n",(unsigned long long)pairs,(unsigned long long)negzero);
}
"""


def run():
    patch = (HERE.parent / "v147_integrated_experts_prefill/patch.diff").read_text()
    assert "if (scale_bytes > 0x7E).any().item():" in patch
    assert "scale_bytes.masked_fill(scale_bytes < 8, 0)" in patch
    results = {}
    with tarfile.open(HERE / "ablations.tar.gz") as artifacts, tempfile.TemporaryDirectory() as temp:
        for tag in TAGS:
            folder = Path(temp) / tag
            for path in (H,M):
                p=folder/path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(source_from_patch(patch,path))
            if tag != "base":
                delta = artifacts.extractfile(tag + ".diff").read()
                subprocess.run(["git","apply","--whitespace=error","-"],cwd=folder,input=delta,check=True)
            full=(folder/H).read_text()
            start=full.index("\n",full.index("constexpr uint32_t kPlannerWarps"))
            pre=full[start:full.index("struct Bank")]
            pre=pre.replace("static __device__ const PairTable", "static const PairTable")
            pre=re.sub(r"SGL_DEVICE void stage_table\(.*?\n}\n","",pre,flags=re.S)
            pre=re.sub(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',
                       r"\1 = pair_mul(\2, \3);",pre)
            pre=re.sub(r'asm\("prmt\.b32 [^\n]+;', 'result = cpu_prmt(lo, hi, selectors);', pre)
            assert "asm(" not in pre
            contract=full[full.index("SGL_DEVICE void contract("):]
            start=contract.index("\n",contract.index("uint32_t packed ="))
            body=contract[start:contract.index("    auto& d =")].replace("w[part]", "word")
            body=body.replace("const uint32_t word = word;", "")
            compute="uint32_t production(uint32_t word,uint32_t packed,uint32_t f){\n"
            if tag=="table":compute+="const uint32_t* table=kPairTable.entries;\n"
            compute+=body+"uint32_t a[]={a0,a1,a2,a3};return a[f];}\n"
            cpp=folder/"program.cc";cpp.write_text(CPP+pre+compute+DRIVER)
            exe=folder/"program"
            subprocess.run(["c++","-std=c++20","-O3",str(cpp),"-o",str(exe)],check=True)
            total,zero=map(int,subprocess.check_output([str(exe)]).split())
            results[tag]=dict(packed_pairs=total,signed_negative_zero_scalars=zero,
                              header_sha256=hashlib.sha256(full.encode()).hexdigest())
    return dict(raw_scale_encodings=256, rejected_by_unchanged_preparation=list(range(127,256)),
                raw_subnormal_flush_including_zero=list(range(8)),data=results,physical_gpu=False)


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
