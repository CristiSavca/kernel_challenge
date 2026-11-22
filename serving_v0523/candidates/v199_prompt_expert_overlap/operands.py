"""Replay production scale staging and transform_b with the published Humming bit unpacker."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

scratch=Path(sys.argv[1])
stage=scratch/"sglang/python/sglang/kernels/jit/csrc/muse_spark/prompt_weight_pipeline.cuh"
vendor=scratch/"humming/include/humming"
here=Path(__file__).resolve().parent
header=stage.read_text()
def method(source,name):
    a=source.index(name)
    left=source.index("{",a)
    depth=0
    for i in range(left,len(source)):
        depth+=int(source[i]=="{")-int(source[i]=="}")
        if not depth:
            return source[a:i+1]
    raise ValueError(name)
core=method(header,"CUDA_INLINE void transform_b")
assert core.count('asm("mul.rn.bf16x2')==2
for dest,b,c in (("temp","value","0x7d807d80u"),("value","temp","i % 2 ? odd : even")):
    asm='asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"('+dest+') : "r"('+b+'), "r"('+c+'));'
    assert core.count(asm)==1
    core=core.replace(asm,dest+' = mul2('+b+', '+c+');')
source=(here/"operands.cpp").read_text()
fragments={"@@LOADER@@":header[:header.index("template <class Ctx, class Arith>")],
           "@@TRANSFORM@@":core,
           "@@CONVERT@@":method((vendor/"datatype/dequant_single.cuh").read_text(),"CUDA_INLINE uint32_t fp_to_fp"),
           "@@UNPACK@@":method((vendor/"datatype/dequant.cuh").read_text(),"CUDA_INLINE void dequant_b1248")}
for k,v in fragments.items(): source=source.replace(k,v)
with tempfile.TemporaryDirectory(dir=scratch) as d:
    path=Path(d)/"test.cpp";path.write_text(source)
    binary=path.with_suffix("")
    subprocess.run(["c++","-std=c++20","-O2","-fno-strict-aliasing","-fno-fast-math",path,"-o",binary],check=True)
    result=json.loads(subprocess.check_output([binary],text=True))
result.update(header_sha256=hashlib.sha256(header.encode()).hexdigest(),driver="portable production loader and all transform_b instructions with independent non-FTZ RN-even BF16 multiply and byte permute implementations",
              native_dequant_sources={f:hashlib.sha256((vendor/("datatype/"+f)).read_bytes()).hexdigest() for f in ("dequant.cuh","dequant_single.cuh")})
print(json.dumps(result,indent=2))
