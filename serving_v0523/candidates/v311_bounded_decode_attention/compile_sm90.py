"""Compile the actual before/after verification kernels with released Triton and Hopper ptxas."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from compile_decode_sm90 import load_jit_module

def main(root):
    cuobjdump=Path.home()/"nvcc126/toolchain/usr/local/cuda-12.9/bin/cuobjdump"
    ptxas=Path(triton.__file__).parent/"backends/nvidia/bin/ptxas"
    result={}
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        temp=Path(tmp)
        modules={n:load_jit_module(root/(n+".py"),temp) for n in ("parent","candidate")}
        def compile_kernel(fn, signature, constants):
            alignment = {(fn.arg_names.index(k),): [["tt.divisibility", 16]]
                         for k, dtype in signature.items() if dtype.startswith("*")}
            binary=triton.compile(ASTSource(fn, signature=signature, constexprs=constants, attrs=alignment),target=GPUTarget("cuda",90,32),
                                options=dict(num_warps=4,num_stages=3))
            ptx=binary.asm["ptx"]
            src=temp/"kernel.ptx";src.write_text(ptx);output=temp/"kernel.cubin"
            output.write_bytes(binary.asm["cubin"])
            asm=subprocess.check_output([cuobjdump,"-sass",output],text=True)
            stats=subprocess.run([ptxas,"-v","-arch=sm_90a",src,"-o",temp/"check.cubin"],capture_output=True,text=True,check=True).stderr
            op=re.compile(r"^\s*/\*[a-fA-F0-9]+\*/\s*(?:@!?[A-Z0-9]+\s+)?([A-Z][\w.]+)\b",re.MULTILINE)
            counts=Counter(op.findall(asm))
            reductions = []
            fp_ops = {}
            if fn.__name__ == "_merge_partitions":
                ir = binary.asm["ttgir"]
                layouts = dict(line.split(" = ",1) for line in ir.splitlines() if line.startswith("#blocked"))
                for operation, operand in re.findall(r'"tt.reduce"\((.*?)\n\s*}\) : \(tensor<(.*?)>\) ->', ir, re.S):
                    if "xf32" in operand:
                        reductions.append((re.search(r"= (arith\.\w+) ",operation)[1],
                            re.sub(r"#blocked\d*", lambda m: layouts[m.group()], operand)))
                opcodes = Counter(re.findall(r"\b(?:fma|add|ex2|shfl)\.[^\s,;]+", ptx))
                fp_ops = {n: opcodes[n] for n in ("fma.rn.f32", "add.f32", "ex2.approx.ftz.f32", "shfl.sync.bfly.b32")}
                assert len(reductions) == 3 and all(fp_ops.values())
            return dict(fp32_reduction_layouts=reductions, merge_fp32_ops=fp_ops, registers=int(re.search(r"Used (\d+) registers",stats)[1]),
                spill_store_load_bytes=list(map(int,re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",stats).groups())),
                sass_instructions=sum(counts.values()),static_branches=sum(v for k,v in counts.items() if k.startswith("BRA")),
                global_loads=sum(v for k,v in counts.items() if k.startswith("LDG")),
                mma_types=sorted({k for k in counts if "MMA" in k}),tensor_map_loads=sum(v for k,v in counts.items() if k.startswith("UTMALDG")),
                shared_bytes=binary.metadata.shared,cubin_sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest())
        shapes = [(batch, window, 131088) for batch in (1, 4, 16, 32) for window in (-1, 2047)]
        shapes.extend([(1, -1, 2048), (2, 127, 131088)])
        for batch,window,cap in shapes:
            tile=64 if window>=0 else 128
            parts=66 if window>=0 else 132
            tab=(cap+tile-1)//tile
            bound=batch*(min(tab,(window+8+2*tile-2)//tile) if window>=0 else tab)
            for ver,mod in modules.items():
                part=min(parts,bound) if ver=="candidate" else parts
                common=dict(B=batch,H_KV=2,GQA=8,D=64,WINDOW_LEFT=window,TILE_M=64,TILE_N=tile,PARTITIONS=part,BLOCK_B=triton.next_power_of_2(batch))
                for direct in ((False, True) if window < 0 else (False,)):
                    fn=mod._partition_attention
                    types=dict(zip(fn.arg_names[:9],["*fp8e4nv"]*3+["*i32"]*3+["*fp32"]*3))
                    args=dict(common, PAGE_ROW_STRIDE=cap,PAGE_SIZE=1,K_PAGE=128,K_TOKEN=128,K_HEAD=64,
                              V_PAGE=128,V_TOKEN=128,V_HEAD=64,KS_ROW=0,KS_HEAD=0,SCALE_LOG2=math.log2(math.e)/8,
                              MAX_Q=8,ORIGIN_STRIDE=34 if window==2047 else 1025,CAPACITY=3420200)
                    args.update(ORIGIN_STRIDE=triton.cdiv(cap, 128) if window < 0 else 0,
                                CAPACITY=5005915 if window < 0 else 0)
                    if direct:types.update(Origins="*i32",KDesc=f"tensordesc<fp8e4nv[{tile},64]>",VDesc=f"tensordesc<fp8e4nv[{tile},64]>")
                    else:args.update(Origins=None,KDesc=None,VDesc=None)
                    r=compile_kernel(fn,types,args)
                    assert r["mma_types"] and bool(r["tensor_map_loads"])==direct,r
                    result[f"b{batch}/w{window}/c{cap}/partition/{'direct' if direct else 'indexed'}/{ver}"]=r
                fn=mod._merge_partitions
                types=dict(zip(fn.arg_names[:6],["*fp32","*fp32","*i32","*i32","*fp32","*bf16"]))
                if ver == "candidate":
                    programs = max(1, batch * 2 * 64 // 132)
                    common["ROWS"] = min(4, 1 << (programs.bit_length() - 1))
                result[f"b{batch}/w{window}/c{cap}/merge/{ver}"] = compile_kernel(
                    fn, types, dict(common, VS_ROW=0, VS_HEAD=0, BLOCK_SPLITS=16)
                )
    for name, old in result.items():
        if name.endswith("/merge/parent"):
            new = result[name[:-len("parent")] + "candidate"]
            assert old["fp32_reduction_layouts"] == new["fp32_reduction_layouts"]
            assert old["merge_fp32_ops"] == new["merge_fp32_ops"]
            assert old["spill_store_load_bytes"] == new["spill_store_load_bytes"] == [0, 0]
    return dict(triton=triton.__version__,source_sha256={n:hashlib.sha256((root/(n+".py")).read_bytes()).hexdigest() for n in ("parent","candidate")},
                builds=result,physical_gpu=False,target="sm90a",page_size=1,
                direct_descriptor="full attention only; page size one", duplicate_compile_cases_excluded=True)
if __name__=="__main__":
    print(json.dumps(main(Path(sys.argv[1]).resolve()),indent=2))
