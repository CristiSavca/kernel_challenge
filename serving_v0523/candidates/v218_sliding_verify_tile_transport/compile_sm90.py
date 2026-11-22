"""Compile V218's submitted Triton kernels on an offline Linux CUDA toolkit.

Usage: python compile_sm90.py source_dir path/to/compile_decode_sm90.py cuobjdump_path
"""
from collections import Counter
from hashlib import sha256
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


def main():
    source, helper, disassembler = map(Path, sys.argv[1:4])
    sys.path.insert(0, str(helper.resolve().parent))
    from compile_decode_sm90 import load_jit_module
    ptxas = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
    records = []

    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        def compile_case(label, func, signature, constants, options):
            aligned = {(func.arg_names.index(k),): [["tt.divisibility", 16]]
                       for k, v in signature.items() if v.startswith("*")}
            signature.update({k: "constexpr" for k in constants})
            program = ASTSource(func, signature, constexprs=constants, attrs=aligned)
            binary = triton.compile(program, target=GPUTarget("cuda", 90, 32), options=options)
            ptx = binary.asm["ptx"]
            ptx_file, cubin_file = temp/"kernel.ptx", temp/"kernel.cubin"
            ptx_file.write_text(ptx)
            cubin_file.write_bytes(binary.asm["cubin"])
            stats = subprocess.run([ptxas, "-v", "-arch=sm_90a", ptx_file, "-o", temp/"assembled.cubin"],
                                   text=True, capture_output=True, check=True).stderr
            asm = subprocess.check_output([disassembler, "-sass", cubin_file], text=True)
            def inst(line):
                match = re.match(r"\s*/\*[a-fA-F0-9]+\*/\s*(?:@!?[A-Z0-9]+\s+)?(\S+)\s", line)
                return match.group(1) if match else ""
            ops = Counter(filter(None, map(inst, asm.splitlines())))
            matrix = {k:n for k,n in ops.items() if "MMA" in k and k.startswith(("HGMMA","HMMA","IMMA","QMMA"))}
            tensor = {k:n for k,n in ops.items() if "TMA" in k and "LDG" in k}
            dots = sorted(set(re.findall(r"(?:(?:wgmma\.mma_async)|(?:mma\.sync))[\w.]+",ptx)))
            spill = list(map(int, re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",stats).groups()))
            record = dict(label=label,shared_bytes=binary.metadata.shared,
                          registers=int(re.search(r"Used (\d+) registers",stats).group(1)),spill_store_load_bytes=spill,
                          cubin_sha256=sha256(binary.asm["cubin"]).hexdigest(),cubin_bytes=len(binary.asm["cubin"]),
                          sass_matrix=matrix,sass_tensor_loads=tensor,ptx_matrix_types=dots)
            records.append(record)
            return record

        path=source/"paged_verify_attention.py"
        producer=source/"metadata.py"
        module=load_jit_module(path,temp)
        kernel=module._partition_attention
        parent_path=source/"v215_paged.py"
        parent=load_jit_module(parent_path,temp)._partition_attention
        modes={}
        for batch in (1,4,16,32):
            for mode in ("v215","indexed","direct"):
                direct=mode=="direct"
                constants = dict(PAGE_ROW_STRIDE=131088,PAGE_SIZE=1,K_PAGE=128,K_TOKEN=128,K_HEAD=64,V_PAGE=128,
                                 V_TOKEN=128,V_HEAD=64,KS_ROW=0,KS_HEAD=0,B=batch,H_KV=2,GQA=8,D=64,
                                 SCALE_LOG2=math.log2(math.e)/8,WINDOW_LEFT=2047,MAX_Q=8,TILE_M=64,TILE_N=64,
                                 PARTITIONS=66,BLOCK_B=batch,ORIGIN_STRIDE=34 if direct else 0,
                                 CAPACITY=3420200 if direct else 0)
                signature=dict(zip(kernel.arg_names[:9],["*fp8e4nv"]*3+["*i32"]*3+["*fp32"]*3))
                if direct:
                    desc="tensordesc<fp8e4nv[64,64]>"
                    signature.update(Origins="*i32",KDesc=desc,VDesc=desc)
                else:
                    constants.update(Origins=None,KDesc=None,VDesc=None)
                record=compile_case(f"partition/b{batch}/{mode}",parent if mode=="v215" else kernel,signature,
                                    constants,dict(num_warps=4,num_stages=3))
                assert bool(record["sass_tensor_loads"]) == direct,record
                types=record["ptx_matrix_types"]
                assert any(".f32.f16.f16" in s or ".f32.e4m3.e4m3" in s for s in types),types
                assert any(".f32.bf16.bf16" in s for s in types),types
                assert record["sass_matrix"],record
                modes[batch,mode]=types
            assert modes[batch,"v215"] == modes[batch,"indexed"] == modes[batch,"direct"]
            fn=module._merge_partitions
            sig=dict(zip(fn.arg_names[:6],["*fp32","*fp32","*i32","*i32","*fp32","*bf16"]))
            cs=dict(VS_ROW=0,VS_HEAD=0,B=batch,H_KV=2,GQA=8,D=64,WINDOW_LEFT=2047,TILE_M=64,TILE_N=64,
                    PARTITIONS=66,BLOCK_B=batch,BLOCK_SPLITS=16)
            compile_case(f"merge/b{batch}/v_descale",fn,sig,cs,dict(num_warps=4,num_stages=3))

        fn=load_jit_module(producer,temp)._fused_decode_metadata_kernel
        for origins in ("both","swa_only","full_only"):
            for mapping in ("legacy","vmm"):
                const=dict(PAGE_SIZE=1,SEQ_LEN_DELTA=8,BLOCK_COLS=128,SKIP_PAGE_TABLE=False,ZERO_TAIL=False,
                           WINDOW_LEFT=-1,ORIGIN_STRIDE=1025 if origins!="swa_only" else 0,
                           SWA_ORIGIN_STRIDE=34 if origins!="full_only" else 0,SWA_WINDOW=2047,VERIFY_WIDTH=8)
                ptr=dict(seq_lens="*i32",req_to_token="*i32",req_pool_indices="*i64",cache_seqlens_int32="*i32",
                         cu_seqlens_k="*i32",page_table="*i32",swa_page_table="*i32")
                if mapping=="legacy":
                    ptr["full_to_swa_index_mapping"]="*i64"
                    const.update(full_v2p=None,swa_v2p=None)
                else:
                    const["full_to_swa_index_mapping"]=None
                    ptr.update(full_v2p="*i32",swa_v2p="*i32")
                for key,enabled in (("origins",origins!="swa_only"),("swa_origins",origins!="full_only")):
                    if enabled:ptr[key]="*i32"
                    else:const[key]=None
                signature={n:ptr.get(n,"i32") for n in fn.arg_names if n not in const}
                compile_case(f"producer/{origins}/{mapping}",fn,signature,const,dict(num_warps=4,num_stages=3))
    print(json.dumps(dict(triton=triton.__version__,target="sm90",physical_gpu_execution=False,
                          sliding_shape=dict(tensor_box=[64,64],query_heads=16,kv_heads=2,head_dim=64,
                                             window_left=2047,verify_width=8,partitions=66,
                                             k_descale="FP32",v_descale="FP32"),
                          input_sha256={p.name:sha256(p.read_bytes()).hexdigest()for p in (path,producer,parent_path,helper)},
                          partition_type_match={str(b):modes[b,"v215"]==modes[b,"indexed"]==modes[b,"direct"]
                                                for b in (1,4,16,32)},
                          profiles=records),indent=2))


if __name__=="__main__":
    main()
