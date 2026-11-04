#!/usr/bin/env python3
"""CPU audit of proposed NW4/NS2 small-M MoE, without creating a candidate."""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from test_wgs_kperm3_cpu import CpuTorch, Tensor, literal

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "muse_moe_w4a16/candidates/native_wgs_kperm3_v9.py"
OUT = ROOT / "experiments/iterations/20261104-small-moe-residency-cpu.json"


def namespace(tree):
    names = {"_stored_pos", "_stored_pos_mul", "_pack_codes", "_pack_scales", "_unit_order"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    ns = {"torch": CpuTorch, "DQ7": True, "_REG_HJ": [(0,0), (1,0), (0,2), (1,2)]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PARENT), "exec"), ns)
    ns["_POS"] = ns["_stored_pos"]()
    return ns


def packed_addresses(tree):
    ns = namespace(tree)
    rows = Tensor(np.arange(16))
    coordinates = np.arange(16*4096, dtype=np.uint32).reshape(16,4096)
    scale_coords = np.arange(16*256, dtype=np.uint32).reshape(16,256)
    code_checks = scale_checks = 0
    touched_codes, touched_scales, touched_x = set(), set(), set()
    for positions in (ns["_stored_pos"](), ns["_stored_pos_mul"]()):
        for pattern in range(4):
            values = (coordinates >> (4*pattern)) & 15
            codes = Tensor((values[:,::2] | (values[:,1::2]<<4)).astype(np.uint8))
            scales = Tensor(((scale_coords >> (8*(pattern%2))) & 255).astype(np.uint8))
            packed = ns["_unit_order"](ns["_pack_codes"](codes, rows, positions),
                                       ns["_pack_scales"](scales, rows)).a
            assert packed.size == 36864
            for warp in range(4):
                for lane in range(32):
                    g,t = lane//4,lane%4
                    for c in range(16):
                        # New consumer merges two consecutive original NW8 warp
                        # regions; each retains its original 8-chunk scale record.
                        original_warp = 2*warp+c//8
                        for h in range(2):
                            si = 32768+original_warp*512+lane*16+(c%8)*2+h
                            assert packed[si] == scales.a[g+8*h,warp*64+c*4+t]
                            touched_scales.add(si)
                            scale_checks += 1
                        for st in range(4):
                            wi = (warp*16+c)*512+lane*16+st*4
                            word = int.from_bytes(packed[wi:wi+4].tobytes(), "little")
                            for reg,(h,j0) in enumerate(ns["_REG_HJ"]):
                                for upper in range(2):
                                    got = sum(((word>>positions[reg,upper,b])&1)<<b for b in range(4))
                                    row = g+8*h
                                    col = warp*1024+c*64+t*16+st*4+j0+upper
                                    assert got == values[row,col]
                                    touched_codes.add((row,col))
                                    code_checks += 1
                        for pair in range(2):
                            for h in range(8):
                                col = warp*1024+c*64+t*16+pair*8+h
                                touched_x.add((g,col))
    assert len(touched_codes)==16*4096 and len(touched_scales)==4096 and len(touched_x)==8*4096
    return dict(code_comparisons=code_checks, scale_comparisons=scale_checks,
                distinct_weight_coordinates=len(touched_codes), distinct_scale_bytes=len(touched_scales),
                distinct_activation_coordinates=len(touched_x),
                encoding_patterns="four coordinate nibbles uniquely identify all 65536 coordinates; both inherited bit layouts")


def resources(src):
    a=src.index("struct Smem {"); b=src.index("\n};",a)+3
    result=[]
    with tempfile.TemporaryDirectory(prefix="okc-small-moe-audit-") as td:
        for nw in (8,4):
            cpp=("#include <cstdint>\n#include <cstdio>\n#define OKC_WGS 0\n#define OKC_WS 1\n"
                 f"constexpr int NW={nw},NRB=4,RED_W=4,JMAX=88,EMAX=3072;\n"+src[a:b]+
                 '\nint main(){printf("%zu\\n",sizeof(Smem));}\n')
            exe=str(Path(td)/f"size{nw}")
            subprocess.run(["clang++","-x","c++","-std=c++17","-o",exe,"-"],
                           input=cpp,text=True,capture_output=True,check=True)
            size=int(subprocess.check_output([exe],text=True)); rounded=(size+127)//128*128
            result.append(dict(NW=nw, threads=32*(nw+1), metadata=size, rounded_metadata=rounded,
                current_host_dynamic=rounded+3*36864+2*64*264*2,
                tiny_only_NS3_dynamic=rounded+3*36864,
                tiny_only_NS2_dynamic=rounded+2*36864,
                two_NS2_with_static=2*(rounded+2*36864+1024), activation_cache_registers=8*(64//nw)))
    return result


def schedule():
    unit_checks=0; range_checks=0
    for jobs in range(45):
        for grid in (132,264,396,528):
            credit=[0]*jobs
            for c in range(grid):
                a0,a1=(jobs*512*c//grid,jobs*512*(c+1)//grid)
                b0,b1=(jobs*256*c//grid,jobs*256*(c+1)//grid)
                na,nb=a1-a0,b1-b0
                assert na>0 or nb==0
                issued=list(range(min(2,na)))
                issued+=list(range(len(issued),min(2,na+nb)))
                issued+=list(range(2,na+nb))
                assert issued == list(range(na+nb))
                for tail in (1,2):
                    nh=na+nb-min(tail,nb)
                    assert nh>=na
                    helper=set(range(nh))
                    consumers=set(range(nh,na+nb))
                    assert len(helper|consumers)==na+nb and not(helper&consumers)
                for u in range(a0,a1): credit[u//512]+=1
                unit_checks+=na+nb; range_checks+=1
            assert credit==[512]*jobs
    examples=[]
    for m,counts in ((1,[1]),(1,[1]*8),(8,[8]*8),(8,[2]*16),(16,[16]*8),
                     (16,[16]+[1]*20),(16,[9]*12+[1]*20)):
        rows=[min(8,c-s) for c in counts for s in range(0,c,8)]
        traffic=[]
        for grid in (132,264):
            byte_reads=0; touches=0
            for units in (512,256):
                for block in range(grid):
                    lo,hi=len(rows)*units*block//grid,len(rows)*units*(block+1)//grid
                    if lo==hi:continue
                    for j in range(lo//units,(hi-1)//units+1):
                        byte_reads+=rows[j]*4096*2; touches+=1
            traffic.append(dict(grid=grid, activation_cache_load_bytes=byte_reads, job_phase_cta_touches=touches))
        examples.append(dict(M=m,counts=counts,jobs=len(rows),weight_bytes=len(rows)*27*1024**2,
                             tma_copies=len(rows)*768,activation_reads=traffic))
    # Finalizer always keeps four output columns per thread.
    for m in range(17):
        for nw,grid in ((8,132),(4,264)):
            nt=nw*32; fseg=4096//(nt*4)
            covered=set()
            for b in range(grid):
                for item in range(b,m*fseg,grid):
                    token,segment=divmod(item,fseg)
                    for tid in range(nt):
                        for k in range(4): covered.add((token,segment*nt*4+tid*4+k))
            assert len(covered)==m*4096
    return dict(job_count_range=[0,44],grids=[132,264,396,528],range_checks=range_checks,
        unit_ownership_checks=unit_checks,activation_examples=examples,
        proof_limit="Abstract ownership and phase DAG assuming all CTAs resident; not generated CUDA execution or numerical reduction proof.")


def main():
    content=PARENT.read_bytes();tree=ast.parse(content)
    observed=json.loads((ROOT/"experiments/modal/20261104T052256Z-4c806134/result.json").read_text())
    registers=[dict(extension=r["extension"],defines=r["defines"],resource_output=r["output"])
               for r in observed["extension_resources"]["baseline"]]
    result=dict(parent_sha256=hashlib.sha256(content).hexdigest(),candidate_created=False,gpu_used=False,
        original_ef7_resources=registers, packed_address_proof=packed_addresses(tree),
        resource_layout=resources(literal(tree,"CUDA_SRC")),schedule=schedule(),
        profile_share_scenarios=[dict(small_moe_latency_reduction=f,
            aggregate_decoder_latency_reduction=1-math.prod(1-f*s for s in (.30,.49,.46))**(1/6))
            for f in (.2,.25,.4,.5)])
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("original_ef7_resources","schedule")},indent=2))


if __name__ == "__main__":main()
