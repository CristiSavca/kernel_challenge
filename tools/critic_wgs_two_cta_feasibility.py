#!/usr/bin/env python3
"""Read-only CPU bounds for an unimplemented WGS residency proposal."""
import ast,hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
P=R/"muse_moe_w4a16/candidates/native_wgs_kperm3_v9.py"
CA=[112,135,140,158,173,185,200,207]
CB=[100,104,115,123,132,144,156,174]
def weight(c,ft,cost):
    full,rem=divmod(c,ft)
    return full*cost[ft//8-1]+(cost[(rem+7)//8-1] if rem else 0)
def maximum(m,ft,cost):
    n=8*m;lo=-10**9
    dp=np.full(n+1,lo,dtype=np.int64);dp[0]=0
    choices=[]
    for expert in range(32):
        nxt=np.full(n+1,lo,dtype=np.int64);pick=np.zeros(n+1,dtype=np.int16)
        for c in range(m+1):
            trial=dp[:n+1-c]+weight(c,ft,cost)
            better=trial>nxt[c:]
            nxt[c:][better]=trial[better];pick[c:][better]=c
        dp=nxt;choices.append(pick)
    k=int(np.argmax(dp));answer=int(dp[k]);hist=[]
    for pick in reversed(choices):c=int(pick[k]);hist.append(c);k-=c
    assert k==0 and sum(hist)<=8*m and max(hist)<=m
    assert sum(weight(c,ft,cost) for c in hist)==answer
    return answer,hist

def main():
    b=P.read_bytes();tree=ast.parse(b)
    src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="CUDA_SRC")
    sa=src.index("struct Smem {");sb=src.index("\n};",sa)+3
    rows=[]
    with tempfile.TemporaryDirectory(prefix="okc-wgs-bound-") as tmp:
        for m in (32,128,256,384):
            for ft in (64,32,16):
                emax=8*m;jmax=32+emax//ft+8
                cpp="#include <cstdint>\n#include <cstdio>\n#define OKC_WGS 1\n#define OKC_WS 0\n"+f"constexpr int NW=16,NRB=2,RED_W=4,JMAX={jmax},EMAX={emax};\n"+src[sa:sb]+"\nint main(){printf(\"%zu\\n\",sizeof(Smem));}\n"
                exe=str(Path(tmp)/"size")
                subprocess.run(["clang++","-x","c++","-std=c++17","-o",exe,"-"],input=cpp,text=True,capture_output=True,check=True)
                size=int(subprocess.check_output([exe],text=True));rounded=(size+127)//128*128
                aw,ah=maximum(m,ft,CA);bw,bh=maximum(m,ft,CB)
                dyn=rounded+2*36864+2*ft*512
                rows.append(dict(MMAX=m,FT=ft,EMAX=emax,JMAX=jmax,metadata=size,rounded_metadata=rounded,dynamic_two_rings=dyn,
                   two_ctas_with_1KiB_each=2*(dyn+1024),shared_fits_using_1KiB_each=2*(dyn+1024)<=228*1024,
                   maximum_phaseA_cost=aw,maximum_phaseA_times512_times264=aw*512*264,phaseA_histogram=ah,
                   maximum_phaseB_cost=bw,maximum_phaseB_times256_times264=bw*256*264,phaseB_histogram=bh))
    # Valid M32 histogram c_e=8 for all32experts: uniform112 stage costs.
    grid=264;stages=32*512;slots=set()
    boundaries=[stages*c//grid for c in range(grid+1)]
    for c in range(grid):
        lo,hi=boundaries[c:c+2]
        for u in range(lo//16,(hi-1)//16+1):
            a=max(lo,u*16);z=min(hi,(u+1)*16)
            if z-a<16:
                owner=max(i for i in range(grid) if boundaries[i]<=u*16)
                slots.add(owner)
    result=dict(parent_sha256=hashlib.sha256(b).hexdigest(),candidate_created=False,gpu_used=False,
      register_ceiling=dict(threads=576,two_cta_threads=1152,registers_per_sm=65536,aggregate_max_integer_per_thread=65536//1152,
          caveat="Necessary aggregate ceiling only; allocation granularity, partitions and actual CUDA occupancy still apply."),
      shared_rows=rows,fixup_counterexample=dict(M=32,counts=[8]*32,grid=264,distinct_split_slots=len(slots),max_slot=max(slots),
          slots_past_parent160=sum(s>=160 for s in slots),proposed_min_slots=265),
      register_source_inventory=dict(gather_GB8_value_words=32,gather_GB8_pointer_words=16,
          async_AF_GS2_two_groups=16,accumulator_words_FT16=8,accumulator_words_FT32=16,accumulator_words_FT64=32,
          caveat="Source live-value inventory under retained scheduling, NOT an architectural register lower bound; allocator can recompute, spill, serialize or use uniform registers."),
      findings=["FT64 two rings alone exceed half-SM shared capacity, before metadata.",
        "FT16 requires NTT3/4 dispatch guards; current FT>8 branch instantiates them despite two-tile arrays.",
        "Per-shape EMAX/JMAX change global scratch pointer offsets as well as Smem; host and all participating kernels must share exact layouts.",
        "V8 split-finalizer capture has no compiler/resource measurements, so cannot attribute MAIN96regs to its finalizer.",
        "Changing FT can multiply logical weight requests/stages up to2x (FT32) or4x (FT16) versusFT64; actual DRAM effect depends on cache reuse."],
      primary_sources=["https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy",
        "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy"],
      decision="NO GO for implementation/GPU from residency arithmetic alone; feasible shared rows do not establish<=56 actual regs without destructive spilling/serialization or useful performance.")
    out=R/"experiments/iterations/20261104-wgs-two-cta-feasibility-critic.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("shared_rows",)},indent=2))
    for row in rows:print(row["MMAX"],row["FT"],row["dynamic_two_rings"],row["shared_fits_using_1KiB_each"],row["maximum_phaseA_times512_times264"])

if __name__=="__main__":main()
