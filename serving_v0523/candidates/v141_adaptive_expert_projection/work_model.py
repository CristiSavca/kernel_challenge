"""Execute the pinned H200 indexed tuning body and compare exact route-tile geometry."""
import ast
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np

vendor=Path(sys.argv[1])
tune=vendor/"humming/tune/sm90.py"
text=tune.read_text()
src=ast.parse(text)
src.body=[ast.parse("from __future__ import annotations").body[0],next(n for n in src.body if isinstance(n,ast.ClassDef))]
d=NS(float16=1,bfloat16=NS(num_bits=16),int8=2,float8e4m3=3,float8e5m2=4)
modes=NS(DENSE="dense",INDEXED="indexed")
env=dict(np=np,math=math,dtypes=d,DeviceHeuristics=object,GemmType=modes)
exec(compile(src,str(tune),"exec"),env)
heur=env["Sm90Heuristics"]
root=Path(__file__).resolve().parents[3]
diag=json.loads((root/"experiments/iterations/20261104-moe-hardware-profile-recovered.json").read_bytes())
cases=[]
for M in (64,128,256):
    cfg={}
    for n,name in ((8192,"w13"),(4096,"w2")):
        layer=NS(a_dtype=d.bfloat16,b_dtype=NS(num_bits=4),shape_n=n,shape_k=4096,num_experts=32,use_packed_k_layout=False)
        cfg[name]=heur.get_config(layer,M,gemm_type=modes.INDEXED)|{"use_stream_k":False}
    # get_configs enumerates all multiples of eight up to 1024; these exact
    # shape samples therefore produce the config returned by the interval table.
    for distribution in ("uniform-distinct", "diagnostic"):
        if distribution=="diagnostic":
            if M!=256:continue
            c=next(x for x in diag["cases"] if x["M"]==256)["routing"]["local_counts"]
            expected=lambda f:sum(f(n) for n in c)
        else:
            prob=8/256
            expected=lambda f:sum(32*math.comb(M,n)*prob**n*(1-prob)**(M-n)*f(n) for n in range(M+1))
        def stats(name,cap):
            block_m=cfg[name]["block_shape"][0]
            vendor_jobs=expected(lambda x:math.ceil(x/block_m))
            jobs=expected(lambda x:math.ceil(x/cap))
            def mma_rows(n):
                tails=n%cap
                return n//cap*cap+ (0 if tails==0 else 8 if tails<=8 else 16 if tails<=16 else 32)
            vendor_rows=vendor_jobs*block_m
            return dict(parent_block_m=block_m,parent_block_n=cfg[name]["block_shape"][1],
                        parent_block_k=cfg[name]["block_shape"][2],parent_aligned_rows=vendor_rows,
                        candidate_mma_rows=expected(mma_rows),parent_weight_tiles=vendor_jobs,
                        candidate_weight_tiles=jobs,parent_nominal_packed_mib=vendor_jobs*(18 if name=="w13" else 9),
                        candidate_nominal_packed_mib=jobs*(18 if name=="w13" else 9))
        cases.append(dict(M=M,distribution=distribution,live=expected(lambda x:x),
               active_experts=expected(lambda x:bool(x)),w13=stats("w13",16),w2=stats("w2",32),
               per_rank_plan_bytes=32*(8*M+1)*4,
               old_reserved_activation_and_scratch_mib=24*M*8/1024,
               new_reserved_activation_output_and_plan_mib=16*M*8/1024+32*(8*M+1)*4/2**20))
print(json.dumps(dict(source_archive_sha256="c96c37c95f74379067d2d2d0c5154d12ee9a8cfa29a7c3370a7f3e34f207ca72",
     actual_vendor_tuning_sha256=sha256(text.encode()).hexdigest(),model_only=True,
     limitations="Weight tiles assume no inter-CTA reuse and one complete packed bank traversal per M tile. This is not measured HBM or serving speed.",
     cases=cases),indent=2))
