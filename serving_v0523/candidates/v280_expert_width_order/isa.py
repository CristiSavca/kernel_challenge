"""Prove the new mapping and original arithmetic appear in the finalized SM90a binary."""
from pathlib import Path
from hashlib import sha256
import json
import re
import sys

here=Path(__file__).resolve().parent
parent=here.parent/"v270_wide_prompt_weight_reuse/isa.py"
scope=dict(__file__=str(parent))
exec(compile(parent.read_text().split("rows = {}",1)[0],str(parent),"exec"),scope)
images=Path(sys.argv[1])
name="OrderedPromptWeightPipeline"
sass=images/(name+".sass")
src=sass.with_suffix(".cu")
asm=sass.read_text()
log=sass.with_suffix(".log").read_text()
row=scope["inspect"](sass,src)
base=json.loads((here.parent/"v274_four_block_prompt_multicast/sm90a.json").read_text())["variants"]["WidePromptWeightPipeline4"]
assert row["shared_bytes"]==base["shared_bytes"] and row["stack_spill_store_load"]==[0,0,0]
assert all(x["hgmma"]==16 and x["drains"]==8 and x["local_loads"]==x["local_stores"]==0 for x in row["loops"].values())
assert {int(k) for k in base["loops"]}==row["loops"].keys() and row["multicast"]==base["multicast"]
static=int(re.search(r"(\d+) bytes smem",log)[1])
assert static==1024 and "extern __shared__ __align__(1024)" in src.read_text()
assert re.search(r"UIADD3 UR\d+, UR\d+, 0x400, URZ",asm)
tokens=("VOTE","SHFL.IDX","SHFL.UP","LDS.128","FENCE.VIEW.ASYNC.S","SYNCS.ARRIVE.TRANS64.RED.A1T0")
op={k:asm.count(k) for k in tokens}
assert all(op.values())
print(json.dumps(dict(toolchain="CUDA 12.9.86 SM90a",kernel=name,image=row,static_shared_bytes=static,
        total_static_plus_dynamic_shared_bytes=static+row["shared_bytes"],opcode_counts=op,production_constructor_sha256=sha256((here/"constructor.json").read_bytes()).hexdigest(),GPU=False),indent=2))
