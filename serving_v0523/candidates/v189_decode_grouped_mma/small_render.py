"""Render exact V68/V187 decode planner configurations through Humming 0.1.12."""
import ast
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as S
from unittest.mock import patch
import torch

root=Path(sys.argv[1]).resolve()
here=Path(__file__).resolve().parent.parent
stage=root/"sglang"
# The V169 codegen driver initializes the real Humming dataclasses and CUDA templates.
policy="python/sglang/srt/layers/muse_spark/prefill_expert_tiles.py"
diff=(here/"v169_grouped_expert_tile_cohorts/patch.diff").read_text()
body=diff.split(f"diff --git a/{policy} b/{policy}\n")[1].split("diff --git ")[0]
p=root/"fixtures"/policy;p.parent.mkdir(exist_ok=True,parents=True)
p.write_text("".join(x[1:] for x in body.splitlines(keepends=True) if x.startswith("+") and not x.startswith("+++")))
sys.argv=["init",str(root/"humming"),str(root/"fixtures"),str(root/"setup")]
with contextlib.redirect_stdout(io.StringIO()):
    driver=runpy.run_path(str(here/"v169_grouped_expert_tile_cohorts/vendor_codegen.py"))
g=driver["scope"]["Sm90Heuristics"];meta=driver["objects"][driver["metas"]["w2"]]
compute=driver["compute"];typ=driver["GemmType"].GROUPED_CONTIGUOUS;kernel=driver["HummingKernel"]
native=driver["OfflineRuntime"]
all_configs=g.get_configs(layer_config=meta,gemm_type=typ)
path=stage/"python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
mod=ast.parse(path.read_text())
limit=next(n.value.value for n in mod.body if isinstance(n,ast.Assign) and n.targets[0].id=="_MAX_DECODE_SLOTS")
shape_m=limit*meta.num_experts//256
config=[(a,b,c|{"use_stream_k":False,"use_tma_c":False}) for a,b,c in all_configs if a<shape_m]
out=root/"original";out.mkdir(exist_ok=True)
rec={}
def render(self):
    w=self.block_shape[0]
    name=f"4096_{w}"
    (out/(name+".cu")).write_text(self.code+"\nauto ptr = reinterpret_cast<void*>(&"+self.kernel_expr+");\n")
    rec[name]=dict(block=self.block_shape,warp=self.warp_shape,stages=self.num_stages,threads=self.num_threads,
        math_threads=self.num_math_threads,shared=self.estimated_smem_size,use_warp_spec=self.use_warp_spec,
        tma=[self.use_tma_a,self.use_tma_b,self.use_tma_bs],sm_version=self.sm_version)
with (patch.object(native,"prepare",render),patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
      patch.object(torch.cuda,"get_device_properties",return_value=S(multi_processor_count=132))):
    for low,hi,conf in config:
        kernel(**(json.loads(driver["metas"]["w2"])|json.loads(compute)|conf))
        rec[f"4096_{conf['block_shape'][0]}"]["bounds"]=[low,hi]
paired=root/"paired";paired.mkdir(exist_ok=True)
header=stage/"python/sglang/kernels/jit/csrc/muse_spark/grouped_decode_mma_pair.cuh"
for name,shape in rec.items():
    native=(out/(name+".cu")).read_bytes()
    first,expr=native.split(b"\nauto ptr = ")
    width=shape["block"][0]
    (paired/(name+".cu")).write_bytes(first+f"\n#define MUSE_OVERLAP_M {width}\n".encode()+header.read_bytes()+b"\nauto ptr = "+expr)

print(json.dumps(dict(configs=rec,grouped_decode_max_slots=limit,estimated_rank_max=shape_m,rungs={
    n:next(c["block_shape"][0] for a,b,c in config if n*8*8*32//256<=b) for n in [1,2,4,8,16,32]}),indent=2))
