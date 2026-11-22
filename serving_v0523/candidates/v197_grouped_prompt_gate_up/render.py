"""Run the actual frozen V195 tune selection, V197 compiler hook and Humming 0.1.12 CUDA generator."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace as S
from unittest.mock import patch
import torch
here=Path(__file__).resolve().parent
root=Path(sys.argv[1]).resolve()
stage=root/"sglang"
# The W13 decode renderer initializes real Humming types, dataclasses, tuner and launcher config.
with contextlib.redirect_stdout(io.StringIO()):
    selected=runpy.run_path(str(here.parent/"v191_decode_gate_up_epilogue/render.py"))
vendor=selected["driver"]
kernel=vendor["HummingKernel"];runtime=vendor["OfflineRuntime"];cfg=vendor["metas"]["w13"];compute=vendor["compute"]
tune=vendor["scope"]["Sm90Heuristics"]
all_tunings=tune.get_configs(layer_config=vendor["objects"][cfg],gemm_type=vendor["GemmType"].GROUPED_CONTIGUOUS)
prefill=json.dumps(all_tunings[-1][2]|{"use_stream_k":False,"use_tma_c":False})
small=[json.dumps(c) for _,_,c in selected["config"]]
mod=ModuleType("sglang.kernels.jit");mod.__file__=str(stage/"python/sglang/kernels/jit/__init__.py")
pkg=ModuleType("sglang.kernels");pkg.jit=mod;sys.modules[pkg.__name__]=pkg
u=ModuleType("sglang.kernels.jit.utils");u.load_jit=lambda *a,**kw:launcher;sys.modules[u.__name__]=u
co=ModuleType("humming.jit.compiler");co.NVCCCompiler=type("NVCCCompiler",(),{});sys.modules[co.__name__]=co
g=runpy.run_path(str(stage/"python/sglang/srt/layers/muse_spark/decode_gate_up.py"))["prepare_gate_up"].__globals__
compiled={}
def render(self):
    w=self.block_shape[0]
    name=("prompt" if self.block_shape[2]==64 else "decode")+f"_{w}"
    path=root/"final"/(name+".cu");path.parent.mkdir(exist_ok=True)
    path.write_text(self.code+"\nauto ptr = reinterpret_cast<void*>(&"+self.kernel_expr+");\n")
    self.func=w
    compiled[name]=dict(width=w,block=self.block_shape,warp=self.warp_shape,shared=self.estimated_smem_size,
        threads=self.num_threads,math_threads=self.num_math_threads,stages=self.num_stages)
runtime.prepare=render;runtime.check_context=lambda self:None
launcher=S(configure=lambda handle,shared:True)
with (patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
      patch.object(torch.cuda,"get_device_properties",return_value=S(multi_processor_count=132))):
    for t in small:assert g["_compile"](0,cfg,compute,t,False) is not None
    assert g["_compile"](0,cfg,compute,prefill,True) is not None
assert len(compiled)==4
print(json.dumps(dict(humming="0.1.12",source="real grouped tuner's final config; public BF16/NVFP4 W13",configs=compiled),indent=2))
