"""Run production prepare_down, vendor HummingKernel construction and native cubin registration on CPU."""
import os
import ast
import struct
import contextlib
import hashlib
import inspect
import io
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace as S
from unittest.mock import patch

import torch

here=Path(__file__).resolve().parent
orig_args=sys.argv[1:]
scratch=Path(sys.argv[1])
stage=scratch/"sglang"
sys.argv=["render",str(scratch)]
src=(here/"render.py").read_text()
vendor={"__file__":str(here/"render.py"),"__name__":"render_fixture"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src[:src.index("    try:\n        HummingKernel")],str(here/"render.py"),"exec"),vendor)
paired=runpy.run_path(str(stage/"python/sglang/srt/layers/muse_spark/prompt_weight_pipeline.py"))
native=vendor["HummingKernel"];runtime=vendor["OfflineRuntime"]
comp=ModuleType("humming.jit.compiler")
comp.NVCCCompiler=type("NVCCCompiler",(),{"_get_env":staticmethod(lambda:{"binaries":{"cuobjdump":"objdump"}})})
sys.modules[comp.__name__]=comp
ops=ModuleType("humming.ops");sys.modules["humming"].ops=ops;sys.modules[ops.__name__]=ops
jit=ModuleType("sglang.kernels.jit");jit.__file__=str(stage/"python/sglang/kernels/jit/__init__.py")
pkg=ModuleType("sglang.kernels");pkg.jit=jit;sys.modules[pkg.__name__]=pkg
# Read the real wheel's cubin kernel names parser (Python standard library).
elf={"struct": struct, "Path": Path, "os": os}
mod=ast.parse((scratch/"humming/utils/cubin.py").read_text())
mod.body=[n for n in mod.body if getattr(n,"name",None) == "get_cubin_kernel_names"]
exec(compile(mod,"humming/utils/cubin.py","exec"),elf)
names=elf["get_cubin_kernel_names"]
rendered=[];registered=[];bad_image=False

def prepare(self):
    w=self.block_shape[0]
    full=self.code+"\nauto ptr = reinterpret_cast<void*>(&"+self.kernel_expr+");\n"
    p=scratch/"rendered"/"pipeline.cu"
    if "--emit" in orig_args:
        p.write_text(full)
    assert full==p.read_text(),p
    self.kernel_filename=str(p.with_suffix(".cubin"));self.cubin_loaded=False
    self.load_cubin()
    rendered.append(w)

def disassemble(cmd,text):
    assert cmd[:2]==["objdump","-sass"] and text
    if bad_image: return "BPT.TRAP"
    return Path(cmd[-1]).with_suffix(".sass").read_text()

def register(file):
    matches=[x for x in names(file) if "humming" in x]
    assert len(matches)==1
    registered.append(Path(file).name)
    return 40+len(registered),matches[0]

runtime.prepare=prepare
runtime.check_context=lambda self: self.load_cubin()
ops.register_kernel=register
c=json.dumps(vendor["cfg"]);meta=vendor["obj"].to_str()
compute=json.dumps({"gemm_type":"grouped_contiguous","use_f16_accum":False})
layer=S(w2_weight=torch.ones((2,4),dtype=torch.int32),w2_weight_scale=torch.ones((2,8),dtype=torch.uint8))
with (patch("subprocess.check_output",side_effect=disassemble),
      patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
      patch.object(torch.cuda,"get_device_properties",return_value=S(multi_processor_count=132))):
    paired["prepare_prompt_down"].__globals__["version"]=lambda name:"0.1.12"
    state=paired["prepare_prompt_down"](layer,meta,compute,c)
    assert state is not None,(json.loads(meta),json.loads(c))
    assert state[0].tolist()==[0,1<<30,41,0] and registered==["pipeline.cubin"] and rendered==[176]
    assert paired["prepare_prompt_down"](layer,meta,compute,c)[0] is state[0]
    bad_image=True
    assert paired["_prepare"](23,meta,compute,c) is None and registered==["pipeline.cubin"]
    denied=[(k,v) for k,v in dict(a_dtype="float8e4m3",has_bias=True,weight_scale_2_type="channel",pad_shape_n=16,
          use_packed_k_layout=True,weight_scale_group_size=32).items()]
    for k,v in denied:
        assert paired["_prepare"](23,json.dumps(dict(json.loads(meta),**{k:v})),compute,c) is None
    for k,v in {"block_shape":[16,256,128],"num_stages":3,"use_tma":False,"num_sms":12,"multi_cast_size_b":2}.items():
        assert paired["_prepare"](23,meta,compute,json.dumps(dict(json.loads(c),**{k:v}))) is None
print(json.dumps(dict(rendered_and_registered=registered,admitted_native_config=state[0].tolist(),
    verified_wheel_symbol=True,bad_instruction_stream_refused=True,incompatible_configs_declined=len(denied)+5,
    context_cache_hits=True,real_cuda_driver=False,
    adapter_sha256=hashlib.sha256((stage/"python/sglang/srt/layers/muse_spark/prompt_weight_pipeline.py").read_bytes()).hexdigest()),indent=2))
