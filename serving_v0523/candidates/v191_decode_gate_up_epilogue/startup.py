"""Run the production Python setup against the pinned Humming generator and its nine real cubins."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import struct
import sys
from types import ModuleType, SimpleNamespace as S
from unittest.mock import patch

import torch

here=Path(__file__).resolve().parent
scratch=Path(sys.argv[1]).resolve()
stage=scratch/"sglang"
sys.argv=["render",str(scratch)]
with contextlib.redirect_stdout(io.StringIO()):
    small=runpy.run_path(str(here/"render.py"))
native=small["driver"]["OfflineRuntime"]; vendor=small["driver"]
jit=ModuleType("sglang.kernels.jit");jit.__file__=str(stage/"python/sglang/kernels/jit/__init__.py")
utils=ModuleType("sglang.kernels.jit.utils");utils.load_jit=lambda *a,**kw:launcher
sys.modules[utils.__name__]=utils
pkg=ModuleType("sglang.kernels");pkg.jit=jit;sys.modules[pkg.__name__]=pkg
comp=ModuleType("humming.jit.compiler")
comp.NVCCCompiler=type("NVCCCompiler",(),{"_get_env":staticmethod(lambda:{"binaries":{"cuobjdump":"objdump"}})})
sys.modules[comp.__name__]=comp
file=stage/"python/sglang/srt/layers/muse_spark/decode_gate_up.py"
ns=runpy.run_path(str(file));g=ns["prepare_gate_up"].__globals__
def real_method(path, name, obj=None):
    tree=ast.parse(path.read_bytes())
    if obj:tree=next(n for n in tree.body if getattr(n,"name",None)==obj)
    node=next(n for n in tree.body if getattr(n,"name",None)==name)
    node.decorator_list=[]
    return ast.unparse(node)
scope=dict(Path=Path,struct=struct,os=os)
exec(real_method(scratch/"humming/utils/cubin.py","get_cubin_kernel_names"),scope)
driver_calls=[]
scope["cbd"]=S(cuModuleLoad=lambda name: (0,name),cuModuleGetFunction=lambda mod,name:(driver_calls.append(name.decode()) or (0,67)))
exec(real_method(scratch/"humming/jit/runtime.py","load_cubin","KernelRuntime"),scope)
native.load_cubin=scope["load_cubin"]
rendered=[];configured=[];reject=None
def prepare(self):
    w=self.block_shape[0];expected=scratch/"both"/f"8192_{w}.cu"
    out=self.code+"\nauto ptr = reinterpret_cast<void*>(&"+self.kernel_expr+");\n"
    assert out==expected.read_text() and self._get_compiler() is comp.NVCCCompiler
    self.kernel_filename=str(expected.with_suffix(".cubin"));self.cubin_loaded=False
    self.load_cubin();rendered.append(w)
native.prepare=prepare
native.check_context=lambda self:self.load_cubin()
class Tensor:
    def __init__(self,shape,dtype,addr=1024): self.shape=shape;self.dtype=dtype;self.address=addr;self.device="gate_device"
    def is_contiguous(self):return True
    def data_ptr(self):return self.address
    def view(self,dtype):return Tensor(self.shape,dtype,self.address)
class Launcher:
    def prepare(self,bank,scales,maps):
        assert bank.dtype is torch.int32 and scales.dtype is torch.uint8 and maps.is_contiguous()
        return True
    def configure(self,handle,shared):
        assert handle==67;configured.append(shared);return True
launcher=Launcher()
layer=S(hidden_size=4096,intermediate_size_per_partition=4096,num_local_experts=32,w13_weight=Tensor((32,256,16384),torch.int32),
        w13_weight_scale=Tensor((32,256,8192),torch.float8_e4m3fn,10240),w13_global_scale=Tensor((32,),torch.bfloat16,20480))
meta=vendor["metas"]["w13"]; compute=vendor["compute"]; tunings=[(hi,json.dumps(c)) for _,hi,c in small["config"]]
def disassemble(args,text):
    assert text and args[:2]==["objdump","-sass"]
    fname=Path(args[2]);return (scratch/(reject or "both")/(fname.stem+".sass")).read_text()
g["version"]=lambda name:"0.1.12"
with (patch("subprocess.check_output",side_effect=disassemble),patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
      patch.object(torch.cuda,"get_device_properties",return_value=S(multi_processor_count=132))):
    state=g["prepare_gate_up"](layer,meta,compute,tunings)
    assert state and len(driver_calls)==3 and len(set(rendered))==3 and len(configured)==3
    assert all(s[1]==67 for s in state[3].values())
    second=g["prepare_gate_up"](layer,meta,compute,tunings)
    assert len(driver_calls)==3 and all(second[3][t] is k for t,k in state[3].items())
    t=tunings[0][1]
    for reject in ("original","fusion"):
        assert g["_compile"](144+int(reject=="fusion"),meta,compute,t) is None
    assert len(driver_calls)==3
    for k,v in dict(a_dtype="float8e4m3",weight_scale_2_type="channel",has_bias=True,pad_shape_n=16,use_packed_k_layout=True).items():
        assert g["_compile"](183,json.dumps(dict(json.loads(meta),**{k:v})),compute,t) is None
    g["version"]=lambda name:"0.1.11"
    assert g["prepare_gate_up"](layer,meta,compute,tunings) is None
print(json.dumps(dict(physical_cuda_driver=False,actual_wheel_stage=True,rendered_selected_widths=rendered,shared_configured=configured,
    loaded_hopper_symbols=len(driver_calls),ablation_sass_rejections=2,incompatible_flags_rejected=5,repeat_reuses_handles=True,
    adapter_sha256=hashlib.sha256(file.read_bytes()).hexdigest()),indent=2))
