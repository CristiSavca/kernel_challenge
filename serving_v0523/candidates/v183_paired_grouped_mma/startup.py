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
scratch=Path(sys.argv[1])
stage=scratch/"sglang"
sys.argv=["codegen",str(scratch/"humming"),str(stage),str(scratch/"build")]
with contextlib.redirect_stdout(io.StringIO()): vendor=runpy.run_path(str(here.parent/"v169_grouped_expert_tile_cohorts/vendor_codegen.py"))
paired=runpy.run_path(str(stage/"python/sglang/srt/layers/muse_spark/prefill_mma_pair.py"))
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
    p=scratch/"final"/f"4096_{w}.cu"
    if full != p.read_text():
        import difflib
        raise SystemExit(''.join(list(difflib.unified_diff(p.read_text().splitlines(True),full.splitlines(True)))[:25]))
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
c=vendor["configs"]["w2"]; meta=vendor["metas"]["w2"]; compute=vendor["compute"]
widths={w:dict(w2=json.dumps(dict(json.loads(c),block_shape=[w,256,64],warp_shape=[w,32,64]))) for w in (136,152,176)}
layer=S(humming_prefill_tiles=(widths,(32,16),132))
with (patch("subprocess.check_output",side_effect=disassemble),
      patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
      patch.object(torch.cuda,"get_device_properties",return_value=S(multi_processor_count=132)),
      patch("importlib.metadata.version",return_value="0.1.12")):
    # version() is imported by value in production.
    paired["prepare_down"].__globals__["version"]=lambda name:"0.1.12"
    plan=paired["prepare_down"](layer,meta,compute,c)
    assert all(value.dtype==torch.int64 and value.device.type=="cpu" and value.tolist()[0:2]==[0,1<<30]
               and 41<=value[2].item()<=44 and value[3].item()==0 for value in plan.values())
    assert set(rendered)=={136,152,176} and len(registered)==len(plan)
    same=paired["prepare_down"](layer,meta,compute,c)
    assert all(same[k] is v for k,v in plan.items())
    bad_image=True
    assert paired["_prepare_down"](23,meta,compute,c) is None
    for k,v in dict(a_dtype="float8e4m3",has_bias=True,weight_scale_2_type="channel",pad_shape_n=16,use_packed_k_layout=True).items():
        assert paired["_prepare_down"](23,json.dumps(dict(json.loads(meta),**{k:v})),compute,c) is None
print(json.dumps(dict(rendered_widths=rendered,registration_calls=registered,native_config_handles=[x.tolist() for x in plan.values()],
    missing_hgmma_rejected=True,incompatible_configs_declined=5,repeat_reuses_config_objects=True,physical_cuda_driver=False,
    adapter_sha256=hashlib.sha256((stage/"python/sglang/srt/layers/muse_spark/prefill_mma_pair.py").read_bytes()).hexdigest()),indent=2))
