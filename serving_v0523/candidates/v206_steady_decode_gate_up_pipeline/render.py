"""Construct the V206 short and prompt W13 and V205 fallback with the pinned Humming tuner."""
from pathlib import Path
from types import ModuleType
import json,sys,runpy,contextlib,io
here=Path(__file__).resolve().parent;scratch=Path(sys.argv[1]);stage=scratch/"sglang"
model=stage/"python/sglang/srt/layers/muse_spark"
name="sglang.srt.layers.muse_spark.decode_mma_pair"
mod=ModuleType(name);mod.__file__=str(model/"decode_mma_pair.py")
exec(compile((model/"decode_mma_pair.py").read_bytes(),mod.__file__,"exec"),mod.__dict__)
sys.modules[name]=mod
fixture=(here.parent/"v197_grouped_prompt_gate_up/render.py").read_text()
scope={"__name__":"production_gate_fixture","__file__":str(here.parent/"v197_grouped_prompt_gate_up/render.py")}
# All eligible production constructors normally select full pairing. Also force
# a checked constructor fallback so its actual regenerated source can be compared.
from unittest.mock import patch
from types import SimpleNamespace
import torch
selected=scope.copy()
src=fixture.replace('path=root/"final"', 'path=root/"full"')
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv=["render",str(scratch)]
    exec(compile(src,str(here/"render.py"),"exec"),selected)
compiled=selected["compiled"].copy()
vendor=selected["vendor"];g=selected["g"]
old=scratch/"old";old.mkdir(exist_ok=True)
def render_old(kernel):
    w=kernel.block_shape[0]
    prefix="prompt" if kernel.block_shape[2]==64 else "decode"
    path=old / f"{prefix}_{w}.cu"
    path.write_text(kernel.code+"\nauto ptr = reinterpret_cast<void*>(&"+kernel.kernel_expr+");\n")
    kernel.func=w
def deny_full(kernel): raise ValueError("offline fallback fixture")
with (patch.object(mod,"enable_full_pair",deny_full),patch.object(selected["runtime"], "prepare", render_old),
      patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
      patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(multi_processor_count=132))):
    for t in selected["small"]:assert g["_compile"](1,selected["cfg"],selected["compute"],t,False)
    assert g["_compile"](1,selected["cfg"],selected["compute"],selected["prefill"],True)
print(json.dumps(compiled,indent=2))
