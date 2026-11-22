"""Render both submitted decode schedules on the real Humming small-shape tuner."""
from pathlib import Path
import contextlib, io, json, runpy, sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import torch
here=Path(__file__).resolve().parent;scratch=Path(sys.argv[1]);stage=scratch/"sglang"
sys.argv=[sys.argv[0],str(scratch)]
with contextlib.redirect_stdout(io.StringIO()):
    donor=runpy.run_path(str(here.parent/"v189_decode_grouped_mma/small_render.py"))
vendor=donor["driver"];conf=donor["config"]
jit=ModuleType("sglang.kernels.jit");jit.__file__=str(stage/"python/sglang/kernels/jit/__init__.py")
pkg=ModuleType("sglang.kernels");pkg.jit=jit;sys.modules[pkg.__name__]=pkg
module=runpy.run_path(str(stage/"python/sglang/srt/layers/muse_spark/decode_mma_pair.py"))
def render(self):
    module["append_pair"](self, both)
    path=scratch/("paired_twice" if both else "paired")/f"4096_{self.block_shape[0]}.cu"
    path.parent.mkdir(exist_ok=True)
    path.write_text(self.code+"\nauto ptr = reinterpret_cast<void*>(&"+self.kernel_expr+");\n")
for both in (False, True):
    with (patch.object(vendor["OfflineRuntime"],"prepare",render),
          patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),
          patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(multi_processor_count=132))):
        for _,hi,cfg in conf:
            vendor["HummingKernel"](**(json.loads(vendor["metas"]["w2"])|json.loads(vendor["compute"])|cfg))
print(json.dumps(donor["rec"],indent=2))
