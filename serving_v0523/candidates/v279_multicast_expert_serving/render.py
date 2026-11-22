"""Render both production prompt kernels in the combined checkout."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType

here=Path(__file__).resolve().parent
vendor,stage,out=map(Path,sys.argv[1:])
# Load the common barrier helper for both independent vendor constructor fixtures.
helper=stage/"python/sglang/srt/layers/muse_spark/prompt_multicast.py"
name="sglang.srt.layers.muse_spark.prompt_multicast"
mod=ModuleType(name)
mod.__file__=str(helper)
exec(compile(helper.read_text(),str(helper),"exec"),mod.__dict__)
sys.modules[name]=mod
results={}
for name,folder in (("gate","v277_multicast_prompt_gate_up"),("down","v274_four_block_prompt_multicast")):
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        result=runpy.run_path(str(here.parent/folder/"render.py"))
    results[name]=json.loads(stream.getvalue())
    if name=="gate":
        from unittest.mock import patch
        from types import SimpleNamespace
        import torch

        observed={}
        with (patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),patch.object(torch.cuda,"current_device",return_value=0),
              patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(multi_processor_count=132))):
            for i,grid in enumerate((0,4,116,120,124,128,132)):
                result["launcher"].prompt_cluster_grid=lambda *_,grid=grid:grid
                value=result["compile_kernel"](i+50,*result["args"],True,True)
                assert (value is not None)==(grid in (128,132))
                observed[grid]=value[-1] if value else None
        results["W13_occupancy_response"]=observed
print(json.dumps(results,indent=2))
