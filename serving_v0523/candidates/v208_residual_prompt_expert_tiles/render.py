"""Render production W13 and W2, then force each optional image to fail in sequence."""
import contextlib
import io
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

here = Path(__file__).resolve().parent
root = Path(sys.argv[1]).resolve()
model = root / "sglang/python/sglang/srt/layers/muse_spark"
out = root / "images"
out.mkdir(exist_ok=True)
for name in ("residual_prompt_mma", "decode_mma_pair"):
    full = "sglang.srt.layers.muse_spark." + name
    mod = ModuleType(full)
    mod.__file__ = str(model / (name + ".py"))
    exec(compile(Path(mod.__file__).read_text(), mod.__file__, "exec"), mod.__dict__)
    sys.modules[full] = mod
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(here.parent / "v197_grouped_prompt_gate_up/render.py"))
w13, vendor, runtime = g["g"]["_compile"], g["vendor"], g["runtime"]
w2 = runpy.run_path(str(model / "prompt_weight_pipeline.py"))["_prepare"]
meta, compute = vendor["metas"], vendor["compute"]
denied, attempts, success = set(), [], {}

def emit(kernel):
    name = type(kernel).__name__
    attempts.append(name)
    if name in denied:
        raise RuntimeError("offline compiler failure")
    src = kernel.code + "\nauto ptr = reinterpret_cast<void*>(&" + kernel.kernel_expr + ");\n"
    dst = out / (name + "_" + str(kernel.block_shape[0]) + ".cu")
    dst.write_text(src)
    kernel.func = 99
    kernel.kernel_id = 37

tuning = vendor["configs"]
with (patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
      patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=132)),
      patch.object(runtime, "prepare", emit)):
    for phase, call in (("W13", lambda ctx: w13(ctx, meta["w13"], compute, g["prefill"], True)),
                        ("W2", lambda ctx: w2(ctx, meta["w2"], compute, tuning["w2"]))):
        denied.clear()
        trials = []
        for ctx in range(2, 5):
            attempts.clear()
            result = call(ctx)
            trials.append(dict(attempts=list(attempts), accepted=result is not None))
            if result is not None:
                denied.add(attempts[-1])
        assert [s["accepted"] for s in trials] == [True, True, False]
        assert [len(s["attempts"]) for s in trials] == [1, 2, 2]
        success[phase] = trials
    denied.clear()
    for cfg in g["small"]:
        attempts.clear()
        assert w13(8, meta["w13"], compute, cfg, False)
        assert attempts == ["GroupedDecodeFullPairGateUpKernel"]
print(json.dumps(success, indent=2))
