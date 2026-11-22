"""Replay both grouped-decode model entries with the frozen plan bodies and changing expert ownership."""
import ast
import hashlib
import json
import runpy
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "v261_fused_decode_route_staging"
HOST = runpy.run_path(str(PARENT / "independent_plan.py"))
NS = SimpleNamespace
tensor = HOST["tensor"]


def load_gate(path):
    text = (path / "patch.diff").read_text()
    name = "python/sglang/srt/layers/muse_spark/decode_gate_up.py"
    entry = text.split(f"diff --git a/{name} b/{name}\n", 1)[1].split("diff --git ", 1)[0]
    body = "".join(line[1:] for line in entry.splitlines(keepends=True) if line.startswith("+") and not line.startswith("+++ "))
    node = next(n for n in ast.parse(body).body if getattr(n, "name", "") == "run_gate_up")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), scope)
    mod = ModuleType("sglang.srt.layers.muse_spark.decode_gate_up")
    mod.run_gate_up = scope["run_gate_up"]
    return mod


class Launcher:
    def __init__(self, fault, record):
        self.fault, self.record = fault, record

    def run(self, x, y, keys, offsets, scales, maps, handle, shared, width, sms):
        direct = len(x) != len(keys)
        self.record.append("direct" if direct else "expanded")
        if self.fault == "both" or self.fault == "direct" and direct:
            return False
        first = int(offsets[0])
        if direct:
            top = len(keys) // len(x)
            span = HOST["padded"](len(keys))
            y[first:] = x[(keys[first:] % span) // top]
        else:
            y[first:] = x[first:]
        return True


def main():
    versions = [(p, HOST["source"](p), load_gate(p)) for p in (PARENT, HERE)]
    stats = Counter()
    for rows, top, width, tile in [(1,2,1024,8), (3,4,1024,8), (7,8,1024,8), (8,8,4096,8),
                                    (8,8,1024,16), (9,8,1024,8), (8,3,1024,8), (8,1,1024,8)]:
        expert_count = 32
        rng = np.random.default_rng(740 + rows * top)
        x = tensor(rng.integers(-97, 98, size=(rows,width)).astype(np.float32))
        mapping = tensor(np.full(256, -1, np.int32))
        mapping[[4,7,19,77]] = [31,1,7,0]
        for pattern in ("one", "mixed", "remote", "aliased"):
            if pattern == "one":
                ids = tensor(np.full((rows,top), 4, np.int64))
            else:
                choices = [7, 7, 19, 77, 4] if pattern == "aliased" else ([0,222,-1] if pattern == "remote" else [4,7,19,99,-1,257])
                ids = tensor(rng.choice(choices, size=(rows,top)).astype(np.int32))
            oracle, staged = HOST["expected"](ids, mapping, 256, expert_count, x)
            for fault in ("none", "direct", "both", "moved", "missing"):
                for i,(directory, source, mod) in enumerate(versions):
                    state=NS(calls=[],reverse=bool(rows % 2))
                    scope=HOST["scope"](source,state)
                    calls=[]
                    bins = (NS(shape=(2,3),data_ptr=lambda:24), NS(shape=(2,4),data_ptr=lambda:64))
                    ptrs=tuple((b.data_ptr(),b.shape) for b in bins)
                    if fault == "moved":
                        ptrs = ((int(ptrs[0][0])+16,ptrs[0][1]),ptrs[1])
                    kernel = None if fault == "missing" else (None,42,81920,tile)
                    gate = (Launcher(fault,calls), None, ptrs, {"fixed":kernel}, 132)
                    layer = NS(num_local_experts=expert_count,global_num_experts=256,intermediate_size_per_partition=width,
                               _expert_map=mapping,phases=[],humming_decode_tunings={n:[(2048,"fixed")] for n in ("w13","w2")},
                               w13_global_scale=None,w13_weight=bins[0],w13_weight_scale=bins[1],humming_decode_gate_up=gate)
                    with patch.dict(sys.modules,{mod.__name__:mod}):
                        returned, inverse, local=scope["humming_grouped_decode_outputs"](layer,x,ids)
                    direct = i and tile==8 and rows<=8 and top>1 and not top & (top-1) and kernel is not None
                    expected_calls = [] if fault in ("moved","missing") else (["direct"] if direct else []) + (
                        ["expanded"] if not direct or fault in ("direct","both") else [])
                    assert calls == expected_calls, (directory.name,pattern,fault,calls)
                    assert state.calls == (["_plan_grouped_decode"] + (
                        ["_gather_grouped_decode"] if fault in ("direct","both","moved") else []) if direct else ["_stage_grouped_decode"])
                    assert all(np.array_equal(a,b) for a,b in zip(oracle,state.metadata))
                    assert np.array_equal(inverse,oracle[3]) and np.array_equal(local,oracle[0])
                    assert np.array_equal(returned,staged,equal_nan=True)
                    stats["rank_forwards"]+=1
                    stats["checked_stage_or_output_words"]+=staged.size
                    stats["direct_success"]+=int(bool(direct and fault=="none"))
                    stats["direct_geometry_fallback"]+=int(bool(direct and fault in ("direct","both","moved")))
    print(json.dumps(dict(patch_sha256=hashlib.sha256((HERE/"patch.diff").read_bytes()).hexdigest(),cases=dict(stats),
                          scope="Ordinary grouped model path and CUDA-launch admission on CPU with native route planner/gather bodies and identity projections; separate C++ checks cover the kernel."),indent=2))


if __name__ == "__main__":
    main()
