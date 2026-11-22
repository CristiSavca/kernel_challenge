"""Execute the actual indexed Humming front door up to dispatch or fallback."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

stage = Path(sys.argv[1])
module_path = stage / "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
source = ast.parse(module_path.read_text())
function = next(f for f in source.body if getattr(f, "name", "") == "humming_expert_outputs")
fallback_at = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "flat_ids")
function.body = [n for n in function.body[:fallback_at] if not isinstance(n, (ast.Import, ast.ImportFrom))]
function.body.append(ast.parse("return FALLBACK").body[0])
program = ast.parse("from __future__ import annotations")
program.body.append(function)

class InspectCall:
    calls = 0

    def run(self, hidden, ids, gate, gate_s, gate_g, down, down_s, down_g, active, out, plan, groups):
        self.calls += 1
        m, h = hidden.shape
        slots = ids.numel()
        used = min(32, slots)
        assert ids.is_contiguous() and hidden.is_contiguous()
        assert active.shape == (slots,128) and out.shape == (slots,h)
        assert active.dtype == out.dtype == hidden.dtype == torch.bfloat16
        assert plan.dtype == ids.dtype == torch.int32
        assert plan.shape == (1+9*(used+(slots-used)//8),)
        assert groups == max(1,min(used,(slots*32+255)//256))
        assert gate_s.dtype == down_s.dtype == torch.uint8
        assert active.data_ptr()!=out.data_ptr()
        out.fill_(7)

probe = InspectCall()
fallback = object()
env = dict(torch=torch, _route_pair_module=lambda: probe, FALLBACK=fallback)
exec(compile(ast.fix_missing_locations(program),str(module_path),"exec"),env)
dispatch = env[function.name]
layer = SimpleNamespace(humming_route_pair=True,num_local_experts=32,global_num_experts=256,
                        w13_weight=object(),w13_weight_scale=torch.empty(2,dtype=torch.float8_e4m3fn),
                        w13_global_scale=object(),w2_weight=torch.empty((32,8,1),dtype=torch.int32),
                        w2_weight_scale=torch.empty((32,1,128),dtype=torch.float8_e4m3fn),w2_global_scale=object())
tests = 0

def run(m,k,mode="native"):
    global tests
    hidden = torch.empty((m,128),dtype=torch.bfloat16)
    ids = torch.zeros((m,k),dtype=torch.int32)
    if mode=="hidden_stride": hidden = torch.empty((m,256),dtype=torch.bfloat16)[:,::2]
    if mode=="id_stride": ids = torch.empty((m,k*2),dtype=torch.int32)[:,::2]
    if mode=="wrong_id_dtype": ids=ids.long()
    if mode=="align": hidden = torch.empty(m*128+1,dtype=torch.bfloat16)[1:].view(m,128)
    if mode=="wrong_h": hidden = torch.empty((m,64),dtype=torch.bfloat16)
    layer.humming_route_pair = mode!="jit_fallback"
    original_compile = torch.compiler.is_compiling
    try:
        if mode=="compile": torch.compiler.is_compiling=lambda: True
        old_calls=probe.calls
        result = dispatch(layer,hidden,ids)
    finally:
        torch.compiler.is_compiling=original_compile
    expected = mode=="native" and m>0 and m*k<=256
    if m==0:
        assert result[0].numel()==0 and result[0].shape==(0,k,128) and result[1] is ids
    elif expected:
        assert result[0].shape==(m,k,128) and result[1] is ids and torch.all(result[0]==7)
    else:
        assert result is fallback, (m,k,mode)
    assert probe.calls-old_calls==int(expected)
    tests+=1

for m,k in ((0,8),(1,1),(1,8),(8,8),(32,8),(33,8),(64,8),(128,8),(256,8),(31,7),(36,7),(37,7)):
    run(m,k)
for m,k in ((8,8),(32,8)):
    for mode in ("hidden_stride","id_stride","wrong_id_dtype","align","wrong_h","jit_fallback","compile"):
        run(m,k,mode)
print(json.dumps(dict(cases=tests,fastpath_calls=probe.calls,
                      adapter_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
                      torch=str(torch.__version__)),indent=2))
