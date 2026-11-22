"""Exercise the submitted wave planner, packer, Humming adapter and finalizer on CPU."""

import ast
import itertools
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

DIR = Path(__file__).resolve().parent
FX = runpy.run_path(str(DIR.parent / "v70_dflash_selector_tree/cpu_check.py"))
array, Pointer, program = (FX[n] for n in ("array", "Pointer", "program"))
cdiv = lambda a, b: (a + b - 1) // b


def extracted(path, names, globals_):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if getattr(n, "name", None) in names]
    for node in tree.body:
        node.decorator_list = []
        node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
    tree.body.insert(0, ast.ImportFrom("__future__", [ast.alias("annotations")], 0))
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), globals_)


class Ptr(Pointer):
    def __init__(self, t):
        a = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
        super().__init__(a)
        self.dtype = SimpleNamespace(element_ty="bfloat16" if t.dtype == torch.bfloat16 else a.dtype)


class Launch:
    def __init__(self, fn, pid, out=()):
        self.fn, self.pid, self.out = fn, pid, out

    def __getitem__(self, grid):
        def execute(*args, **kw):
            args_ = [Ptr(t) if isinstance(t, torch.Tensor) else t for t in args]
            kw.pop("num_warps", None)
            kw.pop("num_stages", None)
            self.fn.__globals__["tl"].num_programs = lambda _: grid[0]
            for x in range(grid[0]):
                self.pid[0] = array(x)
                self.fn(*args_, **kw)
            for i in self.out:
                args[i].copy_(torch.from_numpy(args_[i].data))
        return execute


def make_ops(checkout):
    root = checkout / "python/sglang"
    wf = root / "srt/layers/muse_spark/humming_prefill_waves.py"
    ff = root / "kernels/ops/moe/muse_spark_post_expert_finalize_triton.py"
    pack_pid, (pack,) = program(wf, ("_pack_wave",))
    pid, (row, wave, ordinary) = program(ff, (
        "_post_expert_norm_weighted_sum_row", "_finalize_wave", "_post_expert_norm_weighted_sum_kernel",
    ))
    tl = row.__globals__["tl"]
    tl.static_range = range
    tl.rsqrt = lambda a: array(np.float32(1) / np.sqrt(a))
    finals = dict(torch=torch, triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
                  _MIN_TOKENS_FOR_FUSED=32, _finalize_wave=Launch(wave, pid, (3,)),
                  _post_expert_norm_weighted_sum_kernel=Launch(ordinary, pid, (3,)))
    extracted(ff, {"fused_post_expert_norm_weighted_sum"}, finals)
    gemms = []
    class Humming:
        @staticmethod
        def forward_layer(layer, *, inputs, outputs, expert_layout, compute_config, tuning_config, sublayer_name):
            assert not torch._C._is_alias_of(inputs, outputs)
            assert compute_config == "grouped" and tuning_config == sublayer_name
            bounds = expert_layout.tolist()
            assert bounds == sorted(bounds) and bounds[-1] == len(inputs)
            outputs.fill_(float("nan"))
            for e, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
                if lo < hi:
                    outputs[lo:hi] = inputs[lo:hi].float() @ getattr(layer, sublayer_name)[e].T
            gemms.append((sublayer_name, len(inputs) - bounds[0]))
    class Silu:
        def run(self, raw, activation, keys, offsets, scales, span):
            assert not torch._C._is_alias_of(raw, activation)
            lo = int(offsets[0])
            expert = keys[lo:] // span - 1
            scaled = (raw[lo:].float() * scales[expert, None].float()).bfloat16().float()
            gate, up = scaled.chunk(2, dim=1)
            activation[lo:] = torch.nn.functional.silu(gate) * up
    ops = dict(torch=torch, triton=SimpleNamespace(cdiv=cdiv,next_power_of_2=finals["triton"].next_power_of_2),
               _pack_wave=Launch(pack, pack_pid, (4,)), HummingMethod=Humming, _GROUPED_COMPUTE="grouped",
               _expert_scaled_silu_module=Silu, fused_post_expert_norm_weighted_sum=finals["fused_post_expert_norm_weighted_sum"])
    extracted(wf, {"wave_capacity","plan_waves","grouped_prefill_waves"}, ops)
    return ops, finals, gemms


def reference(layer, hidden, scores, ids):
    tokens, width = hidden.shape
    # Populate the full slot layout first. This oracle has no waves, sorting or compact map.
    values = torch.zeros((*ids.shape,width), dtype=torch.bfloat16)
    for e in range(layer.num_local_experts):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        raw = (hidden[tok].float() @ layer.w13[e].T).bfloat16()
        gate, up = (raw * layer.w13_global_scale[e]).float().chunk(2, dim=1)
        a = (torch.nn.functional.silu(gate) * up).bfloat16()
        down = (a.float() @ layer.w2[e].T).bfloat16()
        values[tok,slot] = down * layer.w2_global_scale[e]
    w = layer._post_expert_norm_weight.float()
    y = values.float()
    if layer._post_expert_norm_weight_first:
        y = y * w
    y = y * torch.rsqrt((y * y).sum(-1,keepdim=True) / width + layer._post_expert_norm_eps)
    if not layer._post_expert_norm_weight_first:
        y = y * w
    acc = torch.zeros((tokens,width))
    for k in range(ids.shape[1]):
        acc += y[:,k] * scores[:,k,None]
    return acc.bfloat16()


def main(checkout):
    ops, finals, gemms = make_ops(checkout)
    rng = torch.Generator().manual_seed(982026)
    def weights(s): return torch.randint(-4,5,s,generator=rng).float() / 8
    cases = elements = empty_waves = 0
    for t,k,e,c,h,i in [(17,4,3,8,16,8),(23,3,4,5,8,16),(9,2,2,19,8,8),(18,8,4,9,16,16),(1,1,1,1,8,8)]:
        for layout,first in itertools.product(("balanced","remote","local","mixed","edge"),(True,False)):
            ids = torch.randint(-2,e+1,(t,k),generator=rng,dtype=torch.int32)
            if layout == "remote": ids.fill_(-1)
            if layout == "local": ids.fill_(e-1)
            if layout == "balanced": ids = torch.arange(t*k,dtype=torch.int32).view(t,k) % e
            if layout == "edge": ids.flatten()[:min(c-1,t*k)] = 0
            hidden = weights((t,h)).bfloat16()
            scores = weights((t,k))
            layer = SimpleNamespace(num_local_experts=e,intermediate_size_per_partition=i,
                w13=weights((e,2*i,h)),w2=weights((e,h,i)),
                w13_global_scale=torch.linspace(.5,1.25,e).bfloat16(),
                w2_global_scale=torch.linspace(.625,1.5,e).bfloat16(),
                _post_expert_norm_eps=1e-5,_post_expert_norm_weight=weights((h,)).bfloat16(),
                _post_expert_norm_weight_first=first,humming_grouped_tunings={"w13":"w13","w2":"w2"})
            gemms.clear()
            actual = ops["grouped_prefill_waves"](layer,hidden,scores,ids,c)
            expected = reference(layer,hidden,scores,ids)
            assert torch.equal(actual, expected), (layout,c,t,k,torch.max((actual-expected).abs()))
            assert all(gemms[n][0]=="w13" and gemms[n+1]==("w2",gemms[n][1]) for n in range(0,len(gemms),2))
            assert sum(count for name,count in gemms if name=="w13") == ((ids>=0)&(ids<e)).sum().item()
            empty_waves += sum(count==0 for name,count in gemms if name=="w13")
            cases += 1
            elements += actual.numel()
    cap = ops["wave_capacity"]
    allocations={}
    for t in (1024,8192,16384,32768,131072):
        c=cap(t,8,32,4096,4096)
        allocations[t] = {"capacity":c,"waves":cdiv(t*8,c) if c else 0,
                          "arena_bytes":min(t*8,c+7)*24576 if c else 0}
        assert allocations[t]["arena_bytes"] <= 384 << 20
    assert cap(0,8,32,4096,4096)==0 and cap(1000,16,32,4096,4096)==0
    print(json.dumps(dict(integrated_cases=cases,bf16_outputs=elements,empty_waves=empty_waves,
                          workspace_geometry=allocations,real_gpu=False), indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
