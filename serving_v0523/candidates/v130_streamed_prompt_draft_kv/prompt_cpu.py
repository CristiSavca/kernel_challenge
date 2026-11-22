"""Exercise production prompt batching, capture guards, stacked projection and physical scatter routing on CPU."""

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch


def methods(path, name, names, scope):
    cls = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", "") == name)
    module = ast.parse("from __future__ import annotations")
    module.body.extend(n for n in cls.body if getattr(n, "name", None) in names)
    exec(compile(module, str(path), "exec"), scope)
    return {n: scope[n] for n in names}


class Norm(torch.nn.Module):
    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 0.01)).to(x.dtype)


def run(root):
    native = root / "python/sglang"
    scope = dict(torch=torch, logger=logging.getLogger("v130"))
    state = NS(direct=True, stream=True, capture=False, invariant=False, assertion=False, inside=False)
    envs = NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda: False),
              SGLANG_ENABLE_ASYNC_ASSERT=NS(get=lambda: state.assertion))
    @contextmanager
    def draft_scope(layout):
        assert not state.inside
        state.inside = True
        try:
            yield
        finally:
            state.inside = False
    scope.update(draft_tp_context=draft_scope, envs=envs, get_spec=lambda: NS(
        speculative_dflash_direct_prefix_kv=state.direct, speculative_dflash_stream_prompt_kv=state.stream),
        KVWriteLoc=lambda x, physical: x)
    Draft = type("Draft", (torch.nn.Module,), methods(native/"srt/models/dflash.py", "DFlashDraftModel",
                                                       ["project_target_hidden"], scope))
    Worker = type("Worker", (), methods(native/"srt/speculative/dflash_worker_v2.py", "DFlashWorkerV2",
        ["_prompt_kv_slab_rows", "_append_target_hidden_to_draft_kv_by_loc", "_append_target_hidden_fused"], scope))
    helper_src = native / "kernels/ops/speculative/fused_kv_materialize.py"
    ops = Counter()
    h, dim, layers = 32, 8, 5
    def calculate(proj, positions):
        k, v = proj.chunk(2, -1)
        x = k.float()
        x *= torch.rsqrt(x.square().mean(-1, keepdim=True) + .03125)
        a, b = x.chunk(2, -1)
        cos = torch.cos(positions.float()).view(-1,1,1)
        sin = torch.sin(positions.float()).view(-1,1,1)
        rotated = torch.cat((a*cos - b*sin, b*cos + a*sin), -1).to(k.dtype)
        return rotated, v
    class Pool:
        def set_kv_buffer(self, layer, loc, k, v, ks, vs):
            ops["ordinary_layer"] += 1
            self.k_data_ptrs[layer.layer_id][loc] = k.squeeze(1)
            self.v_data_ptrs[layer.layer_id][loc] = v.squeeze(1)
    class Launch:
        def __getitem__(self, grid):
            def run_kernel(proj, norm, eps, rope, positions, k, v, *args):
                loc, lens, width = args[-5:-2]
                assert width == 8 and torch.all(lens == width) and loc.numel() == proj.size(0)
                assert grid == (proj.size(0), 1, layers)
                kk, vv = calculate(proj, positions)
                for i in range(layers):
                    k[i][loc.flatten()] = kk[:, i]
                    v[i][loc.flatten()] = vv[:, i]
                ops["direct"] += 1
            return run_kernel
    def ordinary(proj, norm, eps, rope, positions, heads, d, rot, k_out, v_out):
        kk, vv = calculate(proj, positions)
        k_out.copy_(kk.transpose(0,1).unsqueeze(2))
        v_out.copy_(vv.transpose(0,1).unsqueeze(2))
        return k_out, v_out
    scope.update(_fused_norm_rope_kernel_stacked=Launch(), _fused_norm_rope_stacked=ordinary,
                 triton=NS(next_power_of_2=lambda x: 1 << (x-1).bit_length()))
    Helper = type("Helper", (), methods(helper_src, "FusedKVMaterializeHelper",
                                        ["materialize", "full_prompt_cache", "can_stream_prompt", "_ensure_workspace"], scope))
    modules = {
        "sglang.srt.layers.layernorm": NS(RMSNorm=Norm),
        "sglang.srt.models.dflash": NS(DFlashDraftModel=Draft),
        "sglang.srt.batch_invariant_ops": NS(is_batch_invariant_mode_enabled=lambda:state.invariant),
        "sglang.srt.mem_cache.memory_pool": NS(MHATokenToKVPool=Pool),
        "sglang.srt.environ": NS(envs=envs),
    }
    generator = torch.Generator().manual_seed(308)
    d = Draft()
    d.quantizable = False
    d.fc = torch.nn.Linear(h*2, h, bias=False, dtype=torch.bfloat16)
    d.fc.weight.data.copy_(torch.randint(-4, 5, d.fc.weight.shape, generator=generator).float()/16)
    d.hidden_norm = Norm()
    d.layers = [NS(self_attn=NS(attn=NS(layer_id=i,k_scale=1.,v_scale=1.))) for i in range(layers)]
    cache = Pool()
    cache.dtype,cache.is_quantized_kv_cache = torch.bfloat16,False
    helper = Helper()
    helper.n_layers,helper.num_kv_heads,helper.head_dim,helper.rotary_dim = layers,1,dim,dim
    helper.kv_size,helper.layer_out_dim,helper.device = dim,2*dim,torch.device("cpu")
    weight = torch.randint(-2,3,(h,layers*2*dim),generator=generator).float().div_(8).bfloat16()
    helper.flat_kv_weight_t = weight
    helper.k_norm_weights,helper.eps_values = torch.ones(layers,dim),torch.ones(layers)
    helper.max_position_hint = 100000
    helper._ensure_rope_cache = lambda pos: torch.ones(1,dim)
    w = Worker()
    w.draft_model,w.draft_model_runner,w.model_runner = d,NS(token_to_kv_pool=cache),NS(device=helper.device)
    w._fused_kv_helper,w._use_fused_kv_materialize,w.lilicorr = helper,True,None
    w._logged_padding_trim,w.draft_parallel_layout = True,None
    guard_cases = {}
    all_cases = bytes_checked = 0
    with patch.dict(sys.modules, modules), patch.object(torch.cuda, "is_current_stream_capturing", side_effect=lambda:state.capture):
        cache.k_data_ptrs, cache.v_data_ptrs = [], []
        helper._prefix_cache = (cache,cache.k_data_ptrs,cache.v_data_ptrs,())
        meta = NS(device=d.fc.weight.device,dtype=torch.bfloat16,shape=(9999,2*h),ndim=2,is_cuda=True,element_size=lambda:2)
        def guard(name, expected):
            rows = w._prompt_kv_slab_rows(meta)
            assert bool(rows) == expected, (name,rows)
            guard_cases[name] = rows
        guard("ordinary",True)
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            guard("compiling",False)
        state.assertion=True; guard("async_assert",False); state.assertion=False
        cache.dtype=torch.float8_e4m3fn; guard("fp8_pool",False); cache.dtype=torch.bfloat16
        old=cache.k_data_ptrs;cache.k_data_ptrs=[]; guard("replaced_pointer_table",False);cache.k_data_ptrs=old
        cache._store_kv_layer=lambda *a:None; guard("custom_pool_writer",False);del cache._store_kv_layer
        state.capture=True; guard("capture",False); state.capture=False
        state.invariant=True; guard("batch_invariant",False); state.invariant=False
        state.stream=False; guard("disabled",False); state.stream=True
        w.lilicorr=object(); guard("lilicorr",False); w.lilicorr=None
        for name, module in (("fc",d.fc),("norm",d.hidden_norm)):
            handle=module.register_forward_hook(lambda *a: None)
            guard(name+"_hook",False)
            handle.remove()
        d.project_target_hidden=lambda x: x
        guard("projector_override",False)
        del d.project_target_hidden
        # Scale the production 64 MiB budget to 32 rows for the end-to-end shape tests.
        w._prompt_kv_slab_rows=lambda x: 32 if state.stream else 0
        for n in (1,7,8,31,32,33,39,40,63,64,65,71,72,93,128,149):
            for asserted, padding, strided in ((False,0,False),(False,3,False),(True,4,False),(False,4,True)):
                state.assertion=asserted
                x=torch.randint(-4,5,(n+padding,h*2),generator=generator).bfloat16().div_(16)
                all_locs=torch.randperm(3*n+11,generator=generator)
                loc=all_locs[:2*n:2] if strided else all_locs[:n]
                pos=torch.randint(1,9999,(n,),generator=generator)
                expected = None
                for stream in (False,True):
                    state.stream=stream
                    cache.k_data_ptrs=[torch.full((len(all_locs),dim),-399.,dtype=torch.bfloat16) for _ in range(layers)]
                    cache.v_data_ptrs=[torch.full_like(t,-393.) for t in cache.k_data_ptrs]
                    helper._prefix_cache=(cache,cache.k_data_ptrs,cache.v_data_ptrs,())
                    helper._mm_out_supported=True
                    helper._workspace_capacity=0;helper._workspace_dtype=None
                    helper._k_workspace=helper._v_workspace=helper._proj_workspace=helper._full_prompt_lengths=None
                    previous=ops["direct"]
                    w._append_target_hidden_to_draft_kv_by_loc(target_hidden=x,cache_loc=loc,positions=pos,
                                                              extend_lens=torch.tensor([n//2,n-n//2]))
                    assert bool(ops["direct"]-previous) == (stream and n>=32 and not asserted and not strided)
                    result=torch.stack(cache.k_data_ptrs+cache.v_data_ptrs)
                    if expected is None:
                        expected=result
                    else:
                        assert torch.equal(result.view(torch.int16),expected.view(torch.int16))
                        assert n<32 or helper._workspace_capacity<=32
                        bytes_checked += result.numel()*result.element_size()
                all_cases += 1
    return dict(torch=str(torch.__version__),physical_cuda=False,guard_cases=guard_cases,
                paired_appends=all_cases,compute_and_untouched_pool_bytes_checked=bytes_checked,
                calls=dict(ops),gpu_kernel="norm/RoPE arithmetic and pointer stores represented by a CPU adapter")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    torch.set_num_threads(1)
    print(json.dumps(run(parser.parse_args().root),indent=2))
