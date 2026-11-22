"""Run V182's query preparation, paged launcher and both device bodies on FP8 operands."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import types

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = Path("python/sglang")
PAGED = SRC / "kernels/ops/attention/paged_verify_attention.py"
PRODUCER = SRC / "kernels/ops/attention/muse_spark_qk_norm_rope_store.py"
BACKEND = SRC / "srt/layers/attention/flashattention_backend.py"


def load_function(path, name, namespace):
    node = next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def execute_case(stage):
    # Reuse the checked host pointer and BF16 rounding behavior of V62.
    helper = next(HERE.parent.glob("v62_*/cpu_check.py"))
    parsed = ast.parse(helper.read_text())
    cutoff = next(n.lineno for n in parsed.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "parser")
    support = {}
    exec(compile(ast.Module([n for n in parsed.body if n.lineno < cutoff], []), str(helper), "exec"), support)
    kernel_env = {"tl": support["tl"]}
    for name in ("_tile_geometry", "_plan", "_partition_attention", "_merge_partitions"):
        load_function(stage / PAGED, name, kernel_env)
    counts = Counter()

    class HostLaunch:
        def __init__(self, target, write_positions):
            self.kernel = kernel_env[target]
            self.write_positions = write_positions

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                converted = list(args)
                writes = {}
                for index, tensor in enumerate(args):
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    if index in self.write_positions:
                        arr = np.full(tensor.shape, np.nan, dtype=np.float32)
                        writes[index] = arr
                    elif tensor.is_floating_point():
                        arr = tensor.float().numpy()
                    else:
                        arr = tensor.numpy()
                    converted[index] = support["Pointer"](arr)
                for coords in itertools.product(*(range(n) for n in grid)):
                    support["coords"] = coords
                    self.kernel(*converted)
                for index, array in writes.items():
                    args[index].copy_(torch.from_numpy(array))
                counts[self.kernel.__name__] += 1
            return launch

    runtime = types.SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length(),
                                    cdiv=support["ceil_div"])
    ns = dict(torch=torch, math=math, triton=runtime, _partition_attention=HostLaunch("_partition_attention", (7, 8)),
              _merge_partitions=HostLaunch("_merge_partitions", (5,)))
    paged = load_function(stage / PAGED, "paged_verify_attention", ns)
    prep = load_function(stage / BACKEND, "prepare_paged_mha_query", {"torch": torch})
    encode_table = load_function(stage / PRODUCER, "bf16_e4m3_encodings", {"torch": torch})("cpu")
    layer = types.SimpleNamespace(head_dim=64, k_scale=None, v_scale=None)
    dtype = torch.float8_e4m3fn
    backend = types.SimpleNamespace(kv_cache_dtype_str="fp8_e4m3", kv_cache_dtype=dtype,
                                    kv_cache_is_mxfp8=False, fa_impl_ver=3)
    def encode(t):
        return encode_table[t.view(torch.int16).int().bitwise_and(65535).long()].view(dtype)

    # Ensure the production query preparation preserves even the two FP8 NaNs.
    every = torch.arange(256).byte().view(dtype)
    prepared = prep(backend, every, None, None, layer, logical_batch_size=1, kv_head_num=2, is_prefill=True)
    assert prepared[0] is every and torch.equal(prepared[0].view(torch.uint8), every.view(torch.uint8))
    assert all(term is None for term in prepared[1:])

    torch.set_num_threads(1)
    torch.manual_seed(182)
    peak_l2 = 0.
    for prefix, window, rope in itertools.product((5, 2050), (-1, 2047), (False, True)):
        rows, kvheads, groups, dim = 8, 2, 8, 64
        heads, length = kvheads * groups, prefix + rows
        # Physical row zero is reserved. Shuffle each slot to break contiguity.
        slots = torch.randperm(length, dtype=torch.int64) + 1
        pages = slots.int()[None, :]
        src = torch.randn(rows, heads + 2*kvheads, dim).bfloat16()
        q, k, v = src.split((heads, kvheads, kvheads), dim=1)
        def normalized(x):
            y = (x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()).bfloat16()
            if rope:
                angle = (torch.arange(rows) + prefix)[:, None, None] / (1 + torch.arange(dim//2))[None, None, :]
                a, b = y[..., ::2].float(), y[..., 1::2].float()
                y = torch.stack((a*angle.cos()-b*angle.sin(), a*angle.sin()+b*angle.cos()), -1).flatten(-2).bfloat16()
            return y
        q_bf, k_bf = normalized(q), normalized(k)
        q8 = encode(q_bf)
        q_native = q_bf.to(dtype)
        assert torch.equal(q8.view(torch.uint8), q_native.view(torch.uint8))
        stores = []
        for tail in (k_bf, v):
            prefix_rows = torch.randn(prefix, kvheads, dim).bfloat16()
            logical = torch.cat((prefix_rows, tail)).to(dtype)
            physical = torch.zeros(length+1, 1, kvheads, dim, dtype=torch.uint8)
            physical[slots[:prefix], 0] = prefix_rows.to(dtype).view(torch.uint8)
            physical[slots[prefix:], 0] = encode(tail).view(torch.uint8)
            assert torch.equal(physical[slots, 0], logical.view(torch.uint8)) and not physical[0].any()
            stores.append((physical.view(dtype), logical.float()))
        query, _, _, ks, vs = prep(backend, q8, None, None, layer, logical_batch_size=1, kv_head_num=2, is_prefill=True)
        assert query is q8 and ks is None and vs is None
        lens, cu = torch.tensor([length], dtype=torch.int32), torch.tensor([0, rows], dtype=torch.int32)
        options = dict(max_q=rows, window_left=window, softmax_scale=dim**-.5, k_descale=ks, v_descale=vs,
                       out=torch.empty(query.shape, dtype=torch.bfloat16), num_sms=132)
        out = paged(query, stores[0][0], stores[1][0], pages, lens, cu, **options)
        assert out is options["out"] and out.isfinite().all()
        ref = []
        for row in range(rows):
            stop = prefix+row+1
            start = max(0, stop-1-window) if window >= 0 else 0
            key, val = (source[start:stop].double().repeat_interleave(groups, dim=1) for _, source in stores)
            prob = torch.einsum("hd,khd->hk", q_native[row].double(), key).mul(dim**-.5).softmax(-1)
            ref.append(torch.einsum("hk,khd->hd", prob, val))
        expected = torch.stack(ref)
        relative_l2 = ((out.float()-expected).norm()/expected.norm()).item()
        assert relative_l2 < 0.004, (prefix, window, rope, relative_l2)
        peak_l2 = max(relative_l2, peak_l2)
        assert paged(query.bfloat16(), stores[0][0], stores[1][0], pages, lens, cu, **options) is None
        counts["cases"] += 1
        counts["compared_BF16_outputs"] += out.numel()
        counts["compared_new_FP8_QKV_bytes"] += q8.numel()+k_bf.numel()+v.numel()
    return dict(cpu=torch.__version__, counts=counts, all_FP8_encoded_words_preserved=256,
                max_relative_l2_vs_independent_float64=peak_l2, actual_H200=False,
                scope="Production FP8 table, query preparation, paged launcher, partition and merge. Host-generated normalized/rotated BF16 inputs and host simulation of device math.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("stage", type=Path)
    print(json.dumps(execute_case(args.parse_args().stage), indent=2))
