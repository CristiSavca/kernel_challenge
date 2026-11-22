"""Exercise the original complete forward_extend with CPU tensors and recording FA3 calls."""

import argparse
import ast
from collections import Counter
from enum import Enum
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import torch


PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
BACKEND = "python/sglang/srt/layers/attention/flashattention_backend.py"
PACK = "python/sglang/kernels/ops/attention/swa_prefill_kv.py"
PAGED = "python/sglang/srt/mem_cache/layout/paged_view.py"


class Mode(Enum):
    EXTEND = 1
    VERIFY = 2

    def is_target_verify(self):
        return self is Mode.VERIFY


def select(source, names):
    tree = ast.parse(source)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(funcs) == len(names)
    return "from __future__ import annotations\n" + "\n".join(ast.unparse(f) for f in funcs)


def pinned(stage, name):
    return subprocess.check_output(["git", "-C", str(stage), "show", f"{PIN}:{name}"]).decode()


def make_env(source, layout, module, records):
    def fa(kind, **kwargs):
        records.append((kind, kwargs))
        shape = kwargs["q"].shape
        result = torch.zeros(shape, dtype=torch.bfloat16)
        if kwargs.get("return_softmax_lse"):
            return result, torch.zeros((shape[1], shape[0]))
        return result

    class Launch:
        def __getitem__(self, grid):
            def record(*args, **meta):
                records.append(("pack", (grid, args, meta)))
            return record

    env = {
        "torch": torch, "ForwardMode": Mode,
        "AttentionType": NS(ENCODER_ONLY="enc", DECODER_BIDIRECTIONAL="bidi"),
        "is_cp_active": lambda _: False, "flash_attn_with_kvcache": lambda **kw: fa("paged", **kw),
        "triton": NS(cdiv=lambda n, d: (n + d - 1) // d), "_pack_swa_suffix": Launch(),
    }
    exec(select(layout, {"paged_view", "paged_kv_view"}), env)
    exec(select(module, {"pack_swa_prefill", "can_pack_swa_prefill", "with_padding_descale"}), env)
    methods = {"forward_extend", "get_paged_mha_kv_cache", "prepare_paged_mha_query", "_mxfp8_sf_kwargs"}
    exec(select(source, methods), env)
    backend = type("Replay", (), {name: env[name] for name in methods})()
    backend.flash_attn_varlen_func = lambda **kw: fa("dense", **kw)
    return backend


def run(backend, records, page, scenario):
    records.clear()
    batch, heads, dim = 2, 2, 32
    rows = 511 if scenario == "short" else 512
    length = 1097
    columns = (length + page - 1) // page
    phys_slots = columns * page
    dtype = torch.float8_e4m3fn
    def pool(token_pad):
        if scenario == "hnd":
            return torch.zeros((columns, heads, page, dim), dtype=dtype)
        buffer = torch.zeros((phys_slots, heads + token_pad, dim), dtype=dtype)
        return buffer[:, :heads]
    cache = pool(1), pool(2)
    q = torch.zeros((rows, heads, dim), dtype=torch.bfloat16)
    layer = NS(tp_q_head_num=heads, tp_k_head_num=heads, tp_v_head_num=heads,
               head_dim=dim, v_head_dim=dim, layer_id=9, is_cross_attention=False,
               attn_type="bidi" if scenario == "bidi" else "decoder",
               sliding_window_size=-1 if scenario == "full" else 2047,
               logit_cap=3.0, scaling=0.125, k_scale=torch.tensor(0.75), v_scale=torch.tensor(1.25))
    table = torch.arange(batch * columns, dtype=torch.int32).view(batch, -1) % columns
    swa = table.flip([1]) if scenario != "no_swa_map" else None
    lengths = torch.tensor([length, 777], dtype=torch.int32)
    cuq = torch.tensor([0, rows - 9, rows], dtype=torch.int32)
    metadata = NS(swa_out_cache_loc=None, image_token_ranges=None, swa_spec_metadata=None,
                  page_table=table, swa_page_table=swa, max_seq_len_q=rows,
                  cu_seqlens_q=cuq, cache_seqlens_int32=lengths, cu_seqlens_k=cuq)
    backend.__dict__.update(
        fa_impl_ver=3, _pack_swa_fp8_prefill=scenario != "disabled",
        kv_cache_dtype=dtype, kv_cache_dtype_str="auto" if scenario == "query_bf16" else "fp8_e4m3",
        kv_cache_is_mxfp8=False, page_size=page, forward_metadata=metadata,
        use_sliding_window_kv_pool=scenario != "shared", use_mla=False, fa_skip_kv_cache=False,
        is_prefill_aware_swa=scenario == "prefill_aware", local_attn_builder=None, topk=1,
        kv_index_translator=NS(is_translating=scenario == "translation"), num_splits=1,
        token_to_kv_pool=NS(get_kv_buffer=lambda _: cache, translate_loc_from_full_to_swa=lambda x: x),
    )
    fb = NS(batch_size=batch, forward_mode=Mode.VERIFY if scenario == "verify" else Mode.EXTEND,
            dcp_kv_mask=torch.tensor(1) if scenario == "dcp_mask" else None)
    out = backend.forward_extend(q, None, None, layer, fb, sinks=q if scenario == "sinks" else None,
                                 return_lse=scenario == "lse")
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (rows, heads * dim)
    name, args = records[-1]
    assert args["softmax_scale"] == layer.scaling and args["softcap"] == layer.logit_cap
    if name == "paged":
        expected = swa if backend.use_sliding_window_kv_pool and swa is not None and scenario != "full" else table
        assert torch.equal(args["page_table"], expected) and args["page_table"].shape == (batch, columns)
        assert args["k_cache"].shape == (columns, page, heads, dim) and args["ver"] == 3
        # Fallback signatures are compared with the complete pinned parent's dispatch below.
        return name, {key: ((value.shape, value.dtype, value.stride(), float(value.float().sum()))
                            if isinstance(value, torch.Tensor) else value) for key, value in args.items()}
    assert [kind for kind, _ in records] == ["pack", "dense"]
    grid, values, meta = records[0][1]
    assert meta["PAGE_SIZE"] == page and grid[1] == batch + 1
    assert values[10:14] == (*values[0].stride()[:2], *values[1].stride()[:2])
    assert torch.equal(values[2], swa if backend.use_sliding_window_kv_pool and swa is not None else table)
    assert args["seqused_k"].shape == (batch + 1,)
    assert args["max_seqlen_k"] == min(columns * page, rows + 2047)
    assert args["k_descale"].shape == args["v_descale"].shape == (batch + 1, heads)
    assert args["window_size"] == (2047, 0) and args["causal"]
    return name, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    stage = parser.parse_args().stage
    backend = (stage / BACKEND).read_text()
    kernel = (stage / PACK).read_text()
    layout = pinned(stage, PAGED)
    records, stock_records = [], []
    current = make_env(backend, layout, kernel, records)
    stock = make_env(pinned(stage, BACKEND), layout, kernel, stock_records)
    scenarios = ("long", "shared", "no_swa_map", "hnd", "full", "short", "lse", "sinks",
                 "bidi", "translation", "verify", "disabled", "prefill_aware", "dcp_mask", "query_bf16")
    counts = Counter()
    for page in (1, 7, 16, 128, 160, 192):
        for scenario in scenarios:
            name, details = run(current, records, page, scenario)
            want_dense = scenario in ("long", "shared") or (page == 1 and scenario in ("no_swa_map", "hnd"))
            assert (name == "dense") == want_dense, (page, scenario, name)
            if name == "paged":
                assert (name, details) == run(stock, stock_records, page, scenario), (page, scenario)
            counts[name] += 1
    print(json.dumps({"passed": True, "page_sizes": [1, 7, 16, 128, 160, 192], "scenarios": scenarios,
                      "calls": dict(counts), "torch": torch.__version__, "recording_fa3": True,
                      "kernel_sha256": hashlib.sha256(kernel.encode()).hexdigest(),
                      "backend_sha256": hashlib.sha256(backend.encode()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
