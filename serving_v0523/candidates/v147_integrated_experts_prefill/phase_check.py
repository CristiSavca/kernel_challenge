"""Exercise prefill and verification in one model/FA3 backend with shared FP8 storage."""
import argparse
import ast
from collections import Counter
from enum import IntEnum, auto
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

PARENTS = Path(__file__).resolve().parents[1]
WINDOW = runpy.run_path(str(PARENTS / "v146_zero_copy_window_prefill/independent_dispatch.py"))
BASE = WINDOW["BASE"]
VERIFY = runpy.run_path(str(PARENTS / "v136_fused_verify_qk_rope_cache/independent_operands.py"))


def test(stage):
    torch.set_num_threads(1)
    torch.manual_seed(147)
    enum_file = "python/sglang/srt/model_executor/forward_batch_info.py"
    source = subprocess.check_output(["git", "-C", stage, "show", BASE["PIN"] + ":" + enum_file], text=True)
    enum_env = dict(IntEnum=IntEnum, auto=auto)
    mode = WINDOW["method"](source, "ForwardMode", enum_env, ast.ClassDef)
    assert mode.TARGET_VERIFY.is_extend()
    backend, calls, Span = BASE["build"](stage, True)
    env = backend.forward_extend.__globals__
    env.update(ForwardMode=mode, KVWriteLoc=NS(for_layer=lambda fb, layer, swa_loc: fb.slots))
    span_source = (stage / BASE["SPAN"]).read_text()
    Window = WINDOW["method"](span_source, "WindowKVSpan", env, ast.ClassDef)
    model_source = (stage / "python/sglang/srt/models/muse_spark.py").read_text()
    model_env = dict(torch=torch, get_attn_backend=lambda: backend)
    exec(VERIFY["extract"](model_source, "MuseSpark1xAttention", "forward"), model_env)
    backend.use_sliding_window_kv_pool = True
    dtype = torch.float8_e4m3fn
    caches = {(i, j): torch.zeros((1120, 2, 64), dtype=dtype) for i in (0, 1) for j in (0, 1)}
    counts = Counter()

    def store(layer, slots, k, v, *_):
        for i, item in enumerate((k, v)):
            rhs = item.reshape(-1, 2, 64).to(dtype).contiguous().view(torch.uint8)
            caches[layer.layer_id, i].view(torch.uint8).index_copy_(0, slots.long(), rhs)

    def stock_write(*args):
        counts["backend_writes"] += 1
        store(*args)

    backend.token_to_kv_pool = NS(get_kv_buffer=lambda i: (caches[i, 0], caches[i, 1]), set_kv_buffer=stock_write)

    class Layer(NS):
        def __call__(self, q, k, v, fb, save_kv_cache):
            return backend.forward_extend(q, k, v, self, fb, save_kv_cache)

    for is_swa in (False, True):
        left = 255 if is_swa else -1
        layer = Layer(layer_id=int(is_swa), sliding_window_size=left, k_scale=None, v_scale=None,
                      head_dim=64, v_head_dim=64, tp_k_head_num=2, tp_v_head_num=2, tp_q_head_num=16,
                      is_cross_attention=False, attn_type="decoder", scaling=.125, logit_cap=0.)
        for current, fused_ready in [(mode.EXTEND, True), (mode.TARGET_VERIFY, True),
                                     (mode.EXTEND, True), (mode.TARGET_VERIFY, False)]:
            rows = 518 if current == mode.EXTEND else 8
            length = 841
            qkv = torch.randint(-6, 7, (rows, 1280)).to(torch.bfloat16)
            table = torch.arange(17, 17 + length, dtype=torch.int32)[None]
            swa_table = table + 29
            mapping = swa_table if is_swa else table
            slots = mapping[0, -rows:].clone()
            cuq = torch.tensor([0, rows], dtype=torch.int32)
            cuk = torch.tensor([0, length], dtype=torch.int32)
            lens = cuk[1:].clone()
            fb = NS(forward_mode=current, dcp_kv_mask=None, batch_size=1, extend_num_tokens=rows, slots=slots)
            seen = []

            def prepare_store(*_):
                seen.append("prepare")
                return slots if fused_ready else None

            backend.muse_verify_fp8_store = prepare_store

            def norm(raw, kv_store):
                q, k, v = raw.split((1024, 128, 128), dim=-1)
                if kv_store is not None:
                    store(layer, kv_store, k, v)
                    counts["fused_writes"] += 1
                    return q.to(dtype), None
                return q, k

            attn = NS(qkv_proj=lambda hidden: (hidden, None), q_size=1024, kv_size=128, use_o_gate=False,
                      _fused_qk_norm=norm, rotary_emb=None, _fused_kv_store=lambda fb: None,
                      attn=layer, _fused_o_norm_gate=lambda out, _: out, o_proj=lambda x: (x, None))
            certs = []
            for tab, visible in ((table, length), (swa_table, min(length, rows + 255))):
                start, end = int(tab[0, -visible]), int(tab[0, -1])
                desc = torch.tensor([1, 0, start, visible, visible, end], dtype=torch.int64)
                obj = Span(tab, lens, rows, None, desc, BASE["Gate"](True))
                certs.append(obj)
            visible_length = torch.tensor([min(length, rows + 255)], dtype=torch.int32)
            backend.forward_metadata = NS(max_seq_len_k=length, max_seq_len_q=rows, cache_seqlens_int32=lens,
                cu_seqlens_q=cuq, cu_seqlens_k=cuk, page_table=table, swa_page_table=swa_table,
                image_token_ranges=None, swa_spec_metadata=None, swa_out_cache_loc=slots,
                prefill_kv_span=certs[0], window_kv_span=Window(certs[1], cuq, visible_length, 255))
            before = counts["fused_writes"] + counts["backend_writes"]
            with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
                model_env["forward"](attn, None, qkv, fb)
            assert counts["fused_writes"] + counts["backend_writes"] == before + 1
            call = calls.pop()
            assert bool(seen) == current.is_target_verify()
            assert torch.equal(call["q"].contiguous().view(torch.uint8),
                               qkv[:, :1024].reshape(rows, 16, 64).to(dtype).contiguous().view(torch.uint8))
            if current == mode.EXTEND:
                assert call["page_table"] is None
                assert call["cache_seqlens"] is (visible_length if is_swa else lens)
                counts["window_dense" if is_swa else "full_dense"] += 1
            else:
                assert call["page_table"] is mapping and call["cache_seqlens"] is lens
                assert all(c.outcome is None for c in certs)
                counts["target_with_both_certificates"] += 1
            assert call["window_size"] == ((255, 0) if is_swa else (-1, -1))
            for j, original in enumerate(qkv[:, 1024:].split(128, dim=1)):
                expected = original.reshape(rows, 2, 64).to(dtype).contiguous().view(torch.uint8)
                assert torch.equal(caches[layer.layer_id, j].view(torch.uint8)[slots.long()], expected)
                counts["new_physical_kv_bytes"] += expected.numel()
    return dict(counts, pinned_forward_modes=True, production_model_and_fa_wrapper=True,
                device_math="host stand-ins for projections, norm, store and FA3")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", type=Path)
    print(json.dumps(test(p.parse_args().stage), indent=2))
