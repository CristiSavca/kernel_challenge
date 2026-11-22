"""Run the frozen split's actual provider chain, replacing only packing and the CUDA fwd op."""
import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch


STAGE = Path(sys.argv[1])
BASE = STAGE / "python/sglang"
FILES = {
    "backend": BASE / "srt/layers/attention/flashattention_backend.py",
    "planner": BASE / "srt/layers/attention/direct_prompt_kv.py",
    "generic": BASE / "kernels/ops/attention/flash_attention.py",
    "version3": BASE / "kernels/ops/attention/flash_attention_v3.py",
    "provider": BASE / "kernels/aot/python/sgl_kernel/flash_attn.py",
}
NAMES = ("flash_attn_with_kvcache", "flash_attn_varlen_func")
SOURCES = {name: path.read_text() for name, path in FILES.items()}
MODULES = {name: ast.parse(src) for name, src in SOURCES.items()}
FP8 = torch.float8_e4m3fn
torch.set_num_threads(1)


def load(name, defs, namespace):
    body = [n for n in MODULES[name].body if isinstance(n, ast.FunctionDef) and n.name in defs]
    assert len(body) == len(defs)
    for n in body:
        n.decorator_list = []
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *body], type_ignores=[])
    exec(compile(module, str(FILES[name]), "exec"), namespace)
    return namespace


def select(module, node_type, predicate):
    matches = [n for n in ast.walk(module) if isinstance(n, node_type) and predicate(n)]
    assert len(matches) == 1
    return matches[0]


def block(node, omit_import=False):
    if omit_import:
        node.body = [s for s in node.body if not isinstance(s, ast.ImportFrom)]
    return compile(ast.Module(body=[node], type_ignores=[]), "<production-branch>", "exec")


split = block(select(MODULES["backend"], ast.If,
                     lambda n: ast.unparse(n.test).startswith("batched_prompt_kv is not None and query.dtype")), True)
tail_plan = block(select(MODULES["planner"], ast.If, lambda n: ast.unparse(n.test) == "packed_count < count"))
assert "from sgl_kernel.flash_attn import (" in SOURCES["backend"]
assert "self.flash_attn_varlen_func = flash_attn_varlen_func" in SOURCES["backend"]


class TorchWithNativeProbe:
    def __init__(self, op):
        self.ops = NS(sgl_kernel=NS(fwd=NS(default=op)))

    def __getattr__(self, name):
        return getattr(torch, name)


def cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)


def run(prompt_rows, tail_rows, scale_form, returning_fresh, missing_out, with_knew):
    rows = prompt_rows + tail_rows
    p, t = len(prompt_rows), len(tail_rows)
    n, cut = sum(rows), sum(prompt_rows)
    h, d, qh = 2, 64, 4
    totals = [r + 4 + (i * 17) % 59 for i, r in enumerate(rows)]
    used = torch.tensor(totals, dtype=torch.int32)
    cuq = cu(rows)
    cuk = cu(totals) if with_knew else None
    scratch = dict(count=p+t, packed_count=p, rows=rows, metadata=NS(cu_seqlens_k=cuk), cuq=cuq)
    exec(tail_plan, scratch)
    tail = scratch["tail"]
    starts = cu([x+256 for x in totals[:p]])
    layout = (starts, used[:p], int(starts[-1]), max(totals[:p]))
    table = torch.arange((p+t)*(max(totals)+3), dtype=torch.int32).view(p+t, -1)[:, :max(totals)]
    cache = [torch.empty((11, 1, h, d), dtype=FP8) for _ in range(2)]
    write = [torch.empty((n, h, d), dtype=FP8) for _ in range(2)]
    dense = [torch.empty((int(starts[-1]), h, d), dtype=FP8) for _ in range(2)]
    query = torch.empty((n, qh, d), dtype=FP8)
    arena = torch.full((n+18, qh, d), -77, dtype=torch.bfloat16)
    dest = arena[7:7+n]
    values = torch.arange((p+t)*h, dtype=torch.float32).view(-1,h) + 1
    if scale_form == "expanded":
        values = values[:1,:1].expand(p+t,h)
    scales = {} if scale_form == "none" else dict(k_descale=values, v_descale=values+2)
    calls, legacy_retries, aliases = [], [], []
    native = {}

    def check_op(*a, **kw):
        paged = a[14] is not None
        role = "tail" if paged else "head"
        calls.append(role)
        sel = slice(p,None) if paged else slice(None,p)
        token_sel = slice(cut,None) if paged else slice(None,cut)
        operand, ref_out = a[0], dest[token_sel]
        assert operand.shape == ref_out.shape and operand.data_ptr() == query[token_sel].data_ptr()
        assert operand.dtype == FP8 and operand.stride(-1) == 1
        assert a[7].tolist() == cu(rows[sel]).tolist()
        assert a[11].tolist() == totals[sel] and a[12] == max(rows[sel])
        assert a[23:29] == (0.125, True, -1, -1, 0, 3.0)
        assert (a[31] if paged else kw["num_splits"]) == 1
        assert a[20] is None and all(x is None for x in a[3:6])
        for i, key in enumerate(("k_descale", "v_descale"), 21):
            if key in scales:
                expected = scales[key][sel]
                assert a[i].data_ptr() == expected.data_ptr() and a[i].stride() == expected.stride()
                assert a[i].shape == (len(rows[sel]),h) and torch.equal(expected, a[i])
            else:
                assert a[i] is None
        if paged:
            assert a[8] is None and a[14].data_ptr() == table[p:].data_ptr()
            assert a[14].stride() == table.stride() and a[14].shape == (t,table.shape[1])
            assert all(a[i+1].data_ptr() == cache[i].data_ptr() for i in range(2))
            assert a[9] is None if cuk is None else a[9].tolist() == cu(totals[p:]).tolist()
        else:
            assert a[14] is None and a[9] is None and a[8].tolist() == starts.tolist()
            assert a[13] == max(totals[:p]) and all(a[i+1].data_ptr() == dense[i].data_ptr() for i in range(2))
        supplied = a[6]
        if supplied is not None:
            assert supplied.dtype == torch.bfloat16 and supplied.shape == ref_out.shape
            assert supplied.data_ptr() == ref_out.data_ptr()
        out = torch.empty_like(ref_out) if supplied is None or (not paged and returning_fresh) else supplied
        aliases.append(out.data_ptr() == ref_out.data_ptr())
        out.fill_(17 if paged else 39)
        return out, None

    provider = load("provider", ("maybe_contiguous", *NAMES),
                    dict(torch=TorchWithNativeProbe(check_op), is_fa3_supported=lambda: True))

    def paged_provider(*args, **kwargs):
        if missing_out and "out" in kwargs:
            legacy_retries.append(True)
            raise TypeError("unexpected keyword argument 'out'")
        return provider[NAMES[0]](*args, **kwargs)

    native[NAMES[0]] = paged_provider
    v3 = load("version3", ("_call_fa3_kernel", NAMES[0]),
              dict(torch=torch, _is_fa3_supported=lambda: True, _load_fa3_kernels=lambda: native))
    generic = load("generic", (NAMES[0],), dict(torch=torch, fa3_flash_attn_with_kvcache=v3[NAMES[0]]))

    def pack(k, v, wk, wv, pt, lengths, cuq_head, received_layout):
        assert received_layout is layout and [k,v] == cache and [wk,wv] == write
        assert pt.shape[0] == p and pt.data_ptr() == table.data_ptr() and lengths.tolist() == totals[:p]
        assert cuq_head.tolist() == cu(prompt_rows).tolist()
        return dense

    state = dict(batched_prompt_kv=(layout,tail), query=query, _fa_out=dest, page_table=table,
                 cache_seqlens=used, cu_seqlens_q=cuq, max_seqlen_q=max(rows), kwargs=dict(scales),
                 self=NS(kv_cache_dtype=FP8, fa_impl_ver=3, num_splits=1,
                         flash_attn_varlen_func=provider[NAMES[1]]),
                 layer=NS(tp_k_head_num=h, head_dim=d, scaling=.125, logit_cap=3.0), key_cache=cache[0],
                 value_cache=cache[1], write_k=write[0], write_v=write[1], pack_prompt_rows=pack,
                 flash_attn_with_kvcache=generic[NAMES[0]], causal=True, window_size=(-1,-1), result=None)
    exec(split, state)
    assert calls == ["tail", "head"] and len(legacy_retries) == int(missing_out)
    assert aliases == [not missing_out, not returning_fresh]
    assert state["result"].data_ptr() == dest.data_ptr()
    assert dest[:cut].eq(39).all() and dest[cut:].eq(17).all()
    assert arena[:7].eq(-77).all() and arena[7+n:].eq(-77).all()
    return dest.numel(), len(legacy_retries), aliases.count(False)


if __name__ == "__main__":
    cases = elements = retries = copybacks = 0
    for (prompts,tails), scales, fresh, old, knew in itertools.product(
        [([4096],[8]), ([3977,145],[8,8,8]), ([4000,203,67],[8,2])],
        ["none","expanded","ranked"], (False,True), (False,True), (False,True)
    ):
        n, r, c = run(prompts,tails,scales,fresh,old,knew)
        cases += 1
        elements += n
        retries += r
        copybacks += c
    print(json.dumps(dict(cases=cases, bf16_destination_elements=elements, provider_calls=2*cases,
          legacy_out_retries=retries, distinct_storage_copybacks=copybacks,
          tested="Production split + tail rebasing + dense direct SGL provider + generic paged FA3 wrapper + SGL provider. Mock CUDA operator writes separate sentinels. GPU latency and kernel arithmetic are outside this test.",
          source_sha256={name:hashlib.sha256(src.encode()).hexdigest() for name,src in SOURCES.items()}),indent=2))
