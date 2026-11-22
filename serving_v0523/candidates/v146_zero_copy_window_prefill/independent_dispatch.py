"""Execute both frozen backend methods on the same request and actual FP8 physical storage."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch


PARENT = "350f90b8b4c92e4ae6dad2155a510f1763c5d27b"
BASE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                          "v143_zero_copy_prefill_spans/check_dispatch.py"))


def method(source, name, env, kind=ast.FunctionDef):
    body = BASE["node"](source, name, kind)
    exec("from __future__ import annotations\n" + ast.unparse(body), env)
    return env[name]


def tensor_rows(cache, table, b, logical):
    if table is None:
        return cache.view(torch.uint8)[b, logical]
    physical = table[b, logical].long()
    pitch = cache.shape[1]
    return cache.view(torch.uint8)[physical // pitch, physical % pitch]


def visible_interval(length, q, index, left):
    last = length - q + index
    return torch.arange(max(0, last - left) if left >= 0 else 0, last + 1)


def run(stage):
    torch.set_num_threads(1)
    torch.manual_seed(146)
    before, log0, OldSpan = BASE["build"](stage, True)
    after, log1, Span = BASE["build"](stage, True)
    old_src = subprocess.check_output(["git", "-C", stage, "show", PARENT + ":" + BASE["BACKEND"]], text=True)
    type(before).forward_extend = method(old_src, "forward_extend", before.forward_extend.__globals__)
    src = (stage / BASE["SPAN"]).read_text()
    Window = method(src, "WindowKVSpan", after.forward_extend.__globals__, ast.ClassDef)
    left = 2047
    totals = Counter()
    cases = {
        "one_long": ([517], [74160]),
        "one_first": ([699], [699]),
        "equal_visible_ragged": ([517, 800], [74160, 2564]),
        "ragged_original_k": ([513, 513, 513], [7700, 6601, 9913]),
        "unequal_visible": ([517, 800], [74160, 12111]),
    }
    good = ([517, 517], [3853, 5781])
    for name in ("pending", "padding", "old_table", "old_lens", "old_cuq", "changed_cuq",
                 "changed_lens", "changed_table", "wrong_window", "fragment", "wrong_pitch",
                 "unbacked", "capture", "compiled", "determinism", "lse", "sinks", "full", "no_window"):
        cases[name] = good
    for name, (queries, lengths) in cases.items():
        bsz = len(queries)
        views = [min(k, q + left) for k, q in zip(lengths, queries)]
        span_len = views[0]
        stride = max(views) + 43
        starts = [17 + i * stride for i in range(bsz)]
        highest = max(s + v - 1 for s, v in zip(starts, views))
        table = torch.full((bsz, max(lengths)), -1, dtype=torch.int32)
        for i, (k, v, start) in enumerate(zip(lengths, views, starts)):
            table[i, k-v:k] = torch.arange(start, start+v)
        if name == "fragment":
            table[0, lengths[0]-1] -= 8
        full_table = torch.zeros_like(table)
        for i, k in enumerate(lengths):
            full_table[i, :k] = (torch.arange(k) % max(views)) + 1
        qlen = sum(queries)
        cuq = torch.tensor([0, *torch.tensor(queries).cumsum(0).tolist()], dtype=torch.int32)
        cuk = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        lens = torch.tensor(lengths, dtype=torch.int32)
        visible = torch.tensor(views, dtype=torch.int32)
        # Old translated slots before the visible suffix are intentionally negative.
        valid = len(set(views)) == 1 and name != "fragment"
        status = torch.tensor([valid, 0, starts[0], stride, span_len, highest], dtype=torch.int64)
        gate = BASE["Gate"](name != "pending")
        cert = Span(table.clone() if name == "old_table" else table, lens.clone() if name == "old_lens" else lens,
                    qlen, None, status, gate)
        window = Window(cert, cuq.clone() if name == "old_cuq" else cuq, visible, left)
        if name == "changed_cuq":
            cuq.add_(0)
        if name == "changed_lens":
            lens.add_(0)
        if name == "changed_table":
            table.add_(0)
        size = highest + (0 if name == "unbacked" else 1)
        kv = []
        for pitch in (144, 161 if name == "wrong_pitch" else 176):
            r = torch.randint(-4, 5, (size, pitch)).to(torch.float8_e4m3fn)[:, :128].view(size, 2, 64)
            kv.append(r)
        q = torch.randint(-2, 3, (qlen + (name=="padding"), 16, 64)).to(torch.float8_e4m3fn)
        layer = NS(tp_q_head_num=16, tp_k_head_num=2, tp_v_head_num=2, head_dim=64, v_head_dim=64,
                   is_cross_attention=False, attn_type="decoder", sliding_window_size=-1 if name=="full" else
                   left-int(name=="wrong_window"), layer_id=7, k_scale=torch.tensor([.25,.5]),
                   v_scale=torch.tensor([.75,.125]), logit_cap=12., scaling=.125)
        fb=NS(forward_mode=BASE["EXTEND"], dcp_kv_mask=None, batch_size=bsz, extend_num_tokens=qlen)
        for splits in (0, 1, 4):
            for runner in (before, after):
                runner.num_splits=splits
                runner.use_sliding_window_kv_pool=True
                runner.token_to_kv_pool=NS(get_kv_buffer=lambda _: kv)
                metadata=NS(cache_seqlens_int32=lens, max_seq_len_k=max(lengths), max_seq_len_q=max(queries),
                            cu_seqlens_q=cuq, cu_seqlens_k=cuk, page_table=full_table, swa_page_table=table,
                            image_token_ranges=None, swa_spec_metadata=None, swa_out_cache_loc=None)
                if name != "no_window":
                    metadata.window_kv_span=window
                runner.forward_metadata=metadata
                with patch.object(torch.cuda, "is_current_stream_capturing", lambda:name=="capture"), \
                     patch.object(torch.compiler, "is_compiling", lambda:name=="compiled"), \
                     patch.object(torch,"are_deterministic_algorithms_enabled",lambda:name=="determinism"):
                    runner.forward_extend(q, None, None, layer, fb, return_lse=name=="lse", sinks=q if name=="sinks" else None)
            original, actual = log0.pop(), log1.pop()
            used = actual["page_table"] is None
            assert used == (name in {"one_long","one_first","equal_visible_ragged","ragged_original_k"}), name
            totals["wrapper_pairs"] += 1
            totals["dense_calls" if used else "fallback_calls"] += 1
            for k in set(original) - {"page_table","k_cache","v_cache","cache_seqlens"}:
                a,b=original[k],actual[k]
                if isinstance(a,torch.Tensor):
                    assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)),(name,k)
                else:
                    assert a==b,(name,k)
            assert original["cu_seqlens_k_new"] is actual["cu_seqlens_k_new"] is cuk
            assert actual["cache_seqlens"] is (visible if used else lens)
            if not used:
                for k in ("k_cache","v_cache"):
                    assert original[k].data_ptr()==actual[k].data_ptr()
                assert original["page_table"] is actual["page_table"]
                continue
            for i, (qcount,klen) in enumerate(zip(queries,lengths)):
                for qi in sorted({0,1,159,192,qcount//2,qcount-1}):
                    pos0=visible_interval(klen,qcount,qi,left)
                    pos1=visible_interval(views[i],qcount,qi,left)
                    assert torch.equal(pos0,pos1+klen-views[i])
                    for h in (0, 15):
                        values=[]
                        for call, positions in ((original,pos0),(actual,pos1)):
                            kr=tensor_rows(call["k_cache"],call["page_table"],i,positions)[:,h//8]
                            vr=tensor_rows(call["v_cache"],call["page_table"],i,positions)[:,h//8]
                            values.append((kr,vr))
                        assert all(torch.equal(u,w) for u,w in zip(*values))
                        totals["physical_kv_bytes"] += values[0][0].numel()+values[0][1].numel()
                        outputs=[]
                        for call,(kb,vb) in zip((original,actual),values):
                            k=kb.view(torch.float8_e4m3fn).float()
                            v=vb.view(torch.float8_e4m3fn).float()
                            s=(k @ call["q"][int(cuq[i])+qi,h].float()) * call["softmax_scale"] * call["k_descale"][i,h//8]
                            s=call["softcap"]*torch.tanh(s/call["softcap"])
                            outputs.append((s.softmax(0) @ v)*call["v_descale"][i,h//8])
                        assert torch.equal(*outputs)
                        totals["fp32_reference_outputs"]+=outputs[0].numel()
        if name=="one_long":
            events=[]
            window.visible=NS(is_cuda=True,device="mock_device",record_stream=events.append)
            for stream in (object(), object(), object()):
                with patch.object(torch.cuda,"current_stream",return_value=stream):
                    result=window.view(kv[0][:,None],kv[1][:,None],table,lens,qlen,cuq,(left,0))
                    assert result[3] is window.visible and events[-1] is stream
            totals["consumer_stream_registrations"]+=len(events)
    # Execute the pre-existing full-only acceptance suite against V146 as well.
    # It includes both successful full dense/coarse routes and the original pinned wrapper.
    inherited = BASE["test"](stage)
    return dict(**totals, fixture_cases=list(cases), inherited_full_and_fallback=inherited,
                provider="Host PyTorch plus pinned FA3 ABI; reference math, no GPU",
                parent_tree=PARENT, source_sha256=hashlib.sha256(src.encode()).hexdigest(),
                backend_sha256=hashlib.sha256((stage/BASE["BACKEND"]).read_bytes()).hexdigest())


if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("stage",type=Path)
    print(json.dumps(run(args.parse_args().stage),indent=2))
