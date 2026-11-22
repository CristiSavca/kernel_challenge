"""Exercise the extra projected-prompt recovery writer in the V229/V248 lineage against V253 ownership."""
from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
from types import MethodType, SimpleNamespace as NS
import torch

here = Path(__file__).resolve().parent
service = here.parents[1]
base = service / "sglang"
pin = "297e54fd105e974cdf401e5b4d20253a22c16f90"
worker_path = "python/sglang/srt/speculative/dflash_worker_v2.py"


def pinned(path):
    return subprocess.check_output(["git", "-C", base, "show", f"{pin}:{path}"]).decode()


def source(version, path):
    """Reconstruct the one requested file from the verified official export in memory."""
    cand, = here.parent.glob(version + "_*")
    patch = (cand / "patch.diff").read_bytes()
    digest = json.loads((cand / "validation.json").read_bytes())
    assert sha256(patch).hexdigest() == (digest.get("patch_sha256") or digest["sha256"])
    rows = patch.decode().splitlines(keepends=True)
    mark = f"diff --git a/{path} b/{path}\n"
    raw = pinned(path).splitlines(keepends=True)
    if mark not in rows:
        return "".join(raw)
    start = rows.index(mark) + 1
    end = next((i for i in range(start, len(rows)) if rows[i].startswith("diff --git ")), len(rows))
    out = []
    pos = 0
    for row in rows[start:end]:
        if row.startswith("@@"):
            offset = int(re.match(r"@@ -(\d+)", row)[1]) - 1
            out.extend(raw[pos:offset])
            pos = offset
        elif row.startswith(("index ", "---", "+++")):
            continue
        elif row.startswith(("-", " ")):
            assert raw[pos] == row[1:]
            if row[0] == " ":
                out.append(raw[pos])
            pos += 1
        elif row.startswith("+"):
            out.append(row[1:])
        else:
            raise AssertionError(row)
    return "".join(out + raw[pos:])


def symbol(text, name, scope, container=None):
    nodes = ast.parse(text).body
    if container:
        nodes = next(n.body for n in nodes if getattr(n, "name", None) == container)
    node = next(n for n in nodes if getattr(n, "name", None) == name)
    exec("from __future__ import annotations\n" + ast.unparse(node), scope)
    return scope[name]


def collect_sidecar():
    path = here.parent / "v249_private_draft_swa/replay.py"
    spec = importlib.util.spec_from_file_location("v249_owner", path)
    owner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = owner
    spec.loader.exec_module(owner)
    owner.source = lambda _, p: source("v253", "python/sglang/srt/" + p)
    build = owner.method
    captured = {}

    def instrument(stage, path, cls, name, scope):
        func = build(stage, path, cls, name, scope)
        if name == "_draft_write_ids":
            def plan(worker, plan):
                captured.update(worker=worker, plan=plan, kind=scope["IdSpaceKind"])
                return func(worker, plan)
            return plan
        if name == "build_swa_draft_pools":
            def sidecar(*args, **kw):
                spec, entries = func(*args, **kw)
                captured.update(sidecar=entries[0], host_domain=spec[0].indices_from_pool)
                return spec, entries
            return sidecar
        return func

    owner.method = instrument
    prior = owner.run(None)
    assert prior["native_slot_plans"] and captured["host_domain"] == "swa"
    w = captured["worker"]
    assert captured["sidecar"].device_pool is w._context_kv_pool()
    return captured


def run(joined_stage=None):
    persisted = collect_sidecar()
    w, plan = persisted["worker"], persisted["plan"]
    outer = w.draft_model_runner.token_to_kv_pool
    leaf = persisted["sidecar"].device_pool
    text = source("v248", worker_path)
    previous = source("v229", worker_path)
    joined = (Path(joined_stage)/worker_path).read_text() if joined_stage is not None else None
    scope = dict(torch=torch, dataclass=dataclass, draft_tp_context=lambda _:contextlib.nullcontext(),
                 logger=logging.getLogger("ownership.replay"), _is_npu=False)
    mem = pinned("python/sglang/srt/mem_cache/memory_pool.py")
    symbol(mem, "KVWriteLoc", scope)
    symbol(mem, "unwrap_write_loc", scope)
    scope["MLATokenToKVPool"] = type("MLA", (), {})
    type(outer).set_kv_buffer = symbol(pinned("python/sglang/srt/mem_cache/swa_memory_pool.py"),
                                        "set_kv_buffer", scope, "SWAKVPool")
    def bind(source, name):
        return MethodType(symbol(source, name, scope, "DFlashWorkerV2"), w)

    # The three V249 bindings are preserved, including the fused writer callback's owner.
    fused_old = "        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool\n        if self._fused_kv_helper is None:"
    assert text.count(fused_old)==1
    combined_fused = joined or text.replace(fused_old, fused_old.replace(
        "self.draft_model_runner.token_to_kv_pool", "self._context_kv_pool()"))
    w._append_target_hidden_fused = bind(combined_fused, "_append_target_hidden_fused")
    w._append_target_hidden_to_draft_kv_by_loc = bind(joined or text, "_append_target_hidden_to_draft_kv_by_loc")
    if joined is not None:
        scope["IdSpaceKind"]=persisted["kind"]
        w._draft_write_ids=bind(joined,"_draft_write_ids")
        w._context_kv_pool=bind(joined,"_context_kv_pool")
        assert w._context_kv_pool() is leaf
    name = "_append_paired_prompt_kv"
    pattern = "            pool = self.draft_model_runner.token_to_kv_pool\n            if cache_loc_2d is None:"
    assert text.count(pattern) == previous.count(pattern) == 1
    assert ast.dump(ast.parse(text[text.index("    def "+name):text.index("    def _append_target_hidden_sequential")].lstrip())) == ast.dump(ast.parse(previous[previous.index("    def "+name):previous.index("    def _append_target_hidden_sequential")].lstrip()))
    w.lilicorr = None
    for layer in w.draft_model.layers:
        attn = layer.self_attn
        attn.kv_size = attn.head_dim
        attn.v_scale = 1
    allocmap = outer.full_to_swa_index_mapping
    full = plan.write_ids(plan._source)[:8].clone()
    assert len(full)==8 and full.unique().numel()==8
    pos = torch.arange(8)
    lens = torch.tensor([3,1],dtype=torch.int32)
    seq = torch.arange(8*5*2*8)-320
    projected = seq.reshape(8,-1).bfloat16()
    projected.view(torch.int16)[0,8] = -32768  # signed zero in the first value plane
    shaped = projected.view(8,5,2,8)
    failures=[]; matched=live=negative_zero=0; seen=[]
    for remap, fresh in enumerate(([19,12,26,31,4,38,5,10],[5,16,4,27,9,29,21,18])):
        allocmap[full]=torch.tensor(fresh)
        new = type(plan)(source=plan._source,req_pool_indices=plan.req_pool_indices,
                         seq_lens=plan.seq_lens,seq_lens_cpu=None,write_virtual=full)
        ids = w._draft_write_ids(new)
        assert ids.tolist()==list(fresh) and not torch.equal(ids,full)
        for prefix in (False,True):
            valid=torch.tensor([0,1,2,4]) if prefix else torch.arange(8)
            for stop in (0,1,3,5):
                for fixed in (False,True):
                    corrected=joined or text.replace(pattern,pattern.replace(
                        "self.draft_model_runner.token_to_kv_pool", "self._context_kv_pool()"))
                    w._append_paired_prompt_kv=bind(corrected if fixed else text,name)
                    for i in range(5):
                        for t in leaf.get_kv_buffer(i): t.fill_(-913)
                    expected=[]; outputs=[]
                    for i,layer in enumerate(w.draft_model.layers):
                        attn=layer.self_attn
                        k=attn.apply_k_rope(pos,attn.apply_k_norm(shaped[:,i,0].contiguous())).view(-1,1,8)
                        v=(shaped[:,i,1].contiguous()*attn.v_scale).view(-1,1,8)
                        outputs.append((k,v))
                        pair=[]
                        for ref,val in zip(leaf.get_kv_buffer(i),(k,v)):
                            ref=ref.clone();ref[ids[valid]]=val[valid];pair.append(ref)
                        expected.append(pair)
                    def f(ctx_hidden,positions,write_layer_kv,projection_ready):
                        assert projection_ready and torch.equal(ctx_hidden,projected)
                        for i,(k,v) in enumerate(outputs):
                            write_layer_kv(i,k,v)
                            if i+1==stop and stop!=5:raise RuntimeError("injected after a published layer")
                    w._use_fused_kv_materialize=bool(stop);w._fused_kv_helper=NS(materialize=f) if stop else None
                    try:
                        w._append_target_hidden_to_draft_kv_by_loc(target_hidden=projected,
                            cache_loc=ids,positions=pos,cache_loc_2d=ids.view(2,4) if prefix else None,
                            commit_lens=lens if prefix else None,prefill_context_done=True,prefill_kv_projected=True)
                    except (AttributeError,AssertionError) as ex:
                        assert not fixed and stop!=5 and type(ex) is (AttributeError if prefix else AssertionError)
                        if prefix: assert "set_kv_buffer_prefix_valid" in str(ex)
                        failures.append(dict(remap=remap,prefix=prefix,after_layer=stop,exception=type(ex).__name__))
                        continue
                    assert fixed or stop==5
                    for i in range(5):
                        for actual,ref in zip(leaf.get_kv_buffer(i),expected[i]):
                            assert torch.equal(actual.view(torch.int16),ref.view(torch.int16))
                            matched+=actual.numel();live+=len(valid)*8
                            negative_zero+=int((actual.view(torch.int16)==-32768).sum())
                    seen.append(dict(remap=remap,prefix=prefix,after_layer=stop,corrected=fixed))
    assert len(failures)==12 and len(seen)==20 and negative_zero>0
    return dict(joined_worker_sha=sha256(joined.encode()).hexdigest() if joined else None,
        tested_actual_joined_worker=joined is not None, v229_projected_helper_identical_to_v248=True,
        v248_worker_sha=sha256(text.encode()).hexdigest(),v253_worker_sha=sha256(source('v253',worker_path).encode()).hexdigest(),
        native_slot_plan=True,native_translator=True,native_outer_setter=True,sidecar=persisted['host_domain'],
        tested_mutable_physical_maps=2,failures=failures,passed=seen,compared_bf16_words=matched,committed_bf16_words=live,
        negative_zero_words=negative_zero,scope="Frozen source and CPU tensor stores; the fused materializer callback is controlled for failure injection.")

if __name__=="__main__":
    print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else None),indent=2))
