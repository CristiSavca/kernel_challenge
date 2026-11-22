"""Rebuild each candidate's DFlash module from the pinned source and replay its head guard."""
import ast
import builtins
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import symtable
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

CANDIDATES = Path(__file__).resolve().parent.parent
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
MODEL = "python/sglang/srt/models/dflash.py"
UTILS = "python/sglang/srt/utils/common.py"
STOCK = CANDIDATES.parent / "sglang"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode()


def scope(source):
    module = symtable.symtable(source, MODEL, "exec")
    func = next(c for c in module.get_children() if c.get_name() == "candidate_topk")
    globals_used = {s.get_name() for s in func.get_symbols() if s.is_referenced() and s.is_global()}
    defined = {s.get_name() for s in module.get_symbols() if s.is_assigned() or s.is_imported()}
    return sorted(globals_used - defined - set(dir(builtins)))


def defs(source, names):
    return [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]


def execute(nodes, env):
    ast.fix_missing_locations(mod := ast.Module(body=nodes, type_ignores=[]))
    exec(compile(mod, MODEL, "exec", flags=__import__("__future__").annotations.compiler_flag), env)


def replay(source, expect_failure=False):
    # This entry never gets a synthetic is_cuda global. Only imports actually present in
    # the submitted module populate it, using the pinned SGLang implementation.
    utils = ModuleType("sglang.srt.utils")
    utils.torch = torch
    utils.lru_cache = __import__("functools").lru_cache
    execute(defs(git(STOCK,"show",BASE+":"+UTILS), {"is_cuda"}), utils.__dict__)
    utils.is_npu = lambda: False
    utils.set_weight_attrs = lambda *a: None
    entries = [n for n in ast.parse(source).body if isinstance(n,ast.ImportFrom)
               and n.module == "sglang.srt.utils" and any(a.name == "is_npu" for a in n.names)]
    # Keep the common utility aliases of this import available to the extracted function.
    for entry in entries:
        entry.names = [a for a in entry.names if a.name in ("is_cuda","is_npu","set_weight_attrs")]
    env = dict(torch=torch, should_apply_lm_head_quant_method=lambda head, quant:False,
               is_dense_head_weight=lambda w:True, get_parallel=lambda:SimpleNamespace(tp_size=1),
               _flashinfer_top_k=None)
    name = "sglang.kernels.ops.speculative.dflash_candidate_head"
    kernel = ModuleType(name)
    weight = torch.eye(8, dtype=torch.bfloat16)
    hidden = torch.arange(32,dtype=torch.float32).bfloat16().reshape(4,8)
    head = SimpleNamespace(weight=weight, org_vocab_size=8)
    ref_vals, ref_ids = torch.topk(hidden @ weight.T,3)
    cases = []
    with patch.dict(sys.modules, {name:kernel,"sglang.srt.utils":utils}):
        execute(entries + defs(source, {"_project_candidate_logits","_radix_topk","candidate_topk"}),env)
        for available, enabled, fallback in ((True,True,False),(True,True,True),(False,True,False),(True,False,False)):
            calls=[]
            def probe(h,w,num,off,k):
                calls.append(1)
                assert num==8 and off==0 and k==3
                if fallback:
                    return None
                v,i=torch.topk(h @ w.T,k)
                return torch.stack((v.float().view(torch.int32),i.int()),dim=-1).flatten(1)
            kernel.fused_local_candidate_pairs = probe
            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.cuda,"is_available",return_value=available))
                stack.enter_context(patch.object(torch.version,"cuda","12.6" if available else None))
                utils.is_cuda.cache_clear()
                try:
                    ids, vals, _ = env["candidate_topk"](hidden,head,3,fuse_candidate_head=enabled)
                except NameError as e:
                    if expect_failure and e.name=="is_cuda":
                        return "NameError: is_cuda"
                    raise
            assert not expect_failure and torch.equal(ids,ref_ids) and torch.equal(vals,ref_vals)
            assert len(calls)==int(available and enabled)
            cases.append([available,enabled,fallback])
    return len(cases)


if __name__ == "__main__":
    result = {}
    api = "fuse_candidate_head and not with_partition and not use_quant_head and is_cuda()"
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        git(repo, "init", "-q")
        (repo/".git/objects/info/alternates").write_text(str(STOCK/".git/objects")+"\n")
        for p in sorted(CANDIDATES.glob("*/patch.diff")):
            if api not in p.read_text():
                continue
            git(repo,"read-tree",BASE)
            git(repo,"apply","--cached","--include="+MODEL,str(p))
            source=git(repo,"show",":"+MODEL)
            missing=scope(source)
            old = p.parent.name == "v88_v64_fused_draft_head"
            assert missing == (["is_cuda"] if old else []), (p, missing)
            result[p.parent.name] = dict(unbound=missing)
            if old or p.parent.name.startswith(("v88a_", "v93_")):
                result[p.parent.name]["cpu_replay"]=replay(source,old)
    assert all(any(s.startswith(v) for s in result) for v in ("v88_","v88a_","v93_"))
    print(json.dumps(result,indent=2))
