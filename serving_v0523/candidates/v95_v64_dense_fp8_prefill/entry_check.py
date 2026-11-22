"""Compare the real capture-time eligibility and FA call ASTs on the two parents."""

import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace


def attribute_assignment(module, name):
    return next(
        n.value for n in ast.walk(module) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == name for t in n.targets)
    )


def pack_cond(module):
    return next(
        n for n in ast.walk(module) if isinstance(n, ast.If) and n.body
        and isinstance(n.body[0], ast.Assign)
        and isinstance(n.body[0].value, ast.Call)
        and getattr(n.body[0].value.func, "id", None) == "dense_prefill_kv"
    )


def fa_call(module, query_name):
    return next(
        n for n in ast.walk(module) if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "flash_attn_with_kvcache"
        and any(k.arg == "cu_seqlens_k_new" and isinstance(k.value, ast.IfExp) for k in n.keywords)
        and any(k.arg == "q" and ast.unparse(k.value) == query_name for k in n.keywords)
    )


def run(current, v92, stock):
    new = ast.parse(current.read_text())
    old = ast.parse(v92.read_text())
    base = ast.parse(stock.read_text())
    eligible = attribute_assignment(new, "_dense_fp8_prefill")
    sms = attribute_assignment(old, "_hopper_verify_sms")
    assert isinstance(sms, ast.IfExp) and ast.dump(eligible) == ast.dump(sms.test)
    condition = pack_cond(new)
    source = ast.dump(condition.test)
    assert source == ast.dump(pack_cond(old).test).replace("_hopper_verify_sms", "_dense_fp8_prefill")
    assert ast.dump(condition.body[0].value) == ast.dump(pack_cond(old).body[0].value)
    before = fa_call(base, "q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)")
    after = copy.deepcopy(fa_call(new, "query"))
    after.keywords[0].value = before.keywords[0].value
    assert ast.dump(before) == ast.dump(after)

    # Evaluate the actual production guard using synthetic request metadata.
    code = compile(ast.Expression(condition.test), str(current), "eval")
    mode = SimpleNamespace(EXTEND="extend")
    passing = dict(self=SimpleNamespace(_dense_fp8_prefill=True, page_size=1,
                                        kv_index_translator=SimpleNamespace(is_translating=False)),
                   forward_batch=SimpleNamespace(forward_mode=mode.EXTEND, dcp_kv_mask=None),
                   ForwardMode=mode, max_seqlen_q=512, causal=True, is_swa_layer=False,
                   use_local_attn=False, return_lse=False, query=SimpleNamespace(dtype="fp8"),
                   torch=SimpleNamespace(float8_e4m3fn="fp8"), kwargs={"k_descale": 1, "v_descale": 2})
    assert eval(code, passing)
    failures = [("self._dense_fp8_prefill", False), ("self.page_size", 2),
                ("self.kv_index_translator.is_translating", True), ("forward_batch.forward_mode", "verify"),
                ("forward_batch.dcp_kv_mask", 1), ("max_seqlen_q", 511), ("causal", False),
                ("is_swa_layer", True), ("use_local_attn", True), ("return_lse", True),
                ("query.dtype", "bf16"), ("kwargs", {"sinks": 1})]
    for path, value in failures:
        env = dict(passing)
        for key in ("self", "forward_batch", "query"):
            env[key] = copy.deepcopy(passing[key])
        parts = path.split(".")
        if len(parts) == 1:
            env[path] = value
        else:
            obj = env[parts[0]]
            for attr in parts[1:-1]:
                obj = getattr(obj, attr)
            setattr(obj, parts[-1], value)
        assert not eval(code, env), path
    return dict(kernel_entry_decisions=1 + len(failures), same_fa_arguments=True,
                v92_init_predicate_identical=True, v92_request_predicate_identical=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("v95_backend", "v92_backend", "v64_backend"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.v95_backend, args.v92_backend, args.v64_backend), indent=2))
