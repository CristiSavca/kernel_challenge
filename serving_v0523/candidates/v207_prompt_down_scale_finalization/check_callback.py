"""Compare V206 and V207 at the grouped callback and count materialized torch operations.

The real host allocator, sort, gather, branch selection and BF16 multiply run on
CPU. Humming and the final GPU launch use host arithmetic. finalizer.json covers
the actual Triton arithmetic separately with checked pointer reads.
"""

import ast
from collections import Counter
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import MethodType
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode


HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
MANIFEST = json.loads((HERE / "validation.json").read_bytes())
BASE = HERE.parent / "v199_prompt_expert_overlap"
fixture = runpy.run_path(str(BASE / "independent_callback.py"))
layer = fixture["layer"]
host = fixture["ns"]
hum = "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
moe = hum.replace("muse_spark_humming_moe", "muse_spark_fp4_moe")
launches = Counter()


def definition(tree, path, symbol, scope, cls=None):
    source = subprocess.check_output(["git", "-C", STAGE, "show", f"{tree}:{path}"])
    body = ast.parse(source).body
    if cls:
        body = next(node.body for node in body if getattr(node, "name", None) == cls)
    node = next(node for node in body if getattr(node, "name", None) == symbol)
    node.decorator_list = []
    host["add"](ast.unparse(node), scope)
    return scope[symbol]


class Ops(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.mul_.Tensor:
            self.ops["full BF16 elementwise scale passes"] += 1
            self.ops["BF16 elements in materialized scale pass"] += args[0].numel()
        if func == torch.ops.aten.index_select.default and args[0].ndim == 1:
            self.ops["per-route scale gathers"] += 1
        return func(*args, **(kwargs or {}))


class Finalizer:
    def __getitem__(self, grid):
        def launch(values, weight, scores, out, rowmap, scale_ids, scales, **k):
            assert k["HAS_WEIGHT"] and k["APPLY_NORM"] and not k["ROUND_PRODUCT"]
            assert grid == (out.shape[0],)
            packed = k["HAS_ROW_MAP"]
            fused_scale = k["HAS_EXPERT_SCALE"]
            assert not fused_scale or (packed and k["HAS_IDS"] and scales is layer.w2_global_scale)
            launches["scaled compact" if fused_scale else "ordinary compact" if packed else "indexed"] += 1
            total = torch.zeros_like(out, dtype=torch.float32)
            for slot in range(scores.shape[1]):
                live = torch.where(rowmap[:, slot] >= 0)[0]
                if not len(live):
                    continue
                indices = rowmap[live, slot].long()
                x = values[indices].float() if packed else values[live, slot].float()
                if fused_scale:
                    factor = scales[scale_ids[live, slot].long()].float()
                    x = (x * factor[:, None]).to(values.dtype).float()
                if k["WEIGHT_FIRST"]:
                    x = x * weight.float()
                x = x * torch.rsqrt((x * x).sum(1, keepdim=True) / x.shape[1] + k["eps"])
                if not k["WEIGHT_FIRST"]:
                    x = x * weight.float()
                total[live] += x * scores[live, slot, None].float()
            out.copy_(total)
        return launch


def indexed(owner, x, ids):
    # The real indexed fallback yields a three-dimensional already-scaled tensor.
    return x[:, None, :].expand(-1, ids.shape[1], -1).clone(), ids


def compare():
    funcs = []
    fixture["scope"]["_post_expert_norm_weighted_sum_kernel"] = Finalizer()
    for tree in (MANIFEST["parent_tree"], MANIFEST["tree"]):
        group_scope = dict(host["env"], humming_expert_outputs=indexed)
        group = definition(tree, hum, "humming_grouped_expert_outputs", group_scope)
        combine = definition(tree, moe, "_grouped_humming_and_combine_impl",
                             fixture["scope"].copy(), "MuseSpark1xFp4Experts")
        funcs.append((group, combine))

    outcomes = []
    rows = 8192
    for route in ("mixed", "empty", "dense_fallback"):
        x = host["rand"](rows, 128)
        ids = torch.full((rows, 8), -1, dtype=torch.int32)
        if route == "mixed":
            ids[::17, 1] = 4
            ids[::17, 6] = 4
            ids[::23, 2] = 3
        elif route == "dense_fallback":
            ids[:] = torch.arange(8, dtype=ids.dtype)
        scores = host["rand"](rows, 8).float()
        n = int((ids >= 0).sum())
        layer.intermediate_size_per_partition = 64 if route == "dense_fallback" else 128
        for gate_mode in ("paired", "disabled"):
            layer.humming_prompt_gate_up = host["pstate"](gate_mode)
            for down_mode in ("paired", "ordinary"):
                layer.humming_prompt_down = fixture["bind"]() if down_mode == "paired" else None
                for weight_first in (True, False):
                    layer._post_expert_norm_weight_first = weight_first
                    runs = []
                    for group, combine in funcs:
                        fixture["model_hum"].humming_grouped_expert_outputs = group
                        layer._grouped_humming_and_combine = MethodType(combine, layer)
                        with Ops() as ops:
                            out = fixture["forward"](x, scores, ids)
                        runs.append((out, ops.ops))
                    assert torch.equal(runs[0][0].contiguous().view(torch.int16),
                                       runs[1][0].contiguous().view(torch.int16))
                    old, new = runs[0][1], runs[1][1]
                    if route == "mixed":
                        assert old["per-route scale gathers"] - new["per-route scale gathers"] == 1
                        assert old["full BF16 elementwise scale passes"] - new["full BF16 elementwise scale passes"] == 1
                        assert (old["BF16 elements in materialized scale pass"] -
                                new["BF16 elements in materialized scale pass"]) == n * x.shape[1]
                    else:
                        assert old == new
                    outcomes.append(dict(route=route, gate=gate_mode, down=down_mode,
                                         weight_first=weight_first, live_routes=n,
                                         bf16_words=out.numel(), old=dict(old), new=dict(new)))
    return outcomes


if __name__ == "__main__":
    with torch.inference_mode(), patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
        result = compare()
    print(json.dumps(dict(candidate=MANIFEST["patch_sha256"], parent=MANIFEST["parent_sha256"],
                          cases=len(result), matched_output_words=sum(v["bf16_words"] for v in result),
                          finalizer_calls=dict(launches), examples=result, physical_H200=False), indent=2))

