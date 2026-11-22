"""Replay V181's real compact routing + V183 W2 native launch ABI on CPU."""
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch


root = Path(__file__).resolve().parent.parent
stage = Path(sys.argv[1])
source = root / "v177_grouped_gate_up_epilogue/cpu_check.py"
# Reuse the V177 paired-bank/CPU projection fixture without running its old case matrix.
tree = ast.parse(source.read_text())
tree.body = tree.body[:next(i for i, node in enumerate(tree.body) if isinstance(node, ast.For))]
donor = {"__file__": str(source)}
exec(compile(tree, str(source), "exec"), donor)
prior = donor["prior"]
p = stage / "python/sglang/srt/layers/muse_spark/prefill_mma_pair.py"
down = prior.module("down", p)
ops = ModuleType("humming.ops")
humming = ModuleType("humming"); humming.ops = ops
sys.modules["humming"] = humming
w2_by_weight = {}
hits = Counter()


def fixture(counters):
    f = donor["fixture"](counters)
    f.env["run_down"] = down.run_down
    return f


def bank(f, rank, size, hidden):
    layer = donor["bank"](f, rank, size, hidden)
    layer.w2_weight = torch.empty(1, dtype=torch.int32)
    layer.w2_weight_scale = torch.empty(1, dtype=torch.uint8)
    layer.humming_prefill_down = {w: w or 176 for w in (None, 136, 152, 176)}
    if down_mode == "none":
        layer.humming_prefill_down.clear()
    if down_mode == "missing_one":
        layer.humming_prefill_down[136] = None
    w2_by_weight[id(layer.w2_weight)] = (f, layer)
    return layer


def kernel(*, configs, inputs, outputs, weight, weight_scale, expert_layout, locks):
    f, layer = w2_by_weight[id(weight)]
    assert weight is layer.w2_weight and weight_scale is layer.w2_weight_scale
    assert locks is getattr(layer, "locks", None)
    hits["calls"] += 1
    hits["routes"] += len(inputs)
    hits[f"m{configs}"] += 1
    # Only the final native W2 operator is stood in on CPU.
    f.env["HummingMethod"].forward_layer(layer, inputs=inputs, outputs=outputs,
        expert_layout=expert_layout, compute_config="grouped", tuning_config=configs, sublayer_name="w2")


ops.launch_kernel = kernel
prior.fixture = fixture
prior.v168.prior.bank = bank
cases = {}
for gate_mode, down_mode in [("active", "all"), ("map_failure", "all"), ("active", "missing_one"), ("active", "none")]:
    donor["mode"] = gate_mode
    before = hits.copy()
    values = prior.numerical()
    current = hits - before
    assert values["cases"] == 50 and bool(current) == (down_mode != "none")
    cases[f"{gate_mode}_{down_mode}"] = dict(numerical=values, native_w2=current)
assert set(hits) >= {"m136","m176","calls","routes"}
print(json.dumps(dict(cases=cases, actual_native_w2_adapter=hits, physical_gpu=False), indent=2))
