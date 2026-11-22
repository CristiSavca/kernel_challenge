"""Exercise actual storage requests and both projection layouts on host PyTorch."""

import ast
from collections import Counter
from contextlib import redirect_stdout
import io
from itertools import product
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import torch


here = Path(__file__).resolve().parent
stage = Path(sys.argv[1]).resolve()
prior = next(here.parent.glob("v308_*/joined_replay.py"))
log = io.StringIO()
with redirect_stdout(log):
    fixture = runpy.run_path(str(prior))
parent_cases = json.loads(log.getvalue())
world = fixture["bind"].__globals__
ops, layer, graph = (world[n] for n in ("ops", "layer", "graph"))
wf = world["wf"]
tree = ast.parse(wf.read_bytes())
helpers = [n for n in tree.body if getattr(n, "name", None) == "wave_workspace_elements"]
assert len(helpers) == 1
exec(compile(ast.Module(helpers, []), str(wf), "exec"), ops)

hidden, scores, topk, experts = (world[n] for n in ("hidden", "scores", "k", "e"))
tokens, width = hidden.shape
slots = tokens * topk
ordinary_width = max(width, layer.intermediate_size_per_partition) + max(width, 2 * layer.intermediate_size_per_partition)
allocations = []
views = []
original_empty = torch.Tensor.new_empty
original_bind = world["mods"][-1].bind_prompt_wave


def watch_empty(tensor, *args, **kwargs):
    result = original_empty(tensor, *args, **kwargs)
    if tensor is hidden:
        allocations.append((result.untyped_storage().data_ptr(), result.untyped_storage().nbytes()))
    return result


def watch_bind(binding, packed, active):
    address, size = allocations[-1]
    assert packed.untyped_storage().data_ptr() == active.untyped_storage().data_ptr() == address
    assert packed.untyped_storage().nbytes() == active.untyped_storage().nbytes() == size
    assert packed.storage_offset() == 0 and active.storage_offset() == packed.numel()
    views.append((packed.shape[0], packed.numel() + active.numel()))
    return original_bind(binding, packed, active)


ids = torch.arange(slots, dtype=torch.int32).view(tokens, topk).remainder(experts)
layouts = {"all_local": ids, "duplicate": ids.new_zeros(ids.shape), "all_remote": ids.new_full(ids.shape, -1)}
ragged = ids.clone()
ragged[1::2] = -1
ragged[2::4, 1:] = -1
layouts["skew_and_trailing"] = ragged
counts = Counter()
sizes = {}
for mode, conditional, capture, tp, (name, local_ids) in product(
    ("fused", "declined", "missing"), (True, False), (True, False), (1, 8, 64), layouts.items(),
):
    layer.global_num_experts = tp * experts
    layer.humming_conditional_waves = conditional
    layer.humming_prompt_gate_up = None if mode == "missing" else True
    world["w13"] = mode
    layer.humming_prompt_down = (world["config"], "meta", "grouped", "w2", world["weight_binding"])
    world["calls"].clear()
    graph.iterations = None
    allocations.clear()
    views.clear()
    reference = world["base"]["reference"](layer, hidden, scores, local_ids)
    with patch.dict(sys.modules, {m.__name__: m for m in world["mods"]}), patch.object(
        torch.Tensor, "new_empty", watch_empty
    ), patch.object(world["mods"][-1], "bind_prompt_wave", watch_bind), patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=capture
    ), patch.object(torch.cuda, "CUDAGraph", graph):
        output = ops["grouped_prefill_waves"](layer, hidden, scores, local_ids, slots)
    assert torch.equal(reference.view(torch.int16), output.view(torch.int16))
    assert len(allocations) == 1
    assert len(views) == int(mode != "missing")
    expected_live = int(((local_ids >= 0) & (local_ids < experts)).sum())
    for parity in (0, 1):
        assert sum(rows for _, rows in world["calls"][parity::2]) == expected_live
    size = allocations[0][1]
    smaller = conditional and mode != "missing" and tp > 1
    assert (size < 2 * slots * ordinary_width) == smaller
    sizes[f"{mode}/conditional={conditional}/shards={tp}"] = size
    counts["cases"] += 1
    counts["smaller_single_allocation_cases"] += smaller
    counts["bf16_words"] += output.numel()
    counts["live_routes"] += expected_live
    counts["max_projected_wave_pairs"] = max(counts["max_projected_wave_pairs"], len(world["calls"]) // 2)
assert counts["max_projected_wave_pairs"] <= 16
print(json.dumps({
    "inherited_joined_replay": parent_cases,
    "counts": counts,
    "observed_host_arena_bytes": sizes,
    "scope": "PyTorch allocations and bitwise CPU model of graph capture, both projections and finalization.",
    "physical_H200": False,
}, indent=2))
