"""Validate the reservation, row capacity and launch count across supported dimensions."""

import ast
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def cdiv(a, b):
    return (a + b - 1) // b


def power_of_two(n):
    return 1 << (n - 1).bit_length()


stage = Path(sys.argv[1])
wave = stage / "python/sglang/srt/layers/muse_spark/humming_prefill_waves.py"
source = ast.parse(wave.read_bytes())
names = {"wave_capacity", "wave_workspace_elements"}
ops = {"triton": SimpleNamespace(cdiv=cdiv, next_power_of_2=power_of_two)}
exec(compile(ast.Module([n for n in source.body if getattr(n, "name", "") in names], []), str(wave), "exec"), ops)
accepted = smaller = fallback_floor = highest_arena_bytes = highest_static_waves = 0
examples = {}
for tokens, topk, experts, global_experts, hidden, intermediate in product(
    (1, 2, 7, 15, 31, 63, 846, 4095, 4096, 8191, 8192, 16376, 16384, 16395, 24576, 32768, 65535),
    range(1, 9), (1, 7, 32), (8, 56, 256, 512), (7, 256, 4096, 8192), (11, 256, 2048, 4096, 14336),
):
    if experts > global_experts:
        continue
    old_cap = ops["wave_capacity"](tokens, topk, experts, hidden, intermediate)
    if not old_cap:
        continue
    slots = tokens * topk
    rows = min(slots, old_cap + topk - 1)
    ordinary = max(hidden, intermediate) + max(hidden, 2 * intermediate)
    compact = hidden + intermediate
    old = rows * ordinary
    size = ops["wave_workspace_elements"](slots, topk, experts, global_experts, hidden, intermediate, rows)
    assert 0 < size <= old <= (384 << 20) // 2
    layouts = {}
    for path, width in (("fused", compact), ("declined", ordinary)):
        physical = min(slots, size // width)
        cap = physical if physical == slots else physical - topk + 1
        if path == "declined":
            cap = min(cap, old_cap)
        assert 0 < cap <= physical and min(slots, cap + topk - 1) <= physical
        assert (experts + 1) * power_of_two(max(slots, physical)) < 2**31
        max_waves = cdiv(slots, cap)
        assert max_waves <= 16
        layouts[path] = {"rows": physical, "nominal_capacity": cap, "max_waves": max_waves}
        highest_static_waves = max(highest_static_waves, max_waves)
    minimum = min(slots, cdiv(slots, 16) + topk - 1)
    fallback_floor += size == minimum * ordinary < old
    smaller += size < old
    accepted += 1
    highest_arena_bytes = max(highest_arena_bytes, 2 * size)
    if tokens in (8192, 16384, 24576, 32768) and (topk, experts, global_experts, hidden, intermediate) == (8, 32, 256, 4096, 4096):
        examples[tokens] = {
            "V308_arena_bytes": 2 * old, "V309_arena_bytes": 2 * size,
            "exactly_balanced_V204_fused_stage_and_activation_bytes": 2 * cdiv(slots * experts, global_experts) * compact,
            "V308_fused_rows": min(slots, old // compact), "V309_layouts": layouts,
        }
assert smaller and fallback_floor and highest_static_waves == 16
print(json.dumps({
    "admitted_shape_combinations": accepted, "smaller_reservations": smaller,
    "shapes_limited_by_fallback_floor": fallback_floor, "highest_arena_bytes": highest_arena_bytes,
    "highest_static_waves": highest_static_waves, "examples_TP8_BF16": examples,
    "scope": "Workspace reservation and launch bounds from host dimensions. Excludes other live tensors, CUDA graph pools and library allocations.",
}, indent=2))
