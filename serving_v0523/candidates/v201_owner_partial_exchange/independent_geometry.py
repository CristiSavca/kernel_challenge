"""Exercise allocation, eligibility and row-window geometry without allocating a GPU tensor."""
import ast
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

SRC = "python/sglang/srt/layers/communication/sp_owner_sum.py"
ROOT = Path(sys.argv[1])
SOURCE = (ROOT / SRC).read_bytes()


class Shape:
    def __init__(self, dimensions, dtype="bf16", cuda=True, contiguous=True):
        self.shape = dimensions
        self.ndim = len(dimensions)
        self.dtype = dtype
        self.is_cuda = cuda
        self.is_contiguous = contiguous

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 2

    def contiguous(self):
        return self if self.is_contiguous else Shape(self.shape)

    def new_empty(self, dimensions):
        return Shape(dimensions)


def native_functions():
    wanted = {"owner_sum_eligible", "_owner_shape", "_window_rows", "owner_sum"}
    module = ast.parse("from __future__ import annotations")
    for node in ast.parse(SOURCE).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in {"_MAX_RECEIVE_BYTES", "_WINDOW_SLACK_BYTES"}:
            module.body.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            module.body.append(node)
    group = object()
    calls = []

    def exchange(full, selected_group, out, received):
        assert selected_group is group and full.is_contiguous
        assert out.shape[1] == received.shape[2] == full.shape[1]
        calls.append((out.shape, received.shape))

    env = {
        "torch": SimpleNamespace(bfloat16="bf16"),
        "get_parallel": lambda: SimpleNamespace(tp_group=group),
        "_exchange": exchange,
    }
    exec(compile(module, SRC, "exec"), env)
    return env, calls


def verify():
    env, calls = native_functions()
    eligible = env["owner_sum_eligible"]
    max_bytes = 32 << 20
    slack = 256 << 10
    rng = random.Random(181838)
    geometries = set()
    for size in range(2, 17):
        max_width = max_bytes // (2 * (size - 1))
        for width in (1, 3, 255, 256, 1537, 4096, 8192, 8193, max_width - 1, max_width, max_width + 1):
            threshold = (max_bytes + width - 1) // width
            for rows in (0, size - 1, size, threshold - 1, threshold, threshold + 1, 16383, 16384, 16385):
                geometries.add((size, rows, width))
        for _ in range(1024):
            geometries.add((size, rng.randint(1, 64 << 10), rng.randint(1, max_width + 1)))
    positives = declines = improved = tighter = 0
    examples = []
    for size, rows, width in sorted(geometries):
        full = Shape((rows, width), contiguous=False)
        row_bytes = (size - 1) * width * 2
        valid = rows >= size and row_bytes <= max_bytes and full.numel() * 2 >= 2 * max_bytes
        assert eligible(full, size) == valid
        if not valid:
            declines += 1
            continue
        positives += 1
        env["owner_sum"](full, size)
        out, received = calls.pop()
        span = math.ceil(rows / size)
        assert out == (span, width)
        old_window = min(span, max_bytes // row_bytes)
        window = received[1]
        assert received == (size - 1, window, width)
        assert 0 < window <= span
        assert window * row_bytes <= max_bytes + slack
        old_groups = math.ceil(span / old_window)
        groups = math.ceil(span / window)
        assert groups <= old_groups
        improved += groups < old_groups
        tighter += window < old_window
        assert window * width < 2**31
        if size == 8 and rows == 16384 and width in (4096, 8192):
            examples.append(dict(
                tp=size, rows=rows, width=width, span=span, old_window=old_window,
                window=window, old_groups=old_groups, groups=groups,
                old_remote_bytes=old_window * row_bytes, remote_bytes=window * row_bytes,
            ))
    assert [e["groups"] for e in examples] == [4, 7]
    assert [e["old_groups"] for e in examples] == [4, 8]
    assert [e["remote_bytes"] for e in examples] == [28 << 20, (32 << 20) + (48 << 10)]
    invalid_metadata = (
        (Shape((16384, 8192), dtype="fp16"), 8),
        (Shape((16384, 8192), cuda=False), 8),
        (Shape((16384, 1, 8192)), 8),
        (Shape((16384, 8192)), 1),
        (Shape((16384, 8192)), 17),
        (Shape((16384, 0)), 8),
    )
    assert all(not eligible(full, size) for full, size in invalid_metadata)
    return dict(source_sha256=hashlib.sha256(SOURCE).hexdigest(), valid_cases=positives,
                declined_shapes=declines, declined_metadata=len(invalid_metadata),
                fewer_rounds_vs_hard_cap=improved, smaller_buffer_vs_hard_cap=tighter,
                maximum_allowed_remote_bytes=max_bytes + slack, examples=examples)


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
