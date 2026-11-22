"""CPU execution of the compact projection planner, GEMM and dispatch guards."""

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_fixture():
    path = Path(__file__).resolve().parents[1] / "v70_dflash_selector_tree/cpu_check.py"
    spec = importlib.util.spec_from_file_location("tree_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    return fixture


fx = load_fixture()


def kernels(source):
    ids, (plan, project) = fx.program(source, ("_plan_rows", "_project"))
    project.__globals__["tl"].trans = lambda tensor: tensor.T
    return ids, plan, project


def source_map(plan, order, lengths):
    rows = order.size
    width = order.shape[1]
    metadata = np.full(2 * rows + 1, -1, np.int32)
    plan(fx.Pointer(order), fx.Pointer(lengths), fx.Pointer(metadata), rows, width,
         max(256, 1 << (rows - 1).bit_length()))
    live = np.arange(width) < lengths[:, None]
    expected_src = (np.arange(len(lengths))[:, None] * width + order)[live]
    expected_dst = np.flatnonzero(live)
    total = len(expected_src)
    assert metadata[-1] == total
    np.testing.assert_array_equal(metadata[:total], expected_src)
    np.testing.assert_array_equal(metadata[rows:rows + total], expected_dst)
    assert np.all(metadata[total:rows] == -1) and np.all(metadata[rows + total:-1] == -1)
    return metadata, expected_src, expected_dst


def plans(plan, rng):
    cases = 0
    for batch, width in itertools.product((1, 3, 8, 15, 31, 64, 128), (2, 4, 8, 12, 16)):
        order = np.stack([rng.permutation(width) for _ in range(batch)])
        lengths = [np.zeros(batch, int), np.ones(batch, int), np.full(batch, width),
                   np.arange(batch) % (width + 1), rng.integers(width + 1, size=batch)]
        for spans in lengths:
            source_map(plan, order, spans)
            cases += 1
    return cases


def projections(ids, plan, project, rng):
    cases = 0
    for batch, width, k, n in itertools.product((8, 17), (4, 8, 12), (64, 128, 320), (32, 96)):
        rows = batch * width
        order = np.stack([rng.permutation(width) for _ in range(batch)])
        lens = rng.integers(width + 1, size=batch)
        meta, src, dest = source_map(plan, order, lens)
        # Small integers make BF16 inputs and the FP32 sum exactly representable.
        x = rng.integers(-4, 5, size=(rows, k)).astype(np.float32)
        w = rng.integers(-3, 4, size=(n, k)).astype(np.float32)
        x[np.setdiff1d(np.arange(rows), src)] = np.nan
        out = np.zeros((rows, n), np.float32)
        for m_idx, n_idx in itertools.product(range((rows + 15)//16), range(n//32)):
            ids[:2] = m_idx, n_idx
            project(*map(fx.Pointer, (x, w, meta, out)), rows, n, k, 16, 32, 64)
        ref = np.zeros_like(out)
        ref[dest] = x[src] @ w.T
        np.testing.assert_array_equal(out, ref)
        cases += 1
    return cases


class Tensor:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype
        self.device, self.is_cuda, self.contiguous = "cuda:0", True, True

    def is_contiguous(self):
        return self.contiguous


def guards(source):
    ctx = SimpleNamespace(deterministic=False, capability=(9, 0))
    torch = SimpleNamespace(
        bfloat16="bf16", int32="int32", int64="int64",
        cuda=SimpleNamespace(get_device_capability=lambda device: ctx.capability),
        are_deterministic_algorithms_enabled=lambda: ctx.deterministic,
    )
    _, (dispatch,) = fx.program(source, ("project_committed_rows",), dependencies={"torch": torch})
    h = Tensor((128, 40960), "bf16")
    w = Tensor((1024, 40960), "bf16")
    o = Tensor((16, 8), "int64")
    l = Tensor((16,), "int32")
    mutations = [
        (h, "is_cuda", False), (h, "dtype", "fp16"), (w, "dtype", "fp16"),
        (o, "dtype", "int32"), (l, "dtype", "int64"), (l, "shape", (8, 2)),
        (h, "shape", (128, 40959)), (w, "shape", (1023, 40960)),
        (h, "shape", (127, 40960)), (o, "shape", (128, 1)),
        (ctx, "capability", (10, 0)), (ctx, "deterministic", True),
    ]
    mutations.extend((t, "device", "cuda:1") for t in (w, o, l))
    mutations.extend((t, "contiguous", False) for t in (h, w, o, l))
    for obj, field, value in mutations:
        old = getattr(obj, field)
        setattr(obj, field, value)
        assert dispatch(h, w, o, l) is None, (field, value)
        setattr(obj, field, old)
    return len(mutations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    root = parser.parse_args().checkout
    source = root / "python/sglang/kernels/ops/speculative/dflash_committed_projection.py"
    rng = np.random.default_rng(82)
    pids, planner, mul = kernels(source)
    print(f"{plans(planner, rng)} accepted-row compaction layouts")
    print(f"{projections(pids, planner, mul, rng)} projection layouts with poisoned rejected rows")
    print(f"{guards(source)} dispatch fallback cases")
