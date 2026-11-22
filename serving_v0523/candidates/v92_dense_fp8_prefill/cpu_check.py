"""Execute the submitted packing kernel on bounds-checked NumPy byte pointers."""
import argparse
import ast
import itertools
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

here = Path(__file__).resolve().parent
sim = runpy.run_path(str(here.parent / "v70_dflash_selector_tree/cpu_check.py"))
Pointer, make_array = sim["Pointer"], sim["array"]


def main(checkout):
    module = checkout / "python/sglang/kernels/ops/attention/dense_prefill_kv.py"
    pid, (kernel,) = sim["program"](module, {"_pack_prefill_kv"})
    rng = np.random.default_rng(20261118)
    copied, cases, launches = 0, 0, 0
    cdiv = lambda n, d: (n + d - 1) // d

    class Launcher:
        def __getitem__(self, grid):
            def run(k, v, table, lens, dk, dv, ks, vs, ts, width, ROW_BYTES, BLOCK_ROWS):
                nonlocal launches
                launches += 1
                # Include physical stride gaps and tensor storage offsets.
                def ptr(t):
                    n = t.untyped_storage().nbytes() // t.element_size()
                    flat = t.as_strided((n,), (1,), storage_offset=0).numpy()
                    return Pointer(flat, t.storage_offset())
                args = [ptr(t) for t in (k, v, table, lens, dk, dv)]
                for x, y in itertools.product(range(grid[0]), range(grid[1])):
                    pid[:2] = [make_array(x), make_array(y)]
                    kernel(*args, ks, vs, ts, width, ROW_BYTES, BLOCK_ROWS)
            return run

    tree = ast.parse(module.read_text())
    tree.body = [n for n in tree.body if getattr(n, "name", "") == "dense_prefill_kv"]
    env = {"triton": SimpleNamespace(cdiv=cdiv), "torch": torch, "_pack_prefill_kv": Launcher()}
    exec(compile(tree, str(module), "exec"), env)
    pack = env["dense_prefill_kv"]
    for dim, heads in ((16, 1), (32, 1), (64, 1), (64, 2), (64, 4), (128, 4), (128, 8)):
        for width in (1, 7, 31, 32, 33, 159, 160, 191, 192, 255, 256, 257, 513, 1025):
            batch = 4
            def alloc(gap):
                a = rng.integers(0, 256, (width + 4, 1, heads + gap, dim), dtype=np.uint8)
                return torch.from_numpy(a)[1:, :, :heads].view(torch.float8_e4m3fn)
            k, v = alloc(1), alloc(2)
            table = torch.full((batch, width + 9), -10**6, dtype=torch.int32)[:, 2:2 + width]
            lengths = torch.tensor([width, 0, max(1, width // 4), max(0, width - 1)], dtype=torch.int32)
            for i, n in enumerate(lengths.tolist()):
                table[i, :n] = torch.from_numpy(rng.integers(0, len(k), n, dtype=np.int32))
            dk, dv, pages = pack(k, v, table, lengths)
            assert pages is None and dk.is_contiguous() and dv.is_contiguous()
            for src, dst in ((k, dk), (v, dv)):
                for i, n in enumerate(lengths.tolist()):
                    src_rows = src.view(torch.uint8)[:, 0].index_select(0, table[i, :n])
                    assert torch.equal(src_rows, dst[i, :n].view(torch.uint8))
                    assert not torch.any(dst[i, n:min(width, n + 256)].view(torch.uint8))
                    copied += n * heads * dim
            cases += 1

    class Shape:
        def __init__(self, shape, dtype=torch.float8_e4m3fn, stride=None):
            self.shape, self.dtype, self._stride = shape, dtype, stride
        def stride(self, axis=None):
            strides = self._stride or tuple(np.prod(self.shape[i + 1:], dtype=int) for i in range(len(self.shape)))
            return strides if axis is None else strides[axis]
    good = Shape((10, 1, 2, 64))
    tab, lengths = Shape((1, 100)), Shape((1,))
    guards = [
        (good, good, Shape((2, 131072)), Shape((2,))),  # scratch limit
        (good, good, tab, Shape((1,), stride=(2,))),
        (good, good, Shape((1, 0)), lengths),
        (good, good, Shape((0, 100)), Shape((0,))),
        (good, good, Shape((1, 100), stride=(200, 2)), lengths),
        (good, Shape((10, 1, 1, 64)), tab, lengths),
        (good, Shape(good.shape, dtype=torch.bfloat16), tab, lengths),
        (good, Shape(good.shape, stride=(128, 128, 1, 2)), tab, lengths),
    ]
    for shape in ((10, 1, 1, 8), (10, 2, 2, 64), (10, 1, 3, 64), (10, 1, 32, 64)):
        bad = Shape(shape)
        guards.append((bad, bad, tab, lengths))
    for key, val, table, lens in guards:
        assert pack(key, val, table, lens) == (key, val, table)
    assert cases == launches
    print(json.dumps(dict(gather_cases=cases, live_bytes_checked=copied,
                          fallback_cases=len(guards), h200_execution=False), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    main(parser.parse_args().checkout)
