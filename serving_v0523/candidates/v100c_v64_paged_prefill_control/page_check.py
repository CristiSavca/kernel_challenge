"""Check the V60 page indexing and extension kernels in a patched SGLang tree."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class Pointer:
    def __init__(self, array, offset=0):
        self.array = array
        self.offset = offset

    def __add__(self, delta):
        return Pointer(self.array, self.offset + delta)

    def __sub__(self, delta):
        return Pointer(self.array, self.offset - delta)


def addresses(pointer, mask):
    offset, mask = np.broadcast_arrays(np.asarray(pointer.offset), mask)
    active = offset[mask]
    assert np.all((active >= 0) & (active < pointer.array.size))
    return offset, mask, active


def load(pointer, mask=True, other=0):
    offset, mask, active = addresses(pointer, mask)
    result = np.full(offset.shape, other, dtype=pointer.array.dtype)
    result[mask] = pointer.array[active]
    return result


def store(pointer, value, mask=True):
    offset, mask, active = addresses(pointer, mask)
    pointer.array[active] = np.broadcast_to(value, offset.shape)[mask]


def source_fn(path, name, ns):
    mod = ast.parse(path.read_text())
    fn = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


class SignedToken(np.ndarray):
    # Triton integer division truncates toward zero, including during the signed fixup.
    def __floordiv__(self, divisor):
        a = np.asarray(self)
        return (np.sign(a) * (np.abs(a) // divisor)).view(SignedToken)


def main():
    root = Path(sys.argv[1]) / "python/sglang/kernels/ops"
    tl = SimpleNamespace(load=load, store=store, arange=np.arange, sum=np.sum, where=np.where, constexpr=int)
    power2 = lambda n: 1 << (int(n) - 1).bit_length()
    ns = {"tl": tl, "triton": SimpleNamespace(next_power_of_2=power2)}
    alloc = source_fn(root / "memory/allocator.py", "alloc_extend_kernel", ns)
    page_index = source_fn(root / "attention/metadata.py", "_draft_page_index", ns)
    total = alloc_cases = 0
    rng = np.random.default_rng(100)
    for page in (1, 2, 7, 16, 64, 80, 128, 160, 192):
        values = np.arange(-1400, 1401, dtype=np.int64).view(SignedToken)
        shift = page.bit_length() - 1 if page == power2(page) else -1
        np.testing.assert_array_equal(page_index(values, page, shift), np.asarray(values) // page)
        total += values.size
        for lengths in ([0, 1, 7, 8], [page - 1, page, page + 1, 2 * page], [16320, 0, 16384, 1]):
            prior = np.array([0, page, 3 * page - 1, 5 * page + 1])
            lens = np.array(lengths) + prior
            count = (lens + page - 1) // page - (prior + page - 1) // page
            free = rng.permutation(np.arange(10000, 10000 + sum(count)))
            last = np.array([200 * page + (n - 1) % page if n else -1 for n in prior])
            wanted = []
            cursor = 0
            for i, n in enumerate(lengths):
                old = min(n, (-prior[i]) % page)
                wanted.extend(last[i] + np.arange(1, old + 1))
                relative = np.arange(n - old)
                wanted.extend(free[cursor + relative // page] * page + relative % page)
                cursor += count[i]
            output = np.full(sum(lengths) + 2, -34567, dtype=np.int64)
            for i in range(4):
                tl.program_id = lambda axis: i
                alloc(*map(Pointer, (prior, lens, last, free)), Pointer(output, 1), 4, page)
                alloc_cases += 1
            np.testing.assert_array_equal(output, [-34567, *wanted, -34567])
    print(json.dumps({"passed": True, "signed_page_indices": int(total),
                      "allocator_requests": alloc_cases, "page160_chunk_tokens": [16320, 16384]}))


if __name__ == "__main__":
    main()
