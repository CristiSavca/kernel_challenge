"""Run the production KV compactor in Triton's CPU interpreter on byte-exact host buffers."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
import numpy as np
import triton
import triton.language as tl
from triton.runtime.jit import TensorWrapper


class HostStorage:
    """Expose an already host-resident, contiguous NumPy allocation to the interpreter."""

    device = "cpu"

    def __init__(self, array):
        self.data = array
        self.shape = array.shape

    def data_ptr(self):
        return self.data.ctypes.data

    def untyped_storage(self):
        return self

    def cpu(self):
        return self

    def new_empty(self, *args, **kwargs):
        return HostStorage(np.empty(0, dtype=self.data.dtype))

    def set_(self, storage, offset, size, stride):
        assert offset == 0 and size == storage.shape and stride == storage.stride()
        self.__init__(storage.data)

    def storage_offset(self):
        return 0

    def size(self):
        return self.shape

    def stride(self):
        return tuple(s // self.data.itemsize for s in self.data.strides)

    def copy_(self, source):
        np.copyto(self.data, source.data)


def device_arg(array):
    types_by_dtype = {"int64": tl.int64, "int32": tl.int32, "uint64": tl.uint64}
    return TensorWrapper(HostStorage(array), types_by_dtype[array.dtype.name])


def load_kernel(path):
    source = path.read_bytes()
    module = ast.parse(source, filename=str(path))
    module.body = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_compact_tree_rows"]
    assert len(module.body) == 1
    ns = {"triton": triton, "tl": tl, "__name__": "kv_fixture"}
    exec(compile(module, str(path), "exec"), ns)
    return ns["_compact_tree_rows"], hashlib.sha256(source).hexdigest()


def exercise(kernel, rng, width, lengths, translate, page, retained):
    batch = len(lengths)
    rows = (2 * width + 3) * batch + 2 * page + 2
    slots = np.arange(batch * width, dtype=np.int64).reshape(batch, width)
    # A fresh page can be anywhere in the physical pool.
    n_pages = (rows + page - 1) // page
    pmap = rng.permutation(np.arange(1, n_pages + 1))
    logical = slots + (page - 1) + np.arange(batch)[:, None] * (width + 3)
    slots = (pmap[logical // page] * page + (logical % page)).astype(np.int64)
    capacity = (n_pages + 1) * page
    order = np.empty_like(slots)
    for b in range(batch):
        order[b] = np.r_[0, rng.permutation(np.arange(1, width))]
    if width >= 4:
        order[0] = np.r_[0, 2, 3, 1, np.arange(4, width)]
    order[-1] = np.arange(width)
    if batch > 2:
        slots[2] = 0
        lengths[2] = 0
    mapping = np.r_[0, rng.permutation(np.arange(1, capacity))].astype(translate or np.int64)
    dimensions = (16, 32, 64, 67)  # Includes differing K/V row widths and a partial vector tile.
    data = [rng.integers(0, 2**32, (capacity, d), dtype=np.uint32) for d in dimensions]
    originals = [a.copy() for a in data]
    ptrs = np.array([a.ctypes.data for a in data], dtype=np.uint64)
    strides = np.array([4 * d for d in dimensions], dtype=np.int64)
    kernel[(batch, len(data), 3)](
        device_arg(ptrs), device_arg(strides), device_arg(mapping) if translate else None,
        device_arg(slots), device_arg(order), device_arg(np.array(lengths, dtype=np.int32)),
        width, triton.next_power_of_2(width), 32, bool(translate), num_warps=1,
    )
    expected = [a.copy() for a in originals]
    for request, length in enumerate(lengths):
        dest = slots[request, :length]
        src = slots[request, order[request, :length]]
        if translate:
            dest, src = mapping[dest], mapping[src]
        for want, original in zip(expected, originals):
            want[dest] = original[src]
    for want, actual in zip(expected, data):
        np.testing.assert_array_equal(want, actual)
    for want, actual in retained:
        np.testing.assert_array_equal(want, actual)
    # Keep an old bank alive while the subsequent call changes the pointer array and map.
    retained[:] = list(zip(expected, data))
    return sum(x.nbytes for x in data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    kernel, sha = load_kernel(args.source)
    # Setup compiles with an empty path. Null bank addresses and a one-element
    # map are safe because none of their physical rows may be touched here.
    zero = np.zeros((1, 8), dtype=np.int64)
    order = np.arange(8, dtype=np.int64).reshape(1, 8)
    for mapping in (None, device_arg(np.zeros(1, dtype=np.int64))):
        kernel[(1, 2, 1)](
            device_arg(np.zeros(2, dtype=np.uint64)), device_arg(np.full(2, 128, dtype=np.int64)),
            mapping, device_arg(zero), device_arg(order), device_arg(np.zeros(1, dtype=np.int32)),
            8, 8, 32, mapping is not None, num_warps=1,
        )
    rng = np.random.default_rng(1971)
    retained = []
    count = total = 0
    for page in (16, 64):
        for width in range(1, 9):
            for translate in (None, np.int32, np.int64):
                lengths = [width, max(0, width - 1), 0, 1, min(4, width), width]
                total += exercise(kernel, rng, width, lengths, translate, page, retained)
                count += 1
    print(json.dumps({"source_sha256": sha, "triton": triton.__version__, "cases": count,
                      "bytes_checked": total, "page_sizes": [16, 64], "widths": list(range(1, 9)),
                      "mapping_types": ["none", "int32", "int64"], "empty_preflights": 2,
                      "passed": True}, indent=2))


if __name__ == "__main__":
    main()
