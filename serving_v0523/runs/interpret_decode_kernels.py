"""Run with TRITON_INTERPRET=1, Triton 3.8 and NumPy on an applied checkout."""

import itertools
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
from triton.runtime.jit import TensorWrapper
from compile_decode_sm90 import load_jit_module


class HostBuffer:
    """Expose contiguous NumPy storage through the interpreter's CPU tensor API."""

    device = "cpu"

    def __init__(self, array):
        self.array = np.ascontiguousarray(array)
        self.data = self.array
        self.shape = self.array.shape

    def data_ptr(self):
        return self.array.ctypes.data

    def untyped_storage(self):
        return self

    def cpu(self):
        return self

    def new_empty(self, size, device):
        return HostBuffer(np.empty(size, self.array.dtype))

    def set_(self, storage, offset, size, stride):
        assert offset == 0 and storage.shape == size and storage.stride() == stride
        self.array = storage.array
        self.shape = size

    def size(self):
        return self.shape

    def stride(self):
        return tuple(s // self.array.itemsize for s in self.array.strides)

    def storage_offset(self):
        return 0

    def copy_(self, other):
        np.copyto(self.array, other.array)


def buf(array, dtype):
    return TensorWrapper(HostBuffer(array), dtype)


def test_head(module, rng):
    cases = 0
    for m, n, k in itertools.product((1, 17, 28), (1024, 1043, 1089), (4, 16, 32)):
        h = 128
        # Exact BF16 integers keep rank differences observable without MMA order noise.
        x = rng.integers(-3, 4, size=(m, h)).astype("f4")
        w = rng.integers(-3, 4, size=(n, h)).astype("f4")
        # Triton 3.8.0 CPU interprets BF16 dot operands as raw uint16.
        # These small integers are exact in both FP16 and BF16. Use FP16
        # inputs here; the candidate still produces BF16 keys and FP32 scores.
        b = (n + 127) // 128
        partial = np.full((m, b, k), -1, "i4")
        output = np.zeros((m, k, 2), "i4")
        module._project_tiles[(b, (m + 15) // 16)](
            buf(x.astype("f2"), tl.float16),
            buf(w.astype("f2"), tl.float16),
            buf(partial, tl.int32),
            m,
            n,
            h,
            h,
            h,
            k,
            16,
            128,
            64,
        )
        module._merge_tiles[(m,)](
            buf(partial, tl.int32),
            buf(output, tl.int32),
            15,
            b * k,
            k,
            triton.next_power_of_2(b * k),
        )
        scores = x @ w.T
        expected = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        assert np.abs(scores).max() < 256  # Each integer score fits BF16 exactly.
        np.testing.assert_array_equal(output[:, :, 1] - 15, expected)
        np.testing.assert_array_equal(
            output[:, :, 0].copy().view("f4"),
            np.take_along_axis(scores, expected, axis=1),
        )
        cases += 1
    return cases


def test_commit(module, rng):
    cases = 0
    for b, w in itertools.product((1, 4, 9), (8, 12)):
        for repeat in range(20):
            body = rng.integers(0, 8, size=(b, w), dtype="i8")
            parents = np.full((b, w), -1, dtype="i8")
            for t in range(1, w):
                parents[:, t] = rng.integers(0, t, size=b)
            pred = rng.integers(0, 8, size=(b, w), dtype="i8")
            prefix = rng.integers(0, 130000, size=b, dtype="i4")
            commit = np.empty(b, "i4")
            bonus = np.empty(b, "i8")
            toks = np.full((b, w), -1, "i4")
            lens = np.empty(b, "i4")
            order = np.empty((b, w), "i8")
            args = (body, parents, pred, prefix, commit, bonus, toks, lens, order)
            module._accept_tree[(b,)](
                *(buf(a, tl.int32 if a.dtype == np.int32 else tl.int64) for a in args),
                w,
                triton.next_power_of_2(w),
            )
            for i in range(b):
                path = [0]
                while True:
                    kids = np.flatnonzero(
                        (parents[i] == path[-1]) & (body[i] == pred[i, path[-1]])
                    )
                    if not len(kids):
                        break
                    path.append(kids[0])
                np.testing.assert_array_equal(
                    order[i], path + [j for j in range(w) if j not in path]
                )
                np.testing.assert_array_equal(
                    toks[i], list(pred[i, path]) + [0] * (w - len(path))
                )
                assert (
                    lens[i] == prefix[i] + len(path)
                    and commit[i] == len(path)
                    and bonus[i] == pred[i, path[-1]]
                )
            cases += 1
    return cases


def run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(sys.argv[1]) / "python/sglang/kernels/ops/speculative"
        rng = np.random.default_rng(87123)
        tree = load_jit_module(root / "dflash_tree.py", Path(tmp))
        head = load_jit_module(root / "dflash_candidate_head.py", Path(tmp))
        print(
            json.dumps(
                {
                    "commit_cases": test_commit(tree, rng),
                    "integer_fp16_head_cases": test_head(head, rng),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    run()
