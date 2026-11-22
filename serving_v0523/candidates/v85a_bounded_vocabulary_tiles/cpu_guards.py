"""Exercise the fused vocabulary projection's CUDA launch and address guards."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v70_dflash_selector_tree"))
import cpu_check as fx


class Tensor:
    def __init__(self, rows, width, stride):
        self.shape = (rows, width)
        self.ndim = 2
        self.dtype = "bf16"
        self.device = "cuda:0"
        self.is_cuda = True
        self._strides = (stride, 1)

    def stride(self, dim):
        return self._strides[dim]


class Launcher:
    def __getitem__(self, grid):
        assert grid[0] > 0 and (len(grid) == 1 or 0 < grid[1] <= 65535)
        return lambda *args, **kwargs: None


def run(checkout):
    torch = SimpleNamespace(bfloat16="bf16", int32="int32", empty=lambda shape, **kw: shape,
                            cuda=SimpleNamespace(get_device_capability=lambda dev: (9, 0)),
                            are_deterministic_algorithms_enabled=lambda: False)
    triton = SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d,
                             next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    path = checkout / "python/sglang/kernels/ops/speculative/dflash_candidate_head.py"
    _, (launch,) = fx.program(path, ("fused_local_candidate_pairs",), dependencies={
        "torch": torch, "triton": triton, "_project_tiles": Launcher(), "_merge_tiles": Launcher(),
    })
    limit = 2**31 - 1
    trials = 0
    for rows, vocab, width, hs, ws, ok in (
        (1, 1024, 64, 64, 64, True), (1, 65536, 64, 64, 64, True),
        (2, 1024, 64, limit - 64, 64, True), (2, 1024, 64, limit - 63, 64, False),
        (2, 1024, 64, 64, (limit - 64) // 1023, True),
        (2, 1024, 64, 64, (limit - 64) // 1023 + 1, False),
        (16 * 65535, 1024, 64, 64, 64, True), (16 * 65535 + 1, 1024, 64, 64, 64, False),
        (limit // 8192, 65536, 64, 64, 64, True), (limit // 8192 + 1, 65536, 64, 64, 64, False),
        (7, 1023, 64, 64, 64, False), (7, 65537, 64, 64, 64, False),
        (7, 1024, 65, 65, 65, False), (0, 1024, 64, 64, 64, False),
    ):
        result = launch(Tensor(rows, width, hs), Tensor(vocab, width, ws), vocab, 0, 16)
        assert (result is not None) == ok, (rows, vocab, width, hs, ws, ok)
        trials += 1
    return trials


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    print(f"{run(parser.parse_args().checkout)} launch and address boundary cases")
