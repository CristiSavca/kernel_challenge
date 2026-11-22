"""Test the candidate router, rank-dependent hooks and bit-preserving gather on CPU."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v70_dflash_selector_tree"))
import cpu_check as tree


def load_defs(path, names, namespace):
    module = ast.parse(path.read_text())
    module.body = [node for node in module.body if getattr(node, "name", None) in names]
    assert len(module.body) == len(names)
    exec(compile(module, str(path), "exec"), namespace)
    return [namespace[name] for name in names]


class CpuTorch:
    """Emulate CUDA BF16 mm(out_dtype=float32) with exact BF16 operand widening."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def mm(self, a, b, *, out_dtype):
        assert out_dtype is torch.float32 and a.dtype is b.dtype is torch.bfloat16
        return torch.mm(a.float(), b.float())


class Words(np.ndarray):
    def to(self, dtype, bitcast=False):
        # Triton values are SSA. A Python in-place operator on the cast result
        # must leave the original loaded word available for the router store.
        return self.view(dtype).copy() if bitcast else self.astype(dtype)


class CpuKernel:
    def __init__(self, body, index):
        self.body = body
        self.index = index

    def __getitem__(self, grid):
        def run(gathered, pooled, logits, *args):
            pointers = [tree.Pointer(x.numpy()) for x in (gathered, pooled, logits)]
            for ijk in product(*(range(n) for n in grid)):
                self.index[:] = ijk
                self.body(*pointers, *args)
        return run


def test_transport(path):
    index = [0, 0, 0]
    tl = SimpleNamespace(
        int32=np.int32, uint32=np.uint32, constexpr=int, arange=np.arange,
        program_id=index.__getitem__, store=tree.store,
        where=lambda *args: np.where(*args).view(Words),
        load=lambda *args, **kw: tree.load(*args, **kw).view(Words),
    )
    triton = SimpleNamespace(jit=lambda fn: fn, cdiv=lambda a, b: (a + b - 1) // b)
    env = {"torch": torch, "triton": triton, "tl": tl}
    kernel, gather = load_defs(path, ["_unpack", "gather_pooler_and_router"], env)
    env["_unpack"] = CpuKernel(kernel, index)
    checked = 0
    for world, rows, columns in product((4, 8), (1, 8, 32), (32, 512)):
        experts = 256 // world
        # Use raw arbitrary words so NaNs, subnormals and exact signed zeroes survive the check.
        words = torch.randint(-(2**31), 2**31, (world, rows, columns // 2 + experts), dtype=torch.int32)
        words[..., ::11] = 0
        words[..., 1::13] = -2147483648
        words[..., 2::17] = 0x7FC00001
        def group_gather(local, dim):
            assert dim == 0 and torch.equal(local, words[0])
            return words.flatten(0, 1)
        pool, router = gather(
            SimpleNamespace(world_size=world, all_gather=group_gather),
            words[0, :, :columns // 2].view(torch.bfloat16),
            words[0, :, columns // 2:].view(torch.float32),
        )
        ref = words.transpose(0, 1)
        pool_bits = ref[:, :, :columns // 2].reshape(rows, -1).view(torch.int16)
        expected = torch.where(pool_bits == 0, torch.full_like(pool_bits, -32768), pool_bits)
        assert torch.equal(pool.view(torch.int16), expected)
        assert torch.equal(router.view(torch.int32), ref[:, :, columns // 2:].reshape(rows, -1))
        checked += words.numel()
    return checked


def test_router(model):
    env = {"torch": CpuTorch(), "nn": torch.nn, "Optional": Optional,
           "F": torch.nn.functional, "router_gemv_cute_eligible": lambda x, w: False}
    Router, = load_defs(model, ["MuseSpark1xRouter"], env)
    cases, worst = 0, 0
    for world, rows, hidden_size in product((4, 8), (1, 8, 16, 128), (128, 8192)):
        gate = Router(hidden_size, 256, torch.float32)
        gate.weight.normal_(std=hidden_size**-0.5)
        x = torch.randn(rows, hidden_size, dtype=torch.bfloat16)
        full = x.float()
        split = rows > 8
        with torch.inference_mode():
            expected = gate.forward_split2(x) if split else gate(full)
            shards = []
            for rank in range(world):
                hook = gate.register_forward_hook(lambda module, args, out: out + 1) if rank == 0 else None
                try:
                    shard, local = gate.shard_for_gather(x, full, split, rank, world)
                    if hook is not None:
                        # The original split2 call bypasses Module.__call__.
                        torch.testing.assert_close(local, expected if split else expected + 1)
                    else:
                        assert local is None
                    shards.append(shard)
                finally:
                    if hook is not None:
                        hook.remove()
            combined = torch.cat(shards, dim=-1)
            worst = max(worst, float((combined - expected).abs().max()))
            torch.testing.assert_close(combined, expected, atol=2e-5, rtol=2e-5)
            assert torch.equal(combined.argmax(-1), expected.argmax(-1))
            cases += 1
    return cases, worst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    root = args.stage / "python/sglang"
    torch.manual_seed(8301)
    torch.set_num_threads(1)
    count = test_transport(root / "kernels/ops/communication/muse_spark_router_gather.py")
    cases, worst = test_router(root / "srt/models/muse_spark.py")
    print(json.dumps({"raw_words": count, "router_cases": cases, "worst_abs_fp32": worst,
                      "torch": torch.__version__, "device": "cpu"}))
