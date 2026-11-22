"""Execute the frozen Triton bodies on CPU and exercise real process-group MAX."""

import argparse
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v70_dflash_selector_tree"))
import cpu_check as fx


class Words(np.ndarray):
    def to(self, dtype, bitcast=False):
        return (self.view(dtype) if bitcast else self.astype(dtype)).view(Words)


def words(val):
    return np.asarray(val).view(Words)


def pair_max(x, axis, return_indices, return_indices_tie_break_left):
    assert axis == 0 and return_indices and return_indices_tie_break_left
    v = x
    index = np.arange(len(x), dtype=np.int32)
    while len(v) > 1:
        a, b = v[::2], v[1::2]
        ia, ib = index[::2], index[1::2]
        left = (a > b) | ((a == b) & (ia < ib))
        v, index = np.where(left, a, b), np.where(left, ia, ib)
    return words(v[0]), words(index[0])


class Launch:
    def __init__(self, fn, pids):
        self.fn, self.pids = fn, pids

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            pointers = [fx.Pointer(a.numpy()) if isinstance(a, torch.Tensor) else a for a in args]
            for ids in itertools.product(*(range(x) for x in grid)):
                self.pids[:len(ids)] = ids
                self.fn(*pointers)
        return call


def load_kernels(root):
    file = root / "python/sglang/kernels/ops/sampling/softcap_row_argmax.py"
    ids, (old, tile, unpack, reduce) = fx.program(file, (
        "_softcap_row_argmax_kernel", "_softcap_tile_keys", "_unpack_winner_ids", "reduced_softcap_argmax",
    ))
    ns = tile.__globals__
    tl = ns["tl"]
    # The CPU pairwise reducer is identical for old/new; equality of the
    # compiled PTX butterfly order is checked independently in the AOT receipt.
    tl.program_id = lambda dim: words(np.int32(ids[dim]))
    for op in ("where", "load", "arange"):
        f = getattr(tl, op)
        setattr(tl, op, lambda *a, _f=f, **kw: words(_f(*a, **kw)))
    tl.max, tl.uint32 = pair_max, np.uint32
    ns.update(libdevice=SimpleNamespace(tanh=lambda x: words(np.tanh(x))), torch=torch)
    ns["triton"].cdiv = lambda a, b: (a + b - 1) // b
    ns["_softcap_tile_keys"] = Launch(tile, ids)
    ns["_unpack_winner_ids"] = Launch(unpack, ids)
    return ids, old, ns["_softcap_tile_keys"], ns["_unpack_winner_ids"], reduce


def samples(rows, width, rank=0):
    rng = np.random.default_rng(231 + rank)
    a = rng.normal(size=(rows, width)).astype(np.float32)
    a[1] = -np.inf
    a[2] = np.nan
    a[3, 1::7] = np.nan
    a[4, ::23] = np.inf
    a[5] = -0.0
    a[5, 1::2] = 0.0
    a[6, ::13] = 1e20
    a[7] = -1e20
    return torch.from_numpy(a)


def old_pairs(pids, old, x, valid, offset, cap):
    out = np.empty((len(x), 2), np.float32)
    for row in range(len(x)):
        pids[0] = row
        old(fx.Pointer(x.numpy()), fx.Pointer(out), x.shape[1], valid, words(np.int32(offset)), cap, 4096)
    return torch.from_numpy(out)


def test_local(root):
    pids, old, tile, unpack, _ = load_kernels(root)
    cases = 0
    for width in (1, 63, 4096, 4097, 8192, 25256):
        x = samples(8, width)
        for valid in sorted({0, 1, width // 2, width - 1, width}):
            for cap in (1.0, 30.0):
                offset = 3 * width
                t = (width + 4095) // 4096
                partial = torch.empty((8, t), dtype=torch.int64)
                tile[(8, t)](x, partial, width, valid, offset, np.float32(cap), t, 4096)
                keys = partial.amax(dim=1)
                ids = torch.empty(len(x), dtype=torch.int64)
                unpack[(1,)](keys, ids, len(x), 128)
                expected = old_pairs(pids, old, x, valid, offset, np.float32(cap))
                assert torch.equal(ids, expected[:, 1].long())
                got_order = (keys >> 32).int()
                values = expected[:, 0].contiguous().view(torch.int32)
                values = torch.where(expected[:, 0] == 0, torch.zeros_like(values), values)
                assert torch.equal(got_order, values ^ ((values >> 31) & 0x7fffffff))
                cases += len(x)
    return cases


def distributed_worker(rank, world, checkout, init_file):
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method="file://" + init_file)
    try:
        pids, old, _, _, reduce = load_kernels(Path(checkout))
        for width, last_valid in ((17, 0), (4097, 1), (8192, 7191), (25256, 25256)):
            x = samples(8, width, rank)
            valid = width if rank + 1 < world else last_valid
            offset = rank * width
            expected_local = old_pairs(pids, old, x, valid, offset, np.float32(30))
            peers = [torch.empty_like(expected_local) for _ in range(world)]
            dist.all_gather(peers, expected_local)
            raw = torch.stack(peers)
            best = raw[..., 0].argmax(dim=0, keepdim=True)
            expected_ids = raw[..., 1].gather(0, best).squeeze(0).long()
            new_ids = reduce(x, np.float32(30), valid, offset, dist.group.WORLD)
            assert torch.equal(expected_ids, new_ids), (rank, width)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkout", type=Path)
    args = p.parse_args()
    local = test_local(args.checkout)
    with tempfile.TemporaryDirectory() as temp:
        mp.spawn(distributed_worker, args=(2, str(args.checkout), str(Path(temp) / "rendezvous")), nprocs=2, join=True)
    print(json.dumps(dict(local_row_cases=local, gloo_rank_rows=64, ranks=2), indent=2))
