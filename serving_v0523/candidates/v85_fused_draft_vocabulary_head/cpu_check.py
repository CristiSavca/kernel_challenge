"""Execute the candidate projection and selection kernel bodies on CPU."""

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v70_dflash_selector_tree"))
import cpu_check as fixture


class Word(np.ndarray):
    def to(self, dtype, bitcast=False):
        if dtype == "bf16":
            if bitcast:
                return self.view(BFloat16)
            return round_bf16(self).view(BFloat16)
        if bitcast:
            return self.view(dtype).view(Word)
        return self.astype(dtype).view(Word)


class BFloat16(Word):
    def to(self, dtype, bitcast=False):
        if bitcast:
            return self.view(dtype).view(Word)
        return (self.astype(np.uint32) << 16).view(np.float32).astype(dtype).view(Word)


def round_bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = ((bits + np.uint32(32767) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(x), np.uint16(0x7fc0), rounded)


def execute(source):
    idx, (pack, project, merge) = fixture.program(source, ("_ordered_keys", "_project_tiles", "_merge_tiles"))
    tl = project.__globals__["tl"]
    tl.uint16, tl.uint32, tl.bfloat16 = np.uint16, np.uint32, "bf16"

    def wrapped(f):
        return lambda *a, **k: np.asarray(f(*a, **k)).view(Word)
    for name in ("arange", "full", "where", "load", "dot"):
        setattr(tl, name, wrapped(getattr(tl, name)))
    tl.trans = lambda arr: arr.T
    tl.topk = wrapped(lambda arr, k, dim=-1: np.flip(np.sort(arr, axis=dim), axis=dim)[..., :k])
    tl.sort = wrapped(lambda arr, descending: np.sort(arr)[::-1] if descending else np.sort(arr))

    # All 65,536 BF16 encodings, including signed zeros, infinities and NaNs.
    bits = np.arange(65536, dtype=np.uint16)
    floating = (bits.astype(np.uint32) << 16).view(np.float32)
    keys = pack(floating.view(Word), bits.view(Word))
    canonical = np.where(np.isnan(floating), 0x7fc0, bits)
    ordered = np.where(canonical & 0x8000, 0xffff - canonical, 0x8000 + canonical)
    expected = ((ordered.astype(np.uint32) << 16) + (65535 - bits))
    np.testing.assert_array_equal(keys, expected)

    cases = 0
    rng = np.random.default_rng(85)
    for rows, n, h, k in itertools.product((1, 17), (128, 257, 1025), (64, 128), (4, 8, 16, 32)):
        # Exactly representable BF16 operands and FP32 sums isolate the address
        # and ordering contract from vendor GEMM's accumulation order.
        hidden = np.zeros((rows, h + 32), np.float32)
        weight = np.zeros((n, h + 16), np.float32)
        hidden[:, :h] = rng.integers(-16, 17, (rows, h))
        weight[:, :h] = rng.integers(-16, 17, (n, h))
        weight[::17] = weight[0].copy()
        tiles = (n + 127) // 128
        partial = np.zeros((rows, tiles, k), np.int32)
        out = np.zeros((rows, k, 2), np.int32)
        offset = 23456
        for tile, group in itertools.product(range(tiles), range((rows + 15) // 16)):
            idx[:2] = tile, group
            project(*map(fixture.Pointer, (hidden, weight, partial)), rows, n, h,
                    hidden.shape[1], weight.shape[1], k, 16, 128, 64)
        for row in range(rows):
            idx[0] = row
            merge(fixture.Pointer(partial), fixture.Pointer(out),
                  offset, tiles * k, k, 1 << (tiles * k - 1).bit_length())
        ref_bits = round_bf16(hidden[:, :h] @ weight[:, :h].T)
        ref = (ref_bits.astype(np.uint32) << 16).view(np.float32)
        ids = np.argsort(-ref, axis=1, kind="stable")[:, :k]
        np.testing.assert_array_equal(out[..., 1], ids + offset)
        np.testing.assert_array_equal(out[..., 0].view(np.float32), np.take_along_axis(ref, ids, axis=1))
        cases += 1
    return {"bf16_encodings": 65536, "tiled_projections": cases}


def transport(checkout):
    import torch

    model = checkout / "python/sglang/srt/models/dflash.py"
    hidden = torch.ones((5, 3), dtype=torch.bfloat16)
    cases = 0
    for world, k in itertools.product((1, 2, 4, 8), (4, 8, 16, 32)):
        heads, scores, local_pairs = [], [], []
        for rank in range(world):
            n = k + 11 - (rank % 4)
            offset = rank * (k + 11)
            # BF16 ties are valid proposal ties. Check exact values and ID
            # membership for each returned score even when ranks tie.
            vals = torch.arange(n, dtype=torch.float32).repeat(5, 1)
            vals = ((vals + 5 * rank) % (k + 17) - 9).to(torch.bfloat16)
            vals[1].zero_()
            w = torch.zeros((n, 3), dtype=torch.bfloat16)
            heads.append(SimpleNamespace(
                org_vocab_size=world * (k + 11), weight=w, quant_method=None,
                shard_indices=SimpleNamespace(num_org_elements=n, org_vocab_start_index=offset),
            ))
            scores.append(vals)
            v, i = torch.topk(vals, k)
            local_pairs.append(torch.stack((v.float().view(torch.int32), i.int() + offset), dim=-1).flatten(1))
        for rank in range(world):
            head = heads[rank]
            calls = []
            def gather(pairs, dim):
                assert dim == -1 and torch.equal(pairs, local_pairs[rank])
                calls.append(pairs.shape)
                return torch.cat(local_pairs, dim=-1)
            def fused(h, w, num_org, offset, top):
                assert w is head.weight and top == k
                return local_pairs[rank].clone() if rank % 2 == 0 else None
            modname = "sglang.kernels.ops.speculative.dflash_candidate_head"
            fake = sys.modules[modname] = ModuleType(modname)
            fake.fused_local_candidate_pairs = fused
            env = {
                "torch": torch, "is_cuda": lambda: True,
                "is_dense_head_weight": lambda w: True,
                "should_apply_lm_head_quant_method": lambda head, method: False,
                "get_parallel": lambda: SimpleNamespace(tp_size=world),
                "tensor_model_parallel_all_gather": gather,
                "_project_candidate_logits": lambda *args, **kw: scores[rank],
                "_radix_topk": lambda vals, k: torch.topk(vals, k),
            }
            _, (candidate_topk,) = fixture.program(model, ("candidate_topk",), dependencies=env)
            ids, vals, lse = candidate_topk(hidden, head, k, fuse_candidate_head=True)
            assert len(calls) == (world > 1) and lse is None
            ref = torch.cat(scores, dim=-1)
            expected = torch.topk(ref, k).values.float()
            assert torch.equal(vals, expected)
            for row in range(vals.shape[0]):
                assert ids[row].unique().numel() == k
                for slot, token_id in enumerate(ids[row].tolist()):
                    owner, local = divmod(token_id, k + 11)
                    assert vals[row, slot] == scores[owner][row, local]
            cases += 1
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--torch", action="store_true", help="also run PyTorch TP transport")
    args = parser.parse_args()
    root = args.checkout
    source = root / "python/sglang/kernels/ops/speculative/dflash_candidate_head.py"
    result = execute(source)
    if args.torch:
        result["tp_rank_cases"] = transport(root)
    print(json.dumps(result, indent=2))
