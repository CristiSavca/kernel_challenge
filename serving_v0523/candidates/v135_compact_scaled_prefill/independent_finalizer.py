"""Compare the two production route finalizers using BF16 multiplication from PyTorch."""
import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

KERNEL = "_post_expert_norm_weighted_sum_kernel"
SOURCE = "python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py"


def unpack(checkout, patch, root):
    path = root / SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(subprocess.check_output(["git", "-C", str(checkout), "show", "297e54fd:" + SOURCE]))
    subprocess.run(["git", "apply", "--include=" + SOURCE, str(patch)], cwd=root, check=True)
    return path


def evaluator(source, shim, pid):
    node = next(n for n in ast.parse(source.read_text()).body if getattr(n, "name", None) == KERNEL)
    node.decorator_list.clear()
    scope = {"tl": SimpleNamespace(
        constexpr=object(), program_id=lambda axis: pid[axis], load=shim.load, store=shim.put,
        arange=torch.arange, static_range=range, float32=torch.float32, int64=torch.int64,
        zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype), sum=torch.sum,
        rsqrt=torch.rsqrt, where=torch.where,
    )}
    exec(compile(ast.fix_missing_locations(ast.Module([node], [])), str(source), "exec"), scope)
    return scope[KERNEL]


def compare(parent, candidate, shim):
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(4132)
    pid = [0]
    before, after = (evaluator(p, shim, pid) for p in (parent, candidate))
    ptr = shim.Pointer
    totals = dict(cases=0, compared_bf16_values=0, compared_live_routes=0, remote_slot_visits=0,
                  shuffled_case_count=0, raw_multiply_rounding_changes=0, poison_padding_rows=0)
    for width, batch, topk in [(31, 7, 3), (64, 13, 4), (95, 8, 8), (128, 64, 8), (4096, 9, 8)]:
        block = 1 << (width - 1).bit_length()
        for mode in ["local", "mixed", "one_expert", "remote", "one_slot"]:
            ids = torch.randint(0, 11, (batch, topk), generator=rng, dtype=torch.int32)
            if mode == "mixed":
                ids.flatten()[1::3] = -1
            elif mode == "remote":
                ids.fill_(-1)
            elif mode == "one_expert":
                ids.fill_(4)
            elif mode == "one_slot":
                ids.fill_(-1)
                ids[-1, -1] = 6
            alive = torch.nonzero(ids.flatten() >= 0, as_tuple=True)[0]
            count = len(alive)
            # The producer's compact order can change without changing slot ownership.
            order = alive[torch.randperm(count, generator=rng)]
            if not torch.equal(order, alive):
                totals["shuffled_case_count"] += 1
            rows = torch.full_like(ids, -1)
            rows.flatten()[order] = torch.arange(count, dtype=torch.int32)
            scales = torch.tensor([0, 0.125, 0.703125, 1, 1.0078125, 1.046875, 1.875, 2, 3.125, 4.375, 10.125],
                                  dtype=torch.bfloat16)
            raw = torch.randn(count + 3, width, generator=rng).bfloat16()
            raw[count:] = float("nan")
            raw[:, ::19] = -0.0
            raw[count:] = float("nan")
            expanded = torch.full_like(raw, float("nan"))
            if count:
                mult = scales[ids.flatten()[order].long(), None]
                expanded[:count] = raw[:count] * mult
                full_precision = raw[:count].float() * mult.float()
                totals["raw_multiply_rounding_changes"] += int((expanded[:count].float() != full_precision).sum())
            scores = torch.randn(batch, topk, generator=rng, dtype=torch.float32)
            scores[ids < 0] = float("nan")
            w = torch.randn(width, generator=rng).bfloat16()
            for norm, has_w, weight_first, round_product in itertools.product((False, True), repeat=4):
                if (not norm and (has_w or weight_first)) or (not has_w and weight_first):
                    continue
                opts = dict(TOPK=topk, H=width, eps=1e-8, APPLY_NORM=norm, HAS_WEIGHT=has_w,
                            WEIGHT_FIRST=weight_first, HAS_IDS=False, HAS_ROW_MAP=True,
                            ROUND_PRODUCT=round_product, BLOCK_H=block)
                a, b = (torch.full((batch + 2, width), float("nan"), dtype=torch.bfloat16) for _ in range(2))
                for tok in range(batch):
                    pid[0] = tok
                    before(ptr(expanded), ptr(w), ptr(scores), ptr(a[1:-1]), ptr(rows), **opts)
                    after(ptr(raw), ptr(w), ptr(scores), ptr(b[1:-1]), ptr(rows), ptr(ids), ptr(scales),
                          HAS_EXPERT_SCALE=True, **opts)
                assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (mode, width, opts)
                assert torch.isfinite(b[1:-1]).all() and torch.isnan(b[[0, -1]]).all()
                totals["cases"] += 1
                totals["compared_bf16_values"] += batch * width
                totals["compared_live_routes"] += count
                totals["remote_slot_visits"] += ids.numel() - count
                totals["poison_padding_rows"] += 3
    assert totals["raw_multiply_rounding_changes"] > 1000
    return totals


def run(candidates, candidate):
    sys_shim = candidates / 'v104_v64_direct_draft_kv/bytes_cpu.py'
    spec = importlib.util.spec_from_file_location('v104_cpu_pointers', sys_shim)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch)
        checkout = candidates.parent / 'sglang'
        old = unpack(checkout, candidates / 'v64_humming_grouped_sequence_parallel/patch.diff', path / 'parent')
        new = unpack(checkout, candidate, path / 'candidate')
        result = compare(old, new, shim)
        return dict(torch=torch.__version__, parent_body_file_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
                    candidate_body_file_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
                    candidate_patch_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), h200=False, **result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_patch', type=Path)
    parser.add_argument('--candidates', type=Path)
    args = parser.parse_args()
    candidate = args.candidate_patch.resolve()
    print(json.dumps(run((args.candidates or candidate.parent.parent).resolve(), candidate), indent=2))
