"""Exercise the frozen rank-major draft filter on CPU and Gloo without a model download."""

import argparse
import ast
import hashlib
import json
import math
import re
from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F


def production(stage, group):
    srt = Path(stage) / "python/sglang/srt"
    ns = dict(torch=torch, F=F, nn=nn, set_weight_attrs=lambda *args: None,
              sharded_weight_loader=lambda *args, **kw: None,
              get_parallel=lambda: SimpleNamespace(tp_group=group))
    for path, names in [
        ("layers/dflash_context_projection.py", {"DFlashGatheredLinear"}),
        ("models/dflash.py", {"_grouped_conv", "_rank_major_grouped_conv", "DFlashGroupedConv"}),
    ]:
        model = ast.parse((srt / path).read_text())
        model.body = [node for node in model.body if getattr(node, "name", None) in names]
        assert len(model.body) == len(names), names
        for node in model.body:
            node.decorator_list = []
        exec(compile(model, str(srt / path), "exec"), ns)
    return ns


class Group:
    def __init__(self, world, rank, distributed):
        self.world_size, self.rank, self.distributed = world, rank, distributed
        self.calls = []

    def all_gather(self, local, dim):
        if self.distributed:
            # Gloo's worker thread writes outside the caller's inference context.
            with torch.inference_mode(False):
                chunks = [torch.empty_like(local) for _ in range(self.world_size)]
                dist.all_gather(chunks, local.clone())
        else:
            chunks = [local + i / 8 for i in range(self.world_size)]
        canonical_dim = dim % local.dim()
        self.calls.append(canonical_dim)
        # Follow pinned GroupCoordinator's concat-shaped transport and last-dim
        # restoration exactly, including the non-view reshape for ordinary M>1.
        self.raw = torch.cat(chunks, dim=0).view(self.world_size, *local.shape)
        return (self.raw.movedim(0, canonical_dim)
                .reshape(*local.shape[:canonical_dim], self.world_size * local.shape[canonical_dim],
                         *local.shape[canonical_dim+1:]))


def same_bits(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype, (x.shape, y.shape)
    integer = torch.int32 if x.dtype == torch.float32 else torch.int16
    assert torch.equal(x.contiguous().view(integer), y.contiguous().view(integer))


def one(stage, group, rows, groups, width, taps, block, dtype, scenario):
    ns = production(stage, group)
    conv = ns["DFlashGroupedConv"](groups*width, block, taps, width, shard_filter=True).to(dtype)
    torch.manual_seed(19 + group.rank)
    conv.base_kernel.normal_()
    conv.kernel_projection.weight.normal_(std=.03)
    x = torch.randn(rows, groups*width*2, dtype=dtype)[:, ::2]
    tail = torch.randn_like(x)
    hooks = []
    locally_hooked = scenario == "asymmetric" and group.rank == 0
    if locally_hooked:
        hooks.append(conv.kernel_projection.register_forward_hook(lambda module, args, result: result + .25))
    elif scenario == "global" and group.rank == 0:
        hooks.append(nn.modules.module.register_module_forward_pre_hook(lambda *args: None))
        locally_hooked = True
    elif scenario == "override" and group.rank == 0:
        original = conv.kernel_projection.forward
        conv.kernel_projection.forward = lambda inputs: original(inputs)
        locally_hooked = True
    eligible = ((group.world_size % (2*taps) == 0 or (2*taps) % group.world_size == 0)
                and not locally_hooked)
    before = len(group.calls)
    try:
        first, second = conv.prepare(x)
        last = conv.finish(tail, second)
    finally:
        for hook in hooks:
            hook.remove()
    assert group.calls[before:] == [0 if eligible else 1]
    bank = group.raw
    if eligible:
        assert second[0].data_ptr() == bank.data_ptr()
    canonical = bank.movedim(0, 1).reshape(rows, -1)
    if locally_hooked and scenario == "asymmetric":
        canonical = canonical + .25
    coefficient = canonical.view(rows, 2, taps, groups)
    for side, hidden, result in [(0, x, first), (1, tail, last)]:
        ref = ns["_grouped_conv"](hidden, coefficient[:, side], conv.base_kernel[side],
                                  block, groups, width, taps)
        same_bits(result, ref)
    return int(eligible), 2*rows*groups*width


def gloo_worker(rank, world, filename, work, stage):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+filename, rank=rank,
                            world_size=world, timeout=timedelta(seconds=45))
    group = Group(world, rank, True)
    routes, scalars = [], 0
    with torch.inference_mode():
        for taps in (1, 2, 3, 4):
            for mode in ("ordinary", "asymmetric", "global", "override"):
                ok, count = one(stage, group, 11, 24, 3, taps, 4, torch.bfloat16, mode)
                routes.append(ok)
                scalars += count
    (Path(work)/f"{world}_{rank}.json").write_text(json.dumps({
        "rank": rank, "rank_major": sum(routes), "ordinary": len(routes)-sum(routes),
        "output_values": scalars, "collectives": len(group.calls)}))
    dist.destroy_process_group()


def geometry(stage):
    cases = scalars = rank_major = 0
    with torch.inference_mode():
        for world in (2, 3, 4, 5, 8):
            group = Group(world, 0, False)
            for taps in (1, 2, 3, 4):
                for block in (3, 8):
                    for rows in (max(1, taps-1), block, 2*block+1):
                        for dtype in (torch.bfloat16, torch.float16, torch.float32):
                            ok, count = one(stage, group, rows, 60, 2, taps, block, dtype, "ordinary")
                            cases += 1
                            rank_major += ok
                            scalars += count
        for rows in (8, 32, 128, 256):
            # Hold the real 8192 channels locally while the stub projects just
            # the 256 checkpoint channels owned by one tensor-parallel rank.
            ok, count = one(stage, Group(8, 0, False), rows, 512, 16, 2, 8, torch.bfloat16, "ordinary")
            assert ok
            cases += 1
            rank_major += ok
            scalars += count
        group = Group(8, 0, False)
        cls = production(stage, group)["DFlashGatheredLinear"]
        mod = cls(32, 64, group)
        assert mod(torch.empty(0,32)).shape == (0,64)
        assert mod(torch.empty(0,32), rank_major=True).shape == (8,0,8)
        assert not group.calls
    return {"layouts": cases, "rank_major_layouts": rank_major, "ordinary_layouts": cases-rank_major,
            "output_values_compared_by_raw_bits": scalars, "zero_row_collectives": 0}


def codegen(stage):
    from torch._inductor.utils import run_and_get_code
    ns = production(stage, None)
    bank = torch.randn(8, 8, 256).to(torch.bfloat16)
    base = torch.randn(2, 8192).to(torch.bfloat16)
    hidden = torch.randn(8, 8192).to(torch.bfloat16)
    coefficient = bank.movedim(0, 1).reshape(8, 2, 2, 512)
    results = []
    for side in (0, 1):
        ref, old_code = run_and_get_code(torch.compile(ns["_grouped_conv"], fullgraph=True),
                hidden, coefficient[:, side], base, 8, 512, 16, 2)
        out, new_code = run_and_get_code(torch.compile(ns["_rank_major_grouped_conv"], fullgraph=True),
                hidden, bank, base, 8, 512, 16, 2, side)
        same_bits(ref, out)
        src = "\n".join(new_code)
        allocations = re.findall(r"= empty_strided_cpu\((\([\d, ]+\))", src)
        assert len(allocations) == 1 and math.prod(ast.literal_eval(allocations[0])) == 8*8192
        results.append({"side": side, "old_source_sha256": hashlib.sha256("".join(old_code).encode()).hexdigest(),
                        "new_source_sha256": hashlib.sha256(src.encode()).hexdigest(),
                        "allocated_output_tensors": 1})
    return {"backend": "Inductor CPU", "exact": True, "compiled": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("mode", choices=("geometry","gloo","codegen"))
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.mode == "geometry":
        data = geometry(args.stage)
    elif args.mode == "gloo":
        data = {}
        with tempfile.TemporaryDirectory(prefix="v123-gloo-") as tmp:
            for world in (2,4,8):
                init = str(Path(tmp)/f"init{world}")
                mp.start_processes(gloo_worker, (world,init,tmp,args.stage), world, start_method="spawn")
                data[world] = [json.loads((Path(tmp)/f"{world}_{r}.json").read_text()) for r in range(world)]
    else:
        data = codegen(args.stage)
    print(json.dumps({"torch": torch.__version__, "mode": args.mode, "result": data}, indent=2))
