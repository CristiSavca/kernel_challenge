"""Exercise the complete cohort exchange, raw projections and group-wide fallback."""
from collections import Counter
from contextlib import ExitStack
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load("paired_fixture", next(HERE.parent.glob("v113_*/cpu_check.py")))
projection = load("draft_context", STAGE / "python/sglang/srt/layers/dflash_context_projection.py")


def worker(rank, size, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=size,
                            init_method="file://" + str(Path(folder)/"rendezvous"), timeout=timedelta(seconds=75))
    comm = fixture.Comm(rank)
    group = SimpleNamespace(rank_in_group=rank, world_size=size, pynccl_comm=comm, cpu_group=dist.group.WORLD)
    fixture.install("sglang.srt.batch_invariant_ops", {"is_batch_invariant_mode_enabled": lambda: False})
    counts = Counter()
    shapes = ((1, 64, 10), (4, 64, 10), (8, 64, 10), (16, 64, 10), (32, 128, 20), (64, 64, 10))
    g = torch.Generator().manual_seed(rank+100)
    for rows, hidden, channels in shapes:
        for padding in (0, 1, size-1):
            n = rows * size-padding
            x = torch.randint(-4, 5, (rows, hidden), generator=g).bfloat16()
            w = torch.randint(-4, 5, (hidden, channels), generator=g).bfloat16()
            helper = SimpleNamespace(flat_kv_weight_t=w, n_layers=5, layer_out_dim=channels//5)
            volumes = {(r-1)*rows*hidden + (size//r-1)*(hidden*channels+r*rows*channels):r
                       for r in (2, 4) if r < size}
            traffic = min(volumes)
            worthwhile = 8*traffic <= 7*(size-1)*rows*hidden
            native = [torch.empty_like(x) for _ in range(size)]
            dist.all_gather(native, x)
            expected = (torch.cat(native) @ w)[:n]
            before = comm.sent
            with projection.dflash_sp_prefill_scope(None,helper):
                got = projection.project_paired_draft_kv(x, n, group)
            if worthwhile:
                assert torch.equal(got, expected) and comm.sent-before==traffic*w.element_size()
                counts['width_'+str(volumes[traffic])]+=1
                counts['bf16_elements']+=got.numel()
                counts['logical_bytes_sent']+=comm.sent-before
            else:
                assert got is None and comm.sent==before
                counts['geometry_declines']+=1
            assert comm.disabled and comm.ops is None

    # Change a single rank's configuration or inject an allocation failure.
    rows, hidden, channels = 32, 64, 10
    for test in ('disabled_helper','rank_shape','disabled_comm','input_stride','host_padding','alloc','gemm'):
        n = rows*size
        x = torch.randn(rows, hidden).bfloat16()
        w = torch.randn(hidden,channels).bfloat16()
        helper = SimpleNamespace(flat_kv_weight_t=w, n_layers=5, layer_out_dim=2)
        before=comm.sent
        with ExitStack() as overrides:
            if rank == size-1:
                if test=='disabled_helper': helper=None
                if test=='rank_shape': n-=1
                if test=='disabled_comm': overrides.enter_context(patch.object(comm,'available',False))
                if test=='input_stride': x=x.t().contiguous().t()
                if test=='host_padding': n-=size
                if test=='alloc': overrides.enter_context(patch.object(torch.Tensor,'new_empty',side_effect=torch.OutOfMemoryError()))
                if test=='gemm': overrides.enter_context(patch.object(torch,'mm',side_effect=torch.OutOfMemoryError()))
            overrides.enter_context(projection.dflash_sp_prefill_scope(None,helper))
            assert projection.project_paired_draft_kv(x,n,group) is None
        assert (comm.sent > before) == (test=='gemm') and comm.disabled and comm.ops is None
        counts['asymmetric_declines']+=1

    # A declined round must leave the communicator usable.
    with projection.dflash_sp_prefill_scope(None,SimpleNamespace(flat_kv_weight_t=w,n_layers=5,layer_out_dim=2)):
        assert projection.project_paired_draft_kv(x,n,group) is not None
    all_counts=[None]*size
    dist.all_gather_object(all_counts,dict(counts))
    if rank==0:
        summed=Counter()
        for count in all_counts: summed.update(count)
        (Path(folder)/"result.json").write_text(json.dumps(summed))
    dist.destroy_process_group()


if __name__ == "__main__":
    results = {}
    for size in (4,8):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker,args=(size,folder),nprocs=size,join=True)
            results[size]=json.loads((Path(folder)/"result.json").read_bytes())
    print(json.dumps({"gloo":results,"torch":torch.__version__,"physical_h200":False},indent=2))
