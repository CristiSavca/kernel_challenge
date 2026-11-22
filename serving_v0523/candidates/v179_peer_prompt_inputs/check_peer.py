"""Run the production workspace and sparse MoE path across real Gloo ranks."""

from collections import Counter
from contextlib import ExitStack, nullcontext
import ctypes
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
KERNEL = "sglang.kernels.ops.moe.muse_spark_peer_rows"
COMM = "sglang.srt.distributed.device_communicators.custom_all_reduce_v2"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load_module("v169_prompt", HERE.parent / "v169_grouped_expert_tile_cohorts/cpu_check.py")
integration = prior.v168.prior


def worker(rank, size, root, slabs):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(root) / "rendezvous"),
                            rank=rank, world_size=size, timeout=timedelta(seconds=90))
    stats = Counter()
    f = prior.fixture(stats)
    capacity, hidden = slabs.shape[1:]
    class Communication:
        disabled = False
        pull_enabled = True
        has_multicast = True
        _is_graph_mode_supported = True
        device = torch.device("cpu")
        max_pull_size = 2 * capacity * hidden

    comm = Communication()
    comm.rank, comm.world_size = rank, size
    group = SimpleNamespace(rank_in_group=rank, world_size=size, device_group=dist.group.WORLD,
                            cpu_group=dist.group.WORLD, ca_comm=comm)
    class Handle:
        def __init__(self):
            self.rank, self.world_size = rank, size
            self.offset = int("offset" in faults and rank == size - 1) * 128

        def get_remote_tensor(self, source, shape, dtype):
            assert shape[1] == hidden and dtype == torch.bfloat16
            if "alias" in faults and rank == size - 1 and source == rank:
                return slabs[(rank + 1) % size, :shape[0]]
            return slabs[source, :shape[0]]

        def barrier(self, channel):
            assert channel == 0
            dist.barrier()
            stats["peer_barriers"] += 1

    faults = set()
    def allocate(shape, strides, dtype, device):
        assert strides == (shape[1], 1)
        if "allocation" in faults and rank == size - 1:
            raise RuntimeError("injected allocation error")
        return slabs[rank, :shape[0]]

    def rendezvous(storage, group):
        assert group == dist.group.WORLD.group_name
        stats["rendezvous"] += 1
        dist.barrier()
        return Handle()

    class TritonError(Exception):
        pass

    def read_bits(addresses, live_rows, output, shard_rows):
        assert addresses.dtype == torch.int64 and live_rows.dtype == torch.int64
        if "compile" in faults and rank == 0:
            raise TritonError("injected optional kernel failure")
        assert torch.all(live_rows[1:] >= live_rows[:-1])
        for source in range(size):
            mask = live_rows // shard_rows == source
            array = (ctypes.c_int16 * (shard_rows * hidden)).from_address(int(addresses[source]))
            peer = torch.frombuffer(array, dtype=torch.int16).view(shard_rows, hidden)
            output.view(torch.int16)[mask] = peer[live_rows[mask] % shard_rows]
        stats["bit_rows_read"] += live_rows.numel()
        stats["read_calls"] += 1
        stats["empty_reads"] += int(not live_rows.numel())

    class Stream:
        def __init__(self, *args, **kwargs):
            pass
        def wait_stream(self, other):
            stats["stream_dependencies"] += 1
        def synchronize(self):
            pass

    driver = ModuleType(COMM)
    driver.CustomAllReduceV2 = Communication
    triton_errors = ModuleType("triton.errors")
    triton_errors.TritonError = TritonError
    kernel = ModuleType(KERNEL)
    kernel.read_peer_rows = read_bits
    allocator = SimpleNamespace(empty_strided_p2p=allocate, rendezvous=rendezvous)
    def namespace(**kwargs):
        if set(kwargs) == {"rank_in_group", "world_size", "device_group"}:
            return group
        return SimpleNamespace(**kwargs)

    with ExitStack() as contexts:
        add = contexts.enter_context
        add(torch.no_grad())
        add(patch.dict(sys.modules, {COMM: driver, KERNEL: kernel, "triton.errors": triton_errors}))
        add(patch.object(torch._C._distributed_c10d, "_SymmetricMemory", allocator, create=True))
        add(patch.object(torch.cuda, "Stream", Stream))
        add(patch.object(torch.cuda, "stream", lambda _: nullcontext()))
        add(patch.object(torch.cuda, "current_stream", lambda *_: Stream()))
        add(patch.object(torch.Tensor, "record_stream", lambda *_: None))
        add(patch.object(torch.Tensor, "is_cuda", property(lambda _: True)))
        add(patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)))
        add(patch.object(torch.cuda, "is_current_stream_capturing", side_effect=lambda: f.capture.get() is not None))
        add(patch.object(integration, "NS", namespace))
        stage_path = STAGE / "python/sglang/srt/layers/muse_spark/peer_prompt_inputs.py"
        source = load_module("prod_peer", stage_path)
        for fault in ("rank_topology", "width", "allocation", "compile", "offset", "alias"):
            faults.add(fault)
            comm.has_multicast = not (fault == "rank_topology" and rank == size - 1)
            width = hidden + int(fault == "width" and rank == size - 1)
            before = stats["rendezvous"]
            assert source.prepare_peer_prompt_inputs(group, width, capacity * size) is None
            assert stats["rendezvous"] == before + int(fault in ("compile", "offset", "alias"))
            faults.clear()
            stats["declined_startups"] += 1
        comm.has_multicast = True
        workspace = source.prepare_peer_prompt_inputs(group, hidden, capacity * size)
        assert workspace is not None and workspace.storage.shape == (capacity, hidden)
        f.Expert.humming_prefill_tiles = prior.state
        f.Expert.humming_prefill_silu = prior.v168.ScaledSilu()
        for mode in ("ready", "absent_on_one_rank"):
            f.Expert._peer_prompt_inputs = workspace if mode == "ready" or rank else None
            before = stats["peer_barriers"]
            for tokens in (257, 1025, capacity * size + 7):
                integration.exercise(f, rank, size, tokens, stats)
            if mode == "ready" and size >= 4:
                assert stats["peer_barriers"] > before
            if mode == "absent_on_one_rank":
                assert stats["peer_barriers"] == before
        # Exercise empty consumers and raw BF16 bit patterns on the real wrapper.
        for epoch in range(3):
            rows = capacity - epoch
            bits = (torch.arange(size * rows * hidden, dtype=torch.int32).reshape(size, rows, hidden)
                    + 32703 + epoch * 271).short()
            bits[:, :, :3] = torch.tensor([-32768, 32705, -1], dtype=torch.int16)
            live = torch.arange(rank % 3, rows * size, 5)
            if rank == (epoch % size):
                live = live[:0]
            actual = workspace.read(bits[rank].view(torch.bfloat16), live)
            assert torch.equal(actual.view(torch.int16), bits.flatten(0, 1)[live])
            stats["guarded_BF16_values"] += actual.numel()
            stats["bit_pattern_replays"] += 1
        assert stats["peer_barriers"] % 2 == 0 and stats["stream_dependencies"] == stats["peer_barriers"]
    (Path(root) / f"rank{rank}.json").write_text(json.dumps(stats))
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_num_threads(1)
    results = {}
    for size in (2, 4, 8):
        with tempfile.TemporaryDirectory() as root:
            slabs = torch.empty((size, 512, 12), dtype=torch.bfloat16).share_memory_()
            mp.start_processes(worker, (size, root, slabs), nprocs=size, start_method="fork")
            results[size] = sum((Counter(json.loads((Path(root) / f"rank{r}.json").read_text()))
                                 for r in range(size)), Counter())
    print(json.dumps(dict(physical_H200=False, torch=torch.__version__, gloo=results,
          host_substitutes=["symmetric allocation and mapped peer address spaces", "CUDA stream and NVLink barriers",
                            "Triton load executing via host pointers", "lower grouped GEMMs and CUDA finalization"]), indent=2))
