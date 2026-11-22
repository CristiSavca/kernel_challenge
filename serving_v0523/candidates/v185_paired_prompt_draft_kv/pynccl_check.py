"""Run the transport test through the pinned PyNCCL methods using real raw host pointers."""
import ast
from contextlib import contextmanager
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import check_transport as original

SOURCE = "python/sglang/srt/distributed/device_communicators/pynccl.py"
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
native = subprocess.check_output(["git", "-C", str(original.STAGE), "show", PIN + ":" + SOURCE])
owner = next(n for n in ast.parse(native).body if getattr(n, "name", None) == "PyNcclCommunicator")
methods = {"send", "recv", "change_state", "group_start", "group_end", "_resolve_stream"}
parts = [n for n in owner.body if getattr(n, "name", None) in methods]
assert len(parts) == len(methods)


class Comm:
    def __init__(self, rank):
        self.rank = self.comm = rank
        self.device = torch.device("cpu")
        self.nccl = self
        self.disabled = self.available = True
        self.ops = None
        self.sent = self.groups = self.messages = 0

    def ncclGroupStart(self):
        assert self.ops is None and not self.disabled
        self.ops = []
        self.groups += 1

    def command(self, send, pointer, count, dtype, peer, owner, stream):
        assert owner == self.rank and peer != owner and 0 <= peer < dist.get_world_size()
        assert dtype == torch.bfloat16 and stream == 123 and self.ops is not None
        raw = (ctypes.c_uint16 * count).from_address(pointer)
        view = torch.frombuffer(raw, dtype=torch.int16)
        assert view.data_ptr() == pointer and view.numel() == count
        self.ops.append(dist.P2POp(dist.isend if send else dist.irecv, view, peer))
        self.messages += 1
        if send:
            self.sent += 2 * count

    def ncclSend(self, *args):
        self.command(True, *args)

    def ncclRecv(self, *args):
        self.command(False, *args)

    def ncclGroupEnd(self):
        handles = dist.batch_isend_irecv(self.ops)
        for handle in handles:
            handle.wait()
        self.ops = None


env = dict(contextmanager=contextmanager, buffer_type=int, cudaStream_t=int,
           ncclDataTypeEnum=SimpleNamespace(from_torch=lambda dtype: dtype),
           get_current_device_stream_fast=lambda: SimpleNamespace(cuda_stream=123))
exec("from __future__ import annotations\n" + ast.unparse(ast.Module(parts, [])), env)
for method in methods:
    setattr(Comm, method, env[method])


def worker(rank, size, folder):
    comm = Comm(rank)
    original.fixture.Comm = lambda expected_rank: comm if expected_rank == rank else None
    original.worker(rank, size, folder)
    (Path(folder) / ("native_" + str(rank))).write_text(json.dumps([comm.groups, comm.messages]))


if __name__ == "__main__":
    results = {}
    for size in (4, 8):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker, args=(size, folder), nprocs=size, join=True)
            records = [json.loads((Path(folder) / ("native_" + str(i))).read_bytes()) for i in range(size)]
            results[size] = dict(groups=sum(r[0] for r in records), calls=sum(r[1] for r in records),
                                 transport=json.loads((Path(folder) / "result.json").read_bytes()))
    print(json.dumps(dict(source=SOURCE, pin=PIN, source_sha256=sha256(native).hexdigest(),
                          results=results, physical_h200=False,
                          scope="Pinned PyNCCL methods and actual tensors. Host Gloo substitutes for NCCL and CUDA stream execution."), indent=2))
