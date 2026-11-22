"""Exercise V142's projected layout through the pinned graph and PyNccl wrappers on CPU."""

import argparse
import ast
from contextlib import contextmanager, nullcontext
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch


def load_method(source, class_name, name, namespace):
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    exec(compile(module, name, "exec"), namespace)
    return namespace[name]


def main(stage):
    pinned = {}
    for short, name in (("coordinator", "distributed/parallel_state.py"),
                        ("py_nccl", "distributed/device_communicators/pynccl.py")):
        path = "python/sglang/srt/" + name
        pinned[short] = subprocess.check_output(["git", "-C", stage, "show", f"HEAD:{path}"]).decode()
    projection = (stage / "python/sglang/srt/layers/muse_spark/projection.py").read_text()
    totals = dict(layouts=0, rank_projections=0, peer_exchanges=0, bf16_words=0,
                  old_2d_call_rejected=0)
    torch.set_num_threads(1)
    stream = SimpleNamespace(cuda_stream=113)
    env = dict(torch=torch, nullcontext=nullcontext, contextmanager=contextmanager,
               get_current_device_stream_fast=lambda: stream,
               buffer_type=int, cudaStream_t=int,
               ncclDataTypeEnum=SimpleNamespace(from_torch=lambda dtype: dtype),
               get_forward=lambda: SimpleNamespace(sp_active=True))
    project = load_method(projection, "MuseShardedExpertProjection", "forward", env)
    capture = load_method(pinned["coordinator"], "GroupCoordinator", "graph_capture", env)
    public = load_method(pinned["coordinator"], "GroupCoordinator", "all_to_all_single", env)
    dispatch = load_method(pinned["coordinator"], "GroupCoordinator", "_all_to_all_single", env)
    transport = load_method(pinned["py_nccl"], "PyNcclCommunicator", "all_to_all_single", env)
    resolve = load_method(pinned["py_nccl"], "PyNcclCommunicator", "_resolve_stream", env)
    state = load_method(pinned["py_nccl"], "PyNcclCommunicator", "change_state", env)

    for ranks in (2, 4, 8):
        for rows, features, width in ((1, 13, 3), (3, 32, 11), (7, 32, 16),
                                      (8, 8192, 512), (16, 4096, 1024)):
            chunk = rows * width
            inputs = [torch.full((rows, features), float(i), dtype=torch.bfloat16) for i in range(ranks)]
            whole = torch.cat(inputs)
            words = torch.arange(ranks * rows * width, dtype=torch.int32)
            # Hash each bit position, including sign, infinities, NaNs and subnormals.
            def payload(i):
                all_bits = ((words * 241 + i * 37813) % 65536 - 32768).short()
                all_bits[:3] = torch.tensor([0, -32768, 32705], dtype=torch.int16)
                return all_bits.view(torch.bfloat16).view(ranks * rows, width)

            payloads = [payload(i) for i in range(ranks)]
            for rank in range(ranks):
                data = payloads[rank]
                nccl = SimpleNamespace()
                comm = SimpleNamespace(device=torch.device("cpu"), disabled=True, comm=rank,
                                       world_size=ranks, available=True, nccl=nccl)
                comm.all_to_all_single = lambda out, inp: transport(comm, out, inp)
                comm.change_state = lambda **kw: state(comm, **kw)
                comm._resolve_stream = lambda: resolve(comm)
                group = SimpleNamespace(world_size=ranks, ca_comm=None, pynccl_comm=comm,
                                        pymscclpp_comm=None, device_module=SimpleNamespace(stream=lambda s: nullcontext()),
                                        unique_name="test_tp")
                group.all_to_all_single = lambda out, inp: public(group, out, inp)
                weight = torch.empty((width, features), dtype=torch.bfloat16)
                input_storage = inputs[rank]
                def gather(x, dim):
                    assert dim == 0 and x.data_ptr() == input_storage.data_ptr()
                    return whole
                group.all_gather = gather
                def gemm(x, w, bias):
                    assert x is whole and w is weight and bias is None
                    return data
                env["bf16_gemm_dispatch"] = gemm
                pointers = {}
                def registered(output, input_, group_name):
                    assert group_name == group.unique_name and output.ndim == input_.ndim == 1
                    assert input_.data_ptr() == data.data_ptr() and output.is_contiguous()
                    pointers["out"] = output.data_ptr()
                    dispatch(group, output, input_)
                env["reg_all_to_all_single"] = registered
                sent, received, boundaries = [], [], []
                nccl.ncclGroupStart = lambda: boundaries.append("start")
                nccl.ncclGroupEnd = lambda: boundaries.append("end")
                def send(ptr, count, dtype, peer, own, current_stream):
                    assert (count, dtype, own, current_stream) == (chunk, torch.bfloat16, rank, stream.cuda_stream)
                    assert ptr == data.data_ptr() + 2 * peer * chunk
                    sent.append(peer)
                def recv(ptr, count, dtype, peer, own, current_stream):
                    assert (count, dtype, own, current_stream) == (chunk, torch.bfloat16, rank, stream.cuda_stream)
                    assert ptr == pointers["out"] + 2 * peer * chunk
                    source = payloads[peer].data_ptr() + 2 * rank * chunk
                    ctypes.memmove(ptr, source, 2 * chunk)
                    received.append(peer)
                nccl.ncclSend, nccl.ncclRecv = send, recv
                with capture(group, SimpleNamespace(stream=stream)):
                    assert not comm.disabled
                    out = project(SimpleNamespace(tp_group=group, weight=weight), input_storage)
                assert comm.disabled
                assert sent == received == list(range(ranks)) and boundaries == ["start", "end"]
                expected = torch.cat([p[rank * rows:(rank + 1) * rows] for p in payloads], 1)
                assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
                totals["rank_projections"] += 1
                totals["peer_exchanges"] += ranks
                totals["bf16_words"] += out.numel()
                # Show the previous shape would enter an invalid dimension-zero narrow.
                nccl.ncclSend = nccl.ncclRecv = lambda *args: None
                with state(comm, enable=True):
                    try:
                        transport(comm, torch.empty_like(data), data)
                    except (IndexError, RuntimeError) as error:
                        assert any(term in str(error) for term in ("start", "length", "narrow", "exceeds dimension"))
                    else:
                        raise AssertionError("the previous matrix-shaped call unexpectedly succeeded")
                totals["old_2d_call_rejected"] += 1
            totals["layouts"] += 1
    return dict(tested="Pinned graph-capture, group dispatch, PyNccl send/recv and V142 projection; CPU pointer transport",
                hashes={name: hashlib.sha256(src.encode()).hexdigest()
                        for name, src in {**pinned, "projection": projection}.items()},
                **totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    print(json.dumps(main(parser.parse_args().stage), indent=2))
