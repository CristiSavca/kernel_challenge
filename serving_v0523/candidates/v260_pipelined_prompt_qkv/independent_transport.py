"""Replay V260's grouped PyNCCL ABI and breakable graph stream tracking on a CPU."""
import ast
from collections import Counter
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
SRT = "python/sglang/srt/"
FILES = {
    "qkv": "layers/communication/sp_half_qkv.py",
    "comm": "distributed/device_communicators/pynccl.py",
    "common": "utils/common.py",
    "breakable": "model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py",
    "runner": "model_executor/runner/prefill_cuda_graph_runner.py",
    "model": "models/muse_spark.py",
    "sp": "layers/layernorm_sp.py",
    "linear": "layers/linear.py",
    "unquant": "layers/quantization/unquant.py",
    "gemv": "../kernels/ops/gemm/hopper_bf16_gemv.py",
    "mixed": "speculative/dflash_mixed_forward.py",
}


def read_export():
    cfg = json.loads((HERE/"validation.json").read_bytes())
    repo = HERE.parents[1]/"sglang"
    patch = HERE/"patch.diff"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == cfg["patch_sha256"]
    with TemporaryDirectory(prefix="okc-v260-native-") as d:
        objects = Path(d)/"objects"
        objects.mkdir()
        env = dict(os.environ, GIT_INDEX_FILE=d+"/index", GIT_OBJECT_DIRECTORY=str(objects),
                   GIT_ALTERNATE_OBJECT_DIRECTORIES=str(repo/".git/objects"))
        def git(*args):
            return subprocess.check_output(["git","-C",repo,*args],env=env)
        git("read-tree",cfg["base"])
        git("apply","--cached","--whitespace=error-all",patch)
        tree = git("write-tree").decode().strip()
        assert tree == cfg["tree"]
        files = {k:git("show", tree+":"+os.path.normpath(SRT+v)).decode() for k,v in FILES.items()}
    return cfg, files


def native(source, name, env, cls=None):
    body = ast.parse(source).body
    if cls:
        body = next(node.body for node in body if getattr(node,"name",None) == cls)
    node = next(node for node in body if getattr(node,"name",None) == name)
    node.decorator_list = []
    exec("from __future__ import annotations\n" + ast.unparse(node), env)
    return env[name]


class Stream:
    def __init__(self, i):
        self.stream_id = self.cuda_stream = i


class Tensor:
    device = 3
    dtype = "bf16"

    def __init__(self, length, pointer):
        self.length = length
        self.pointer = pointer

    def numel(self):
        return self.length

    def data_ptr(self):
        return self.pointer

    def copy_(self, local):
        assert self.length == local.length


class Half(Tensor):
    def __init__(self, count, local, pointer):
        super().__init__(count * local, pointer)
        self.shape = (count, local)

    def __getitem__(self, i):
        return Tensor(self.shape[1], self.pointer + 2 * self.shape[1] * i)


def replay(files):
    stream = ContextVar("stream")
    cap = ContextVar("capturing", default=None)
    forks = ContextVar("forked", default=None)
    main, transfer = Stream(71), Stream(125)
    module = NS(current_device=lambda:3, current_stream=lambda device: stream.get() if device==3 else None)
    get_stream = native(files["common"], "get_current_device_stream_fast",
                        dict(torch=NS(get_device_module=lambda:module),cached_device_index=-1))
    edges = []
    hook = native(files["breakable"], "_hooked_wait_stream", dict(
        _forked_streams_var=forks, _current_stream_var=cap,
        _original_wait_stream=lambda a,b:edges.append((a.stream_id,b.stream_id)),
        _is_stream_capturing=lambda a:a in forks.get()))
    abi = dict(get_current_device_stream_fast=get_stream,buffer_type=int,cudaStream_t=int,
               ncclDataTypeEnum=NS(from_torch=lambda dt: 9 if dt=="bf16" else None))
    cls = type("Native", (), {name:native(files["comm"],name,abi,"PyNcclCommunicator")
                              for name in ("_resolve_stream","group_start","group_end","send","recv")})
    cohort = native(files["qkv"],"_exchange_half",{})
    buddy = native(files["qkv"],"_exchange_peer",{})
    summaries = []
    for size in (4, 6, 8, 10):
        for mode in ("eager", "full_capture", "breakable_capture"):
            half = size//2
            local = 2048*8192
            sent, received = Counter(), Counter()
            direction_calls = Counter()
            for rank in range(size):
                depth = []
                phase = 0

                def emit(kind, ptr, count, typ, peer, comm, cuda_stream):
                    assert len(depth)==1 and typ==9 and comm==rank
                    assert cuda_stream==(main.cuda_stream if phase==0 else transfer.cuda_stream)
                    expected = local if phase==0 else half*local
                    assert count==expected
                    if kind == "send":
                        assert ptr == (100 if phase==0 else 200)
                        sent[rank,peer,count,phase] += 1
                    else:
                        assert ptr == (200+2*local*(peer%half) if phase==0 else 300)
                        received[peer,rank,count,phase] += 1
                    direction_calls[kind] += 1

                def begin():
                    assert not depth
                    depth.append(True)

                def end():
                    assert depth.pop()

                abi_recorder=NS(ncclSend=lambda *x:emit("send",*x), ncclRecv=lambda *x:emit("recv",*x),
                                ncclGroupStart=begin,ncclGroupEnd=end)
                nc=cls()
                nc.device,nc.disabled,nc.comm,nc.nccl=3,False,rank,abi_recorder
                stream.set(main)
                cap.set(main if mode=="breakable_capture" else None)
                forks.set(set() if mode=="breakable_capture" else None)
                edges.clear()
                cohort(nc,rank,(rank//half)*half,Tensor(local,100),Half(half,local,200))
                hook(transfer,main)
                if mode=="breakable_capture":
                    assert forks.get()=={transfer}
                phase=1
                stream.set(transfer)
                buddy(nc,(rank+half)%size,Half(half,local,200),Half(half,local,300))
                stream.set(main)
                hook(main,transfer)
                assert edges==[(125,71),(71,125)] and not forks.get() and not depth
            assert sent==received and all(n==1 for n in sent.values())
            summaries.append(dict(mode=mode,ranks=size,matched_peer_transfers=sum(sent.values()),
                                  calls=dict(direction_calls)))
    return summaries


def backend(files):
    gemv = native(files["gemv"], "use_hopper_bf16_gemv", {"_is_sm12x":lambda:False})
    torch_linear = lambda *args: "F.linear"
    cfg = dict(_enable_bf16_splitk_gemm=False,_use_hopper_bf16_gemv=gemv,
               _use_cutedsl_bf16_gemm=None,F=NS(linear=torch_linear))
    project = native(files["unquant"],"_bf16_gemm_dispatch_impl",cfg)
    shapes = []
    for tp in (4,8):
        n, k = 2304*8//tp, 8192
        for rows in (512,576,1024,1152,2048,2080,4096):
            if rows*tp//2<n:
                continue
            for m in (rows*tp//2,rows*tp):
                x=NS(shape=(m,k),numel=lambda:m*k)
                assert project(x,NS(shape=(n,k)),None)=="F.linear"
                shapes.append((m,n,k))
    return sorted(set(shapes))


def main():
    cfg, files = read_export()
    assert "sp_active = layernorm_sp.runs_sp(forward_batch.forward_mode)" in files["model"]
    assert "if get_forward().sp_active and tp_size > 1:" in files["linear"]
    assert "forward_mode == ForwardMode.EXTEND" in files["sp"]
    assert "mix = copy(prompt_view)" in files["mixed"]
    assert "self.capture_forward_mode = ForwardMode.EXTEND" in files["runner"]
    assert "self.prefill_backend_name in (Backend.BREAKABLE, Backend.FULL)" in files["runner"]
    qkv=files["qkv"]
    assert qkv.index("transfer.wait_stream(compute)") < qkv.index("_exchange_peer(comm, (rank + half) % size, owned, remote)", qkv.index("def half_qkv(local:"))
    assert "finally:\n            # Join even if projection fails." in qkv and "compute.wait_stream(transfer)" in qkv
    assert "if not get_platform().is_sm100:" in files["unquant"] and 'requires SM100/SM103' in files["unquant"]
    results = replay(files)
    print(json.dumps(dict(patch=cfg["patch_sha256"],tree=cfg["tree"],abi_cases=results,
                          total_matched_transfers=sum(r["matched_peer_transfers"] for r in results),
                          native_hopper_gemv_dispatch_to_dense_shapes=backend(files),
                          capture_operation="Concrete pinned breakable hook with simulated stream status",
                          pynccl_operation="Native stream resolver and PyNCCL wrappers with recorded ABI calls",
                          physical_nccl=False,physical_cuda_graph=False),indent=2))


if __name__=="__main__":
    main()
