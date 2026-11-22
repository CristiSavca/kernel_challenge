"""Exercise the production prefix staging helper and bridge with an ordered CPU stream."""
import gc
import hashlib
import importlib.util
import json
import sys
import weakref
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import PropertyMock, patch

import torch

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v159_bridge", here.parent / "v159_mixed_verify_forward/bridge_cpu.py")
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
start = ref.env["start_mixed_prefix_readback"]
forward = ref.env["try_forward_proposed_mixed"]
copy_cpu = torch.Tensor.copy_
empty_cpu = torch.empty


class Stream:
    def __init__(self):
        self.queue = []
        self.completed = 0
        self.waits = []
        self.source_records = []

    def submit(self, name, fn=lambda: None):
        self.queue.append((name, fn))
        return len(self.queue)

    def finish(self, sequence):
        while self.completed < sequence:
            self.queue[self.completed][1]()
            self.completed += 1


class Event:
    def __init__(self, stream):
        self.stream = stream

    def record(self, stream):
        assert stream is self.stream
        self.at = stream.submit("event")

    def synchronize(self):
        self.stream.waits.append(self.at)
        self.stream.finish(self.at)


class PinnedAllocator:
    """Model ATen's storage hold until asynchronous copy completion; ordinary tensors are CPU."""
    def __init__(self, stream):
        self.stream = stream
        self.blocks = []

    def alloc(self, shape, *, dtype, device, pin_memory):
        assert dtype == torch.int64 and device == "cpu" and pin_memory
        for block in self.blocks:
            if (block.view() is None and self.stream.completed >= block.ready
                    and block.storage.shape == shape):
                break
        else:
            block = NS(storage=empty_cpu(shape, dtype=dtype), ready=0)
            self.blocks.append(block)
        view = block.storage.view(shape)
        block.view = weakref.ref(view)
        return view

    def start_copy(self, dst, src, *, non_blocking):
        assert non_blocking and src is self.stream.source_records[-1]
        block = next(block for block in self.blocks if block.view() is dst)
        block.storage.fill_(-1)
        block.ready = self.stream.submit("copy", lambda: copy_cpu(block.storage, src))
        return dst

    def start(self, batch):
        with ExitStack() as ctx:
            ctx.enter_context(patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True))
            ctx.enter_context(patch.object(torch, "empty", self.alloc))
            ctx.enter_context(patch.object(torch.Tensor, "copy_", lambda d,s,**kw: self.start_copy(d,s,**kw)))
            ctx.enter_context(patch.object(torch.cuda, "current_stream", lambda device: self.stream))
            ctx.enter_context(patch.object(torch.cuda, "Event", lambda: Event(self.stream)))
            def record(source, stream):
                assert stream is self.stream
                stream.source_records.append(source)
            ctx.enter_context(patch.object(torch.Tensor, "record_stream", record))
            return start(batch)


def check_abandon(n):
    stream = Stream()
    allocator = PinnedAllocator(stream)
    src = torch.arange(n)
    batch = NS(reqs=list(range(n)), seq_lens=src, seq_lens_cpu=None)
    first = allocator.start(batch)
    addr = first[1].data_ptr()
    ref_first = weakref.ref(first[1])
    del first
    gc.collect()
    assert ref_first() is None
    second = allocator.start(batch)
    assert addr != second[1].data_ptr() and not stream.waits
    stream.finish(2)
    third = allocator.start(batch)
    assert addr == third[1].data_ptr()
    assert not stream.waits


def run_one(backend, tails, nonce, mirror=False, veto=None, corrupt=None):
    stream = Stream()
    allocator = PinnedAllocator(stream)
    def simulate(worker, prompt, decode, proposal, **kwargs):
        handle = allocator.start(decode)
        stream.submit("draft")
        if mirror:
            assert handle is None
        else:
            assert handle[0] is proposal.prefix_lens
            assert handle[1].numel() * handle[1].element_size() == tails * 8
        if corrupt:
            src, dst, event = handle
            if corrupt == "identity":
                src = src.clone()
            elif corrupt == "shape":
                dst = dst[:, None]
            elif corrupt == "dtype":
                dst = dst.int()
            handle = src, dst, event
        pinned = (lambda x: handle is not None and x is handle[1] and corrupt != "not_pinned")
        with patch.object(torch.Tensor, "is_pinned", pinned):
            result = forward(worker, prompt, decode, proposal, prefix_readback=handle, **kwargs)
        if corrupt:
            assert result is None and worker.is_current_decode_proposal(decode, proposal)
        expected_wait = not mirror and not corrupt and veto not in ("sampling", "cost", "stale")
        assert stream.waits == ([2] if expected_wait else [])
        assert stream.completed == (2 if expected_wait else 0)
        assert stream.queue[-1][0] == "draft"
        return result
    with patch.dict(ref.env, try_forward_proposed_mixed=simulate):
        # A descriptive non-built-in veto name makes the base fixture verify a
        # declined path without modifying ordinary setup.
        ref.run(backend, tails, nonce, mirror, veto if not corrupt else "readback_" + corrupt)


with torch.inference_mode():
    n = 0
    for backend in ("FULL", "BREAKABLE"):
        for t in (1, 2, 4):
            for i in range(8):
                for existing in (False, True):
                    run_one(backend, t, i, existing)
                    n += 1
    for veto in ("sampling", "cost", "stale", "no_bucket", "no_headroom", "rank_disagrees"):
        run_one("FULL", 2, 1, veto=veto)
    for corrupt in ("identity", "shape", "dtype", "not_pinned"):
        run_one("BREAKABLE", 2, 1, corrupt=corrupt)
    for nrows in (1, 2, 4, 32, 257):
        check_abandon(nrows)
    guards = [(None,None,1), (torch.arange(1),torch.arange(1),1),
              (torch.arange(1),None,1), (torch.arange(1,dtype=torch.int32),None,1),
              (torch.arange(1),None,2), (torch.arange(0),None,0),
              (torch.zeros((1,1),dtype=torch.int64),None,1)]
    for g,(base, mirror, size) in enumerate(guards):
        batch = NS(reqs=list(range(size)), seq_lens=base, seq_lens_cpu=mirror)
        if g == 2:
            assert start(batch) is None
        else:
            assert PinnedAllocator(Stream()).start(batch) is None
    # The V159 default caller supplies only on_publish.
    for backend in ("FULL", "BREAKABLE"):
        for t in (1,2,4):
            ref.run(backend,t,3,False)
    with patch.object(torch.Tensor,"record_stream",lambda *args:None):
        fallback = ref.standalone_fallback()

# Exercise the actual scheduler run_batch/overlap role method from the assembled
# production candidate. Each follow-up ordinary decode runs with no mixed readback.
sched = ref.module("v160_schedule_fixture", here/"execution_cpu.py")
bindings = sys.modules["sglang.srt.speculative.dflash_mixed_forward"]
native_propose = sched.Worker.propose_decode_block
scheduler_rounds = 0
for nonce in range(8):
    for allowed in (False, True):
        for host in (False, True):
            stream = Stream()
            pool = PinnedAllocator(stream)
            calls = []
            def early(d):
                assert not calls and (d.seq_lens_cpu is not None) == host
                # Initial role lengths are 13. FutureMap has already resolved
                # these positions to that round's previous published lengths.
                assert torch.equal(d.seq_lens, d.spec_info.future_indices + 40)
                calls.append(("staged", d.seq_lens))
                return pool.start(d)
            def propose(worker,d):
                if len(calls) == 1:
                    assert d.seq_lens is calls[0][1]
                    stream.submit("draft")
                    calls.append(("proposal", None))
                return native_propose(worker,d)
            def voted(worker,p,d,proposal,*,on_publish,prefix_readback):
                assert [c[0] for c in calls] == ["staged","proposal"]
                assert (prefix_readback is None) == host
                if prefix_readback is not None:
                    src, pinned, ready = prefix_readback
                    assert src is d.seq_lens
                    if allowed:
                        ready.synchronize()
                        assert torch.equal(pinned,proposal.prefix)
                assert stream.completed == (2 if allowed and not host else 0)
                assert stream.queue[-1][0] == "draft"
                calls.append(("voted",None))
                return sched.try_mixed(worker,p,d,proposal,on_publish=on_publish,prefix_readback=d.seq_lens)
            with patch.object(bindings,"start_mixed_prefix_readback",early), patch.object(
                    bindings,"try_forward_proposed_mixed",voted), patch.object(sched.Worker,"propose_decode_block",propose):
                sched.setup(nonce,allowed,host,True,True)
            assert [c[0] for c in calls] == ["staged","proposal","voted"]
            scheduler_rounds += 1

delta = here / "readback_delta.diff"
cuda_source = "https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/"
print(json.dumps(dict(
    delta_sha256=hashlib.sha256(delta.read_bytes()).hexdigest(),
    bridge_sha256=hashlib.sha256((ref.path/"dflash_mixed_forward.py").read_bytes()).hexdigest(),
    scheduler_rounds_with_following_ordinary_tick=scheduler_rounds, graph_mode_host_dispatches=n, ordered_event_waits=n//2, exact_host_mirror_bypasses=n//2,
    declined_graph_or_consensus_after_early_event=3, cheap_rejections_without_wait=3,
    invalid_handle_drops_without_wait=4, helper_guard_cases=len(guards),
    default_v159_calls=6, standalone_fallbacks=fallback, abandon_reuse_lengths=[1,2,4,32,257],
    physical_cuda=False, allocator="Ordered CPU model. ATen CUDA source checked independently.",
    copy_source=cuda_source+"native/cuda/Copy.cu#L413-L457",
    allocator_source=cuda_source+"core/CachingHostAllocator.h#L711-L745",
), indent=2))
