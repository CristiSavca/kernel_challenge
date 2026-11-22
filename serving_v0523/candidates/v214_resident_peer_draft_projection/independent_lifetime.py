"""CPU audit of weight-loader invalidation and allocator stream bookkeeping in V214."""
import ast
import contextlib
import ctypes
import gc
import hashlib
import importlib.util
import json
import logging
import sys
import traceback
import types
import weakref
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch


HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
SRT = STAGE / "python/sglang/srt"
SOURCES = {}


def module_from(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def method(path, owner, name, env):
    data = (SRT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(data).hexdigest()
    body = ast.parse(data).body
    if owner:
        body = next(node.body for node in body if getattr(node, "name", None) == owner)
    node = next(node for node in body if getattr(node, "name", None) == name)
    exec("from __future__ import annotations\n" + ast.unparse(node), env)
    return env[name]


qa = module_from("prefix_qa", HERE.parent / "v212_mixed_paired_draft_kv/worker_prefix_check.py")
projection = module_from("peer_projection", SRT / "layers/dflash_context_projection.py")
for path in ("layers/dflash_context_projection.py", "../kernels/ops/speculative/fused_kv_materialize.py",
             "managers/scheduler.py", "utils/common.py"):
    SOURCES[path] = hashlib.sha256((SRT / path).read_bytes()).hexdigest()
invariant = types.ModuleType("sglang.srt.batch_invariant_ops")
invariant.is_batch_invariant_mode_enabled = lambda: False
sys.modules[invariant.__name__] = invariant
logging.disable(logging.CRITICAL)
torch.set_num_threads(1)


def updater_lifecycle():
    worker_file = "speculative/dflash_worker_v2.py"
    manager_file = "managers/scheduler_components/weight_updater.py"
    loader_env = {"torch": torch}
    weight_loader = method("model_loader/weight_utils.py", None, "default_weight_loader", loader_env)
    helper_env = dict(qa.init_scope, torch=torch)
    qa.Worker.invalidate_draft_kv_projection = method(worker_file, "DFlashWorkerV2",
                                                      "invalidate_draft_kv_projection", helper_env)
    qa.Worker.weight_update_runners = method(worker_file, "DFlashWorkerV2", "weight_update_runners", helper_env)
    options = [
        ("disk", "all", None), ("disk", "all", "target"), ("disk", "all", "draft"),
        ("ipc", "all", None), ("ipc", "all", "target"), ("ipc", "all", "draft"),
        *[(source, role, fail) for source in ("tensor", "distributed")
          for role in ("target", "draft", "all") for fail in (None, role if role != "all" else "target", "draft")
          if fail is None or role in (fail, "all")],
        ("distributed", "all", "receive"),
        *[(stage, role, None) for stage in ("begin", "end") for role in ("target", "draft", "all")],
        *[(action, role, None) for action in ("checksum", "compare", "snapshot", "reset_tensors")
          for role in ("target", "draft", "all")],
    ]
    options = list(dict.fromkeys(options))
    updates = 0
    stale = 0
    retained = 0
    outcomes = {}
    apply_reset = method("utils/weight_checker.py", "WeightChecker", "_reset_tensors",
                         {"_is_skip_weight_check": lambda *a: False, "_random_like": lambda p: p + 11})
    for source, role, fail in options:
        case = qa.build("fused", True, (1, 3), 4, 214502)
        worker = case.worker
        # Parameter storage and the prepacked bank must have independent identity.
        for layer in worker.draft_model.layers:
            linear = layer.self_attn.qkv_proj
            linear.weight = torch.nn.Parameter(linear.weight, requires_grad=False)
        worker._fused_kv_helper = None
        with torch.inference_mode():
            worker._init_fused_kv_helper()
        bank = worker._fused_kv_helper
        assert bank is not None and not torch.is_inference(bank.flat_kv_weight_t)
        bank._paired_kv_peer = object()
        weight = worker.draft_model.layers[1].self_attn.qkv_proj.weight
        old_tensor_version = weight._version
        before = bank.flat_kv_weight_t.clone()
        calls = []
        state = NS(selector=role, loaded_weights=False)
        session = lambda **kw: NS(**{"loaded_weights": False, **kw})
        service = {"torch": torch, "logger": logging.getLogger("V214"), "traceback": traceback,
                   "_WeightUpdateSession": session, "get_parallel": lambda: NS(tp_rank=0),
                   "msgspec": NS(structs=NS(replace=lambda obj, **kw: NS(**{**vars(obj), **kw})))}
        for name in ("UpdateWeightFromDiskReqOutput", "UpdateWeightsFromTensorReqOutput",
                     "UpdateWeightsFromDistributedReqOutput", "UpdateWeightsFromIPCReqOutput",
                     "BeginWeightUpdateReqOutput", "EndWeightUpdateReqOutput", "CheckWeightsReqOutput"):
            service[name] = NS
        method(manager_file, None, "_parse_runner_selector", service)
        class Manager:
            pass
        manager_names = ("_select_runners", "_weight_load_runners", "update_weights_from_disk",
                         "update_weights_from_tensor", "update_weights_from_distributed", "update_weights_from_ipc",
                         "begin_weight_update", "end_weight_update", "check_weights")
        for name in manager_names:
            setattr(Manager, name, method(manager_file, "SchedulerWeightUpdaterManager", name, service))

        def loader_for(selected):
            def update(*args, **kwargs):
                calls.append(selected)
                if selected == "draft":
                    assert worker._fused_kv_helper is None
                    weight_loader(weight, weight.detach().clone() + 7)
                return (selected != fail, "test")

            def receive(**kwargs):
                if fail == "receive":
                    raise ValueError("no payload")
                return [("fixture", torch.tensor(1))]

            return NS(update_weights_from_disk=update, load_weights_from_distributed=update,
                      receive_weights_from_distributed=receive, update_weights_from_tensor=update,
                      update_weights_from_ipc=update, begin_weight_update=update, end_weight_update=update)

        def checker(*, action, role, **kw):
            if action == "reset_tensors" and role == "draft":
                assert worker._fused_kv_helper is None
                calls.append("draft")
                apply_reset(NS(_model_state=lambda: [("layer.1.kv_proj.weight", weight)]))
            return None

        manager = Manager()
        target = NS(weight_updater=loader_for("target"), check_weights=checker)
        worker.draft_model_runner.weight_updater = loader_for("draft")
        worker.draft_model_runner.check_weights = checker
        manager.tp_worker = NS(model_runner=target, weight_update_runners=lambda: [("target", target)],
                               deserialize_own_rank=lambda payload: payload)
        manager.draft_worker = worker
        manager.tp_cpu_group = object()
        manager._observe_weight_load = lambda *args: contextlib.nullcontext()
        manager.record_weight_version_after_update = lambda *args: None
        manager.flush_cache_after_weight_update = lambda req, ok, message: (ok, message)
        manager._session = None if source == "begin" else state
        args = NS(selector=role, weight_version=None, model_path="", load_format=None, recapture_cuda_graph=False,
                  serialized_named_tensors=[None], serialized_named_weights={0:None}, names=[], dtypes=[], shapes=[],
                  group_name="", action=source, allow_quant_error=False, skip_tensor_list=None, update_name=None, handles={})
        with torch.inference_mode(), patch.object(torch.distributed, "get_world_size", return_value=1), \
             patch.object(torch.distributed, "barrier"):
            if source in ("checksum", "compare", "snapshot", "reset_tensors"):
                result = manager.check_weights(args)
            elif source in ("begin", "end"):
                result = getattr(manager, source + "_weight_update")(args)
            else:
                result = getattr(manager, "update_weights_from_" + source)(args)
        assert result.success == (fail is None), (source, role, fail, vars(result))
        mutated = "draft" in calls
        assert (worker._fused_kv_helper is None) == mutated, (source, role, fail, calls)
        if mutated:
            # A true .data write changes parameters without changing PyTorch's version.
            if source != "reset_tensors":
                assert weight._version == old_tensor_version
                stale += 1
            assert torch.equal(before, bank.flat_kv_weight_t)
            with torch.inference_mode():
                qa.settle(case, False, False)
            rebuilt = worker._fused_kv_helper
            assert rebuilt is not None and rebuilt is not bank and not torch.is_inference(rebuilt.flat_kv_weight_t)
            assert getattr(rebuilt, "_paired_kv_peer", None) is None
            expected = torch.stack([a.self_attn.qkv_proj.weight[a.self_attn.q_size:] for a in worker.draft_model.layers])
            expected = expected.flatten(0, 1).T.contiguous()
            assert torch.equal(expected, rebuilt.flat_kv_weight_t) and not torch.equal(before, expected)
            updates += 1
        else:
            assert worker._fused_kv_helper is bank and bank._paired_kv_peer is not None
            retained += 1
        outcomes["/".join([source, role, fail or "ok"])] = "rebuilt" if mutated else "retained"
    return dict(cases=len(options), rebuilt=updates, preserved=retained, unchanged_parameter_versions=stale,
                outcomes=outcomes)


def stream_lifecycle():
    """Redirect the pinned PyNCCL ABI to CPU pointers; tag modeled CUDA streams and retirement events."""
    events = []
    state = NS(active=None, event=0, mismatch=False)

    class Stream:
        def __init__(self, index):
            self.cuda_stream = index

        def record_event(self):
            state.event += 1
            events.append(("record_event", self.cuda_stream, state.event))
            return state.event

        def wait_event(self, event):
            events.append(("wait_event", self.cuda_stream, event))

    class TrackedTensor(torch.Tensor):
        @property
        def is_cuda(self):
            # Storage remains CPU. This engages the production stream-accounting block.
            return True

        def record_stream(self, stream):
            events.append(("record_tensor", stream.cuda_stream, self.data_ptr()))

    class NativeCalls:
        def ncclSend(self, address, n, dtype, peer, comm, stream):
            events.append(("send", stream, address, n, peer))
            assert stream == state.active.cuda_stream

        def ncclRecv(self, address, n, dtype, peer, comm, stream):
            events.append(("recv", stream, address, n, peer))
            assert stream == state.active.cuda_stream
            values = torch.full((n,), 1.0 + peer, dtype=torch.bfloat16)
            ctypes.memmove(address, values.data_ptr(), n * values.element_size())

        def ncclGroupStart(self):
            events.append(("start", state.active.cuda_stream))

        def ncclGroupEnd(self):
            events.append(("end", state.active.cuda_stream))

    nccl_env = {"torch": torch, "contextmanager": contextlib.contextmanager, "get_current_device_stream_fast": lambda: state.active,
                "buffer_type": int, "cudaStream_t": int, "ncclDataTypeEnum": NS(from_torch=lambda dtype: repr(dtype))}
    class Comm:
        pass
    for name in ("_resolve_stream", "group_start", "group_end", "send", "recv", "change_state"):
        setattr(Comm, name, method("distributed/device_communicators/pynccl.py", "PyNcclCommunicator", name, nccl_env))
    comm = Comm()
    comm.disabled, comm.available, comm.device, comm.comm, comm.nccl = True, True, torch.device("cpu"), 1, NativeCalls()
    group = NS(world_size=4, rank_in_group=0, pynccl_comm=comm, cpu_group=object())
    rows, hidden, channels = 256, 64, 8

    def agree(peers, local, group=None):
        for peer in peers:
            peer.copy_(local)
        if state.mismatch:
            peers[2][-1] = 1 - local[-1]

    def bank(tracked):
        weight = torch.ones((hidden, channels), dtype=torch.bfloat16)
        return weight.as_subclass(TrackedTensor) if tracked else weight

    h = NS(flat_kv_weight_t=bank(True), n_layers=2, layer_out_dim=4)
    read_calls = []
    def use(stream_id):
        events.clear()
        state.active = Stream(stream_id)
        inp = torch.full((rows, hidden), stream_id / 7, dtype=torch.bfloat16)
        input_ref = weakref.ref(inp)
        with projection.dflash_sp_prefill_scope(None, h), torch.inference_mode():
            value = projection.project_paired_draft_kv(inp, rows * 4 - 1, group)
        del inp
        gc.collect()
        assert input_ref() is None, "captured a request activation"
        if value is not None:
            assert value.shape == (rows * 4 - 1, channels)
            output_ref = weakref.ref(value)
            del value
            gc.collect()
            assert output_ref() is None, "retained the returned request buffer"
        read_calls.append(list(events))
        return list(events)

    with patch.object(torch.cuda, "current_stream", side_effect=lambda device: state.active), \
         patch.object(torch.distributed, "all_gather", side_effect=agree), \
         patch.object(torch.distributed, "all_reduce", side_effect=lambda *a, **kw: None):
        first = use(11)
        live = h._paired_kv_peer
        first_event = live.ready_event
        peer_ptr = live.remote_weight.data_ptr()
        local_ptr = h.flat_kv_weight_t.data_ptr()
        assert any(e[:3] == ("record_tensor", 11, local_ptr) for e in first)
        assert any(e[0] == "recv" and e[2] == peer_ptr for e in first)
        assert first[-1] == ("record_event", 11, first_event)
        assert h._paired_kv_peer.remote_weight.untyped_storage().nbytes() == hidden * channels * 2
        for other_stream in (17, 23):
            warm = use(other_stream)
            assert h._paired_kv_peer is live
            assert warm.index(("wait_event", other_stream, first_event)) < next(i for i,e in enumerate(warm) if e[0]=="send")
            assert ("record_tensor", other_stream, local_ptr) in warm
            assert ("record_tensor", other_stream, peer_ptr) in warm
            assert not any(e[0] == "recv" and e[2] == peer_ptr for e in warm)
            assert not any(e[0] == "record_event" for e in warm)
        previous = weakref.ref(live.remote_weight)
        del live
        state.mismatch = True
        decline = use(31)
        assert decline == [] and h._paired_kv_peer is None
        gc.collect()
        assert previous() is None
        state.mismatch = False
        refill = use(37)
        assert len([e for e in refill if e[0] == "record_event"]) == 1
        version = h.flat_kv_weight_t._version
        h.flat_kv_weight_t.add_(2)
        assert h.flat_kv_weight_t._version > version
        replaced = use(41)
        assert not any(e[0] == "wait_event" for e in replaced)
        assert any(e[0] == "record_event" for e in replaced)
        h.flat_kv_weight_t = bank(True)
        identity_change = use(43)
        assert any(e[0] == "record_event" for e in identity_change)
        with torch.inference_mode():
            h.flat_kv_weight_t = bank(False)
        for step in (47, 53):
            # Inference-only banks have no usable version, but fresh exchanges remain valid.
            fresh = use(step)
            assert h._paired_kv_peer is None
            assert any(e[0] == "recv" and e[3] == hidden * channels for e in fresh)
    return dict(calls=len(read_calls), synthetic_streams=[11,17,23,31,37,41,43,47,53],
                retained_weight_bytes=hidden * channels * 2, warm_cross_stream_waits=2,
                local_allocator_stream_recorded=True, peer_allocator_stream_recorded=True,
                no_request_tensors_retained=True, rank_disagreement_before_p2p=True,
                in_place_mutation_and_weight_replacement_refreshed=True,
                inference_only_untracked_weights_always_exchange=True)


if __name__ == "__main__":
    print(json.dumps(dict(production_entry={
        "initialization": "run_scheduler_process -> Scheduler.__init__/init_model_worker -> maybe_init_draft_worker -> DFlashWorkerV2.__init__ -> _init_fused_kv_helper -> FusedKVMaterializeHelper. Stacked source rows are copied into a fresh contiguous versioned tensor under inference_mode(False) and no_grad. The ordinary event loops run under DynamicGradMode.",
        "updates": "SchedulerWeightUpdaterManager selects role-specific runners. Its mutating iterator invalidates just before the draft runner receives disk, IPC, tensor, distributed, begin/end session or reset_tensors work. Existing caches survive read-only checks and target-only operations. The next append rebuilds packed QKV using current draft weights.",
        "streams": "PyNcclCommunicator.send and recv resolve the current device stream, and pass raw pointers and cuda_stream handles to NCCL. The resident source gets record_stream on every accepted CUDA call. On a warm different stream the producer event is waited and the retained peer allocation gets record_stream before GEMM/P2P."
    }, weight_updates=updater_lifecycle(), streams=stream_lifecycle(), sources=SOURCES,
                          torch=torch.__version__, hardware=False,
                          limits="Exact scheduler updater, frozen native worker/helper, PyTorch version tracking, and PyNCCL ordering. The rank votes and transport execute with CPU test doubles. CUDA DMA, event scheduling and allocator reuse were not observed."), indent=2))
