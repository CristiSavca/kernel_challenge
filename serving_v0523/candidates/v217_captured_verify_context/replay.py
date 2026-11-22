"""Run extracted production graph, admission, projection and worker materialization on Gloo."""
import ast
import dataclasses
import json
import sys
import time
import hashlib
import types
from statistics import median
from unittest.mock import patch
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.modules import module as module_hooks


@dataclasses.dataclass
class Output:
    hidden_states: torch.Tensor
    next_token_logits: torch.Tensor | None = None
    full_logits: torch.Tensor | None = None
    last_hidden_states: torch.Tensor | None = None
    dflash_verify_fc_shard: torch.Tensor | None = None


class Draft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(80, 32, bias=False, dtype=torch.bfloat16)
        self.hidden_norm = torch.nn.RMSNorm(32, eps=1e-5, dtype=torch.bfloat16)
        self.quantizable = False

    def project_target_hidden(self, hidden):
        return self.hidden_norm(self.fc(hidden))


class Group:
    def __init__(self, rank, size):
        self.rank_in_group = rank
        self.world_size = size
        self.cpu_group = dist.group.WORLD
        self.payloads = []

    def all_gather(self, local, dim):
        self.payloads.append(list(local.shape))
        with torch.inference_mode(False):
            parts = [torch.empty_like(local) for _ in range(self.world_size)]
            dist.all_gather(parts, local.contiguous())
        return torch.cat(parts, dim=dim)


def module(source, names, namespace):
    nodes = [n for n in source.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<candidate>", "exec"), namespace)


def find(tree, kind, token):
    matches = [n for n in ast.walk(tree) if isinstance(n, kind) and token(n)]
    assert len(matches) == 1, len(matches)
    return matches[0]


def wrap(source, statements, namespace):
    fn = ast.parse(source).body[0]
    fn.body = statements
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(mod, "<candidate-entry>", "exec"), namespace)
    return namespace[fn.name]


def api(stage):
    root = Path(stage) / "python/sglang/srt"
    def src(path):
        return ast.parse((root / path).read_text())
    ns = dict(torch=torch, module_hooks=module_hooks, dataclasses=dataclasses)
    module(src("layers/dflash_context_projection.py"), {"_agree", "DFlashVerifyFCGraph"}, ns)
    runner = src("model_executor/runner/decode_cuda_graph_runner.py")
    # Stub only the two platform imports in the actual eligibility routine.
    backend = types.ModuleType("full_cuda_graph_backend")
    class Full:
        pass
    backend.FullCudaGraphBackend = Full
    invariant = types.ModuleType("batch_invariant_ops")
    invariant.is_batch_invariant_mode_enabled = lambda:False
    sys.modules["sglang.srt.model_executor.runner_backend.full_cuda_graph_backend"] = backend
    sys.modules["sglang.srt.batch_invariant_ops"] = invariant
    ret = find(runner, ast.Return, lambda n: isinstance(n.value, ast.Call) and
               "dataclasses.replace" == ast.unparse(n.value.func) and "raw_num_token" in ast.unparse(n))
    slice_live = wrap("def slice_live(self, output, next_token_logits=None, full_logits=None): pass", [ret], ns)
    worker = src("speculative/dflash_worker_v2.py")
    append = find(worker, ast.FunctionDef, lambda n: n.name == "_append_target_hidden_to_draft_kv_by_loc")
    branch = find(append, ast.If, lambda n: ast.unparse(n.test) == "verify_fc_shard is not None")
    materialize = wrap("def materialize(self, target_hidden, verify_fc_shard, prefill_context_done=False): pass",
                       [branch, ast.parse("return ctx_hidden").body[0]], ns)
    invalidate = find(worker, ast.FunctionDef, lambda n: n.name == "invalidate_draft_kv_projection")
    retire = wrap("def retire(self): pass", invalidate.body, ns)
    normal = find(worker, ast.FunctionDef, lambda n: n.name == "verify_proposed_decode_block")
    flag = find(normal, ast.keyword, lambda n: n.arg == "verify_fc_shard")
    assert ast.unparse(flag.value) == "logits_output.dflash_verify_fc_shard"
    block = find(append, ast.With, lambda n: branch in n.body)
    assert "draft_tp_context" in ast.unparse(block.items[1].context_expr)
    graph = find(runner, ast.FunctionDef, lambda n: n.name == "capture_one_shape")
    graph_text = ast.unparse(graph)
    assert graph_text.index("capture_hook(self, out") < graph_text.index("verify_fc.capture(out")
    assert graph_text.index("self.backend.capture_one(") < graph_text.index("verify_fc.complete_capture(")
    model = find(src("models/muse_spark.py"), ast.ClassDef, lambda n:n.name == "MuseSpark1xModel")
    forward = next(n for n in model.body if getattr(n,"name",None)=="forward")
    ns.update(layernorm_sp=NS(runs_sp=lambda mode:False),
              get_forward=lambda:NS(set=lambda *a:None))
    body = wrap("def body(self, input_ids, positions, forward_batch, input_embeds): pass", forward.body, ns)
    admission = find(runner, ast.FunctionDef, lambda n:n.name == "can_run_graph")
    assert "consumes_verify_context" in ast.unparse(admission.body[0])
    consumed = wrap("def consumed(self, forward_batch): pass", [admission.body[0], ast.parse("return True").body[0]],ns)
    return ns["DFlashVerifyFCGraph"], Full, slice_live, materialize, retire, body, consumed


class Layer(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.self_attn_input_norm = lambda h:h

    def forward(self, pos, stream, attn, fb):
        return stream + self.delta, attn


def run(rank, size, stage, directory, witness):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory) / f"sync{size}"),
                            rank=rank, world_size=size)
    cls, Full, slice_live, materialize, retire, body, consumed = api(stage)
    torch.manual_seed(145)
    draft, processor, target = Draft(), torch.nn.Identity(), torch.nn.Identity()
    target.model = torch.nn.Module()
    target.model.language_model = torch.nn.Identity()
    processor.keep_last_hidden_states = False
    processor.expected_aux_hidden_states = None
    group = Group(rank, size)
    from collections import namedtuple
    context_type = namedtuple("Context", "draft fc norm project_target_hidden group target logits_processor")
    context = context_type(draft, draft.fc, draft.hidden_norm, draft.project_target_hidden, group, target, processor)
    with torch.no_grad():
        draft.fc.weight.copy_(torch.randint(-3, 4, draft.fc.weight.shape).to(torch.bfloat16) / 4)
    verify = cls(context)
    worker = NS(_verify_fc_graph=verify, draft_model=draft)
    runner = NS(backend=Full(), model_runner=NS(capture_tail_hooks=[], capture_head_hooks=[]),
                ragged_verify_mode=False, enable_pdmux=False, enable_two_batch_overlap=False,
                enable_torch_compile=False, max_num_token=256)
    batch = NS(forward_mode=NS(is_target_verify=lambda:True))
    model = NS(residual_dtype=torch.float32, operator_dtype=torch.bfloat16,
               layers=[Layer((i+1)/4) for i in range(5)], layers_to_capture=list(range(5)), norm=lambda x:x)
    witness_hash = None
    if witness:
        payload = json.loads(Path(witness).read_text())["witness"]
        data = torch.tensor(payload["words"],dtype=torch.int32).to(torch.int16)
        witness_hash = hashlib.sha256(data.numpy().tobytes()).hexdigest()
        assert witness_hash == payload["sha256"]
        witness_tensor = data.view(torch.bfloat16).view(-1,16)
    counts = {"epoch":0,"key":0,"gathers":0,"outputs":0,"raw":0,"counterexamples":0}
    with torch.inference_mode():
        # Unknown tail/head instrumentation and unsupported graph schedules on a single peer.
        for attr in ("capture_tail_hooks","capture_head_hooks","enable_pdmux","ragged_verify_mode",
                     "enable_two_batch_overlap","enable_torch_compile"):
            obj = runner.model_runner if attr.endswith("hooks") else runner
            old = getattr(obj, attr)
            if rank == size-1:
                setattr(obj,attr,[lambda *a:None] if attr.endswith("hooks") else True)
            assert verify.prepare_capture(runner) is None
            setattr(obj,attr,old)
            counts["epoch"]+=1
        for kind in ("new_empty","clone"):
            original = getattr(torch.Tensor, kind)
            def allocate(tensor,*args,**kwargs):
                if rank==size-1: raise torch.OutOfMemoryError("one rank only")
                return original(tensor,*args,**kwargs)
            with patch.object(torch.Tensor,kind,allocate):
                assert verify.prepare_capture(runner) is None
            assert verify.local_weight is None and verify.raw is None
        if rank==1:
            target.model.language_model.forward=lambda x:x.neg_()
        assert verify.prepare_capture(runner) is None
        if rank==1: del target.model.language_model.forward
        assert verify.prepare_capture(runner) is verify
        # Failed growth retains previous published graph storage on every rank.
        old=verify.raw
        runner.max_num_token*=2
        original=torch.Tensor.new_empty
        def grow(tensor,*args,**kwargs):
            if rank==size-1: raise torch.OutOfMemoryError("growth")
            return original(tensor,*args,**kwargs)
        with patch.object(torch.Tensor,"new_empty",grow):
            assert verify.prepare_capture(runner) is None
        assert verify.raw is old
        runner.max_num_token//=2
        counts["allocation_rollbacks"]=3
        assert verify.local_weight.untyped_storage().data_ptr() != draft.fc.weight.untyped_storage().data_ptr()
        runner._verify_fc_capture = verify
        assert consumed(runner, NS(spec_info=NS(consumes_verify_context=True)))
        assert not consumed(runner, NS(spec_info=NS(consumes_verify_context=False)))
        assert not consumed(runner, NS(spec_info=None))
        assert consumed(NS(), NS(spec_info=None))
        # Duplicates in a custom capture list cannot write a compact tensor
        # with a fictitious fifth tap. Stock returns the actual four captures.
        batch.dflash_verify_raw_buffer=verify.raw[:8]
        model.layers_to_capture=[0,0,1,2,3]
        _, duplicate_taps=body(model,None,None,batch,torch.ones(8,16))
        assert isinstance(duplicate_taps,list) and len(duplicate_taps)==4
        model.layers_to_capture=list(range(5))
        initial = verify.raw.untyped_storage().data_ptr()
        assert verify.prepare_capture(runner) is verify and verify.raw.untyped_storage().data_ptr()==initial
        for capacity, live in ((8,8),(32,24),(64,40),(128,104),(256,248),(256,8),(128,128)):
            inputs = torch.randint(-5,6,(capacity,16)).float()
            if witness and (capacity,live)==(128,128):
                inputs = witness_tensor.float()
            batch.dflash_verify_raw_buffer = None
            native_hidden, taps = body(model,None,None,batch,inputs)
            native_raw = torch.cat(taps,dim=-1)
            batch.dflash_verify_raw_buffer = verify.raw[:capacity]
            target_hidden, packed = body(model,None,None,batch,inputs)
            assert target_hidden.equal(native_hidden) and packed is batch.dflash_verify_raw_buffer
            assert torch.equal(packed.view(torch.int16),native_raw.view(torch.int16))
            counts["raw"] += packed.numel()
            out = Output(packed)
            verify.capture(out,batch)
            verify.complete_capture(out)
            assert out.hidden_states is packed and out.dflash_verify_fc_shard.shape == (capacity,32//size)
            trimmed = slice_live(NS(raw_num_token=live),out)
            got = materialize(worker,trimmed.hidden_states,trimmed.dflash_verify_fc_shard)
            expected = draft.project_target_hidden(native_raw[:live])
            assert torch.equal(got.view(torch.int16), expected.view(torch.int16))
            assert group.payloads[-1] == [live,32//size]
            counts["outputs"] += got.numel()
            counts["key"]+=1
        inputs = torch.arange(8*80,dtype=torch.float32).view(8,80).to(torch.bfloat16)
        batch.dflash_verify_raw_buffer=verify.raw[:8]
        batch.dflash_verify_raw_buffer.copy_(inputs)
        out=Output(batch.dflash_verify_raw_buffer)
        verify.capture(out,batch)
        verify.complete_capture(out)
        baseline_weight = draft.fc.weight.clone()
        for case in ("hook","replaced_data","inplace","partial_loader"):
            handle=None
            if rank==size-1:
                if case == "hook":
                    handle=draft.fc.register_forward_hook(lambda _m,_a,v: v.neg())
                elif case == "replaced_data":
                    saved_data=draft.fc.weight.data
                    draft.fc.weight.data=saved_data.neg()
                elif case == "inplace":
                    draft.fc.weight.neg_()
                else:
                    retire(worker)
                    draft.fc.weight.data[:3].add_(2)
            got=materialize(worker,out.hidden_states,out.dflash_verify_fc_shard)
            expected=draft.project_target_hidden(inputs)
            assert got.equal(expected),case
            if rank==size-1:
                if handle is not None:
                    handle.remove()
                if case == "replaced_data":
                    draft.fc.weight.data=saved_data
                if case == "inplace":
                    draft.fc.weight.copy_(baseline_weight)
            counts["counterexamples"]+=1
        # One bad alias must turn off the optimization for this captured key everywhere.
        mismatch = Output(out.hidden_states.clone() if rank==1 else out.hidden_states)
        verify.capture(mismatch,batch)
        verify.complete_capture(mismatch)
        assert mismatch.dflash_verify_fc_shard is None
        assert materialize(worker,mismatch.hidden_states,None).equal(draft.project_target_hidden(inputs))
        # Real inference Parameters have no version. Exercise source .data writes
        # after a supported pre-write invalidation on a single rank.
        inference_draft=Draft()
        assert torch.is_inference(inference_draft.fc.weight)
        inference_draft.fc.weight.data.copy_(baseline_weight)
        inf_context=context._replace(draft=inference_draft,fc=inference_draft.fc,norm=inference_draft.hidden_norm,
                                     project_target_hidden=inference_draft.project_target_hidden)
        inf=cls(inf_context)
        assert inf.version is None and inf.prepare_capture(runner) is inf
        inf_worker=NS(_verify_fc_graph=inf, draft_model=inference_draft)
        batch.dflash_verify_raw_buffer=inf.raw[:8]
        batch.dflash_verify_raw_buffer.copy_(inputs)
        observed=Output(batch.dflash_verify_raw_buffer)
        inf.capture(observed,batch)
        inf.complete_capture(observed)
        if rank==1:
            retire(inf_worker)
            inference_draft.fc.weight.data[:3].neg_()
        got=materialize(inf_worker,observed.hidden_states,observed.dflash_verify_fc_shard)
        assert got.equal(inference_draft.project_target_hidden(inputs))
        counts["counterexamples"]+=1
        # Time the rejected per-verify Gloo gate on this host only.
        from torch.distributed import ReduceOp
        stat=torch.ones((),dtype=torch.int32)
        times=[]
        for i in range(48):
            start=time.perf_counter_ns()
            dist.all_reduce(stat,op=ReduceOp.MIN,group=group.cpu_group)
            if i>=8: times.append((time.perf_counter_ns()-start)/1000)
        counts["gathers"]=len(group.payloads)
    Path(directory,f"{size}-{rank}.json").write_text(json.dumps(dict(ranks=size,rank=rank,
        counts=counts,obsolete_gloo_median_us=median(times),witness=witness_hash)))
    dist.destroy_process_group()


if __name__ == "__main__":
    reports = []
    with tempfile.TemporaryDirectory() as directory:
        for size in (4,8):
            mp.spawn(run, args=(size, sys.argv[1], directory, sys.argv[2] if len(sys.argv)>2 else None), nprocs=size)
            reports.extend(json.loads(Path(directory, f"{size}-{r}.json").read_text()) for r in range(size))
    print(json.dumps(dict(physical_h200=False, torch_version=str(torch.__version__), devices="CPU Gloo",
        compared_bf16_values=sum(r["counts"]["outputs"] for r in reports), rank_cases=reports), indent=2))
