"""Run the real mixed scheduler with live prepared views, CPU model bodies and real TP Gloo."""
import ast
from contextlib import nullcontext
from datetime import timedelta
from hashlib import sha256
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.modules import module as module_hooks

HERE = Path(__file__).resolve().parent
SRT = Path(sys.argv[1]) / "python/sglang/srt"
MIXED = "speculative/dflash_mixed_forward.py"
PARENT_PATCH = HERE.parent / "v206_steady_decode_gate_up_pipeline/patch.diff"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_mixed():
    # This component was a new file in the V206 base-relative export.
    raw = PARENT_PATCH.read_bytes()
    assert sha256(raw).hexdigest() == "1919f7bd4dec48dd03da35ccd8eea2f4a13096c887b964c904b90cc5be40c6ce"
    rel = "python/sglang/srt/" + MIXED
    diff = raw.decode().split(f"diff --git a/{rel} b/{rel}\n", 1)[1].split("diff --git ", 1)[0]
    assert "new file mode" in diff
    return "".join(line[1:] for line in diff.splitlines(keepends=True)
                   if line.startswith("+") and not line.startswith("+++")).encode()


def mixed_functions(raw, namespace):
    tree = ast.parse(raw)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(namespace)
    exec(compile(ast.Module(functions, []), MIXED, "exec"), ns)
    return ns


def check_rank(rank, size, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(folder) / "init"),
                            rank=rank, world_size=size, timeout=timedelta(seconds=90))
    try:
        run_rank(rank, size, folder)
    finally:
        dist.destroy_process_group()


def run_rank(rank, size, folder):
    test = load("scheduler", HERE.parent / "v160_mixed_dflash_serving/independent_round.py")
    qa = load("context", HERE.parent / "v202_mixed_target_serving/independent_mixed.py")
    vocab = load("vocab", HERE.parent / "v203_mixed_sharded_verify_head/independent_gloo.py")
    # Later fixture initializers install their private stand-ins under the same module names.
    test.sched.module("sglang.srt.speculative.dflash_mixed_forward",
                      start_mixed_prefix_readback=test.bridge.env["start_mixed_prefix_readback"])
    target_module = sys.modules["sglang.srt.speculative.dflash_mixed_forward"]
    ref = test.ref
    ref.env.update(project_dflash_prefill_shard=qa.projection.project_dflash_prefill_shard,
                   project_paired_draft_kv=qa.reject_paired,
                   get_forward=lambda: NS(set=lambda *_: None, scoped=lambda **_: nullcontext()))
    test.admit["dist"] = dist
    old = mixed_functions(parent_mixed(), test.bridge.env)
    new = mixed_functions((SRT / MIXED).read_bytes(), {**test.bridge.env,
                          "module_hooks": module_hooks,
                          "dflash_sp_prefill_scope": qa.projection.dflash_sp_prefill_scope})
    spy, observations, pairs = {}, [], {}
    def gather(out, val):
        with torch.inference_mode(False):
            parts = [torch.empty(tuple(val.shape), dtype=val.dtype) for _ in range(size)]
            dist.all_gather(parts, val.clone().contiguous())
        out.view(size, *val.shape).copy_(torch.stack(parts))
        spy["collectives"].append(list(val.shape))

    group = NS(world_size=size, rank_in_group=rank, cpu_group=dist.group.WORLD,
               pynccl_comm=None, all_gather_into_tensor=gather)
    original_policy = test.pl.model
    def graph_policy(grids):
        policy = original_policy((list(range(1, 2049)), grids[1]))
        test.pl.PARALLEL.tp_group = group
        return policy
    test.pl.model = graph_policy
    ref.PAR.tp_group = group
    # Use the real distributed agreement in the helper; model only the CUDA-specific norm backend.
    qa.projection._supports_shard_norm = lambda *_: scenario != "unsupported_norm"
    gpu_ops = vocab.build_processor(group)
    old_prepare = test.settle.prepare
    # Only fixture data changes: synthetic lengths are selected before it allocates its slots.
    constructor = ast.parse(inspect.getsource(old_prepare))
    length_assignment = next(n for n in ast.walk(constructor) if isinstance(n, ast.Assign)
                             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "lengths")
    length_assignment.value = ast.Name(id="prompt_lengths", ctx=ast.Load())
    prepare_ns = dict(vars(test.settle))
    exec(compile(ast.fix_missing_locations(constructor), "<synthetic scheduler input>", "exec"), prepare_ns)
    def prepare(backend, tails, iteration, do_veto):
        prepare_ns["prompt_lengths"] = lengths
        result = prepare_ns["prepare"](backend, tails, iteration, do_veto)
        worker, prompt, decode, schedule, _, _, _, steps = result
        target = worker.target_worker
        runner = target.model_runner
        model = runner.model
        graph = inspect.getclosurevars(runner.forward).nonlocals["g"]
        proc = model.logits_processor.__self__
        head = qa.NativeHead(proc.forward)
        model.logits_processor = head
        draft = qa.Draft(worker.draft_model)
        worker.draft_model = draft
        marker = torch.nn.Module()
        worker._sp_prefill_context = (draft, draft.fc, draft.hidden_norm, draft.project_target_hidden,
                                      group, marker, head) if scenario != "context_off" else None
        if scenario == "local_hook":
            draft.fc.register_forward_hook(lambda *_: None)
        if scenario == "rank_fc_hook" and rank == size-1:
            draft.fc.register_forward_pre_hook(lambda *_: None)
        for attr in ("is_deferred_shard", "finish_deferred_gather", "deferred_argmax"):
            setattr(head, attr, getattr(gpu_ops, attr))
        # The AST-extracted Muse wrapper has its torch base stripped by the legacy fixture.
        model._forward_hooks = marker._forward_hooks
        model._forward_pre_hooks = marker._forward_pre_hooks
        worker._target_logits_processor = None if scenario == "head_off" else head
        if scenario == "rank_head" and rank == size-1:
            worker._target_logits_processor = None
        def inspect_logits(mod, args, out):
            if mod is head:
                assert out.next_token_logits.shape[1] == ref.V
                spy["width_seen"] = out.next_token_logits.shape[1]
        if scenario == "head_hook" or (scenario == "rank_head_hook" and rank == size-1):
            head.register_forward_hook(inspect_logits)
        if scenario == "head_pre_hook":
            head.register_forward_pre_hook(lambda *_: None)
        if scenario in ("wrapped_head", "compiled_head"):
            raw_forward = head.forward
            def wrapped(*args, **kwargs):
                result = raw_forward(*args, **kwargs)
                inspect_logits(head, args, result)
                return result
            if scenario == "wrapped_head":
                head.forward = wrapped
            else:
                head._compiled_call_impl = wrapped
        if scenario == "global_hook":
            spy["handle"] = module_hooks.register_module_forward_hook(inspect_logits)
        if scenario == "global_pre_hook":
            spy["handle"] = module_hooks.register_module_forward_pre_hook(lambda *_: None)
        if scenario == "target_hook":
            marker.register_forward_pre_hook(lambda *_: None)
        width = math.ceil(ref.V/size)
        gpu_ops.defer_verify_gather = True
        gpu_ops.final_logit_softcapping = 25.
        gpu_ops.vocab_size = ref.V
        proc.do_tensor_parallel_all_gather = True
        proc.defer_verify_gather = True
        proc._can_gather_softcap_fused = lambda *_: True  # CPU device substitution
        proc._gather_softcap_fused = gpu_ops._gather_softcap_fused
        proc._compute_lm_head = lambda h,w,b,out: torch.nn.functional.pad(
            h.float() @ w.T, (0,width*size-ref.V))[:,rank*width:(rank+1)*width].contiguous()
        attn = draft.layers[0].self_attn
        attn.kv_proj_only = lambda x: (x.repeat(1, 5), (2*x).repeat(1, 5))
        state = {}
        def shard(whole):
            span = math.ceil(len(whole)/size)
            return torch.nn.functional.pad(whole,(0,0,0,span*size-len(whole)))[rank*span:(rank+1)*span]
        def body(ids):
            h, taps = ref.body_tokens(ids)
            state.update(hidden=h, raw=torch.cat(taps,dim=1))
            return shard(h), [shard(tap) for tap in taps]
        def materialize(local, num_tokens):
            raw = state["raw"]
            if local.shape[1] == raw.shape[1]:
                full = raw
            elif torch.equal(local, shard(state["hidden"])):
                full = state["hidden"]
            else:
                full = draft.hidden_norm(draft.fc(raw))
            assert torch.equal(local, shard(full))
            return full[:num_tokens]
        def select(local, positions, count):
            assert torch.equal(local, shard(state["hidden"]))
            return state["hidden"][:count][positions]
        ref.env["layernorm_sp"] = NS(runs_sp=lambda mode: mode == ref.Mode.EXTEND,
                                     sp_exit_gather=materialize, sp_gather_selected_rows=select,
                                     maybe_exit_gather=lambda **kw: (state["hidden"][:len(kw["input_ids"])],
                                                                     kw["hidden_states_before_norm"]))
        graph.layer_model.forward = lambda ids,*_,**__: body(ids)
        graph.backend.replay = lambda key,fb,**kw: body(fb.input_ids)
        original = target.forward_batch_generation
        def target_call(*args, **kwargs):
            assert kwargs["batch"] is None and kwargs["is_verify"]
            fb = kwargs["forward_batch"]
            assert qa.projection._draft_kv_helper.get() is None
            assert (qa.projection._prefill_context.get() is not None) == (fixed and scenario != "context_off")
            out = original(*args, **kwargs)
            logits = out.logits_output
            assert not logits.dflash_prefill_kv_projected
            spy.update(request=fb.mixed_verify_sharded_logits, projected=logits.dflash_prefill_context_done,
                       head_width=logits.next_token_logits.shape[1], hidden_width=logits.hidden_states.shape[1],
                       rows=len(fb.input_ids))
            raw = torch.cat(ref.body_tokens(fb.input_ids)[1], dim=-1)
            expected_hidden = draft.hidden_norm(draft.fc(raw)) if spy["projected"] else raw
            assert torch.equal(logits.hidden_states, expected_hidden)
            spy["positions"] = fb.positions
            return out
        target.forward_batch_generation = target_call
        worker.forward_mixed_verify = lambda *_,**__: (_ for _ in ()).throw(AssertionError("unused worker helper reached"))
        spy.update(worker=worker, schedule=schedule, q=sum(lengths), original_pool=worker.draft_model_runner.token_to_kv_pool.k_buffer[0].clone())
        return result
    test.settle.prepare = prepare
    cases = [("long", [130*size+1, 7]), ("short", [7, 11]), ("head_off", [130*size+3,5]),
             ("context_off", [130*size+5,7]), ("local_hook", [130*size+7,2]),
             ("rank_fc_hook", [130*size+2,9]), ("unsupported_norm", [130*size+9,6]),
             ("head_hook", [130*size+3, 9]), ("wrapped_head", [130*size+9,3]),
             ("global_hook", [130*size+4,8]), ("target_hook", [130*size+5,2]),
             ("head_pre_hook", [130*size+6,9]), ("compiled_head", [130*size+7,3]),
             ("global_pre_hook", [130*size+5,8])]
    for backend in ("eager", "FULL", "BREAKABLE"):
        for j,(scenario,lengths) in enumerate(cases):
            for fixed in (False, True):
                spy.clear()
                spy["collectives"]=[]
                target_module.try_forward_proposed_mixed = (new if fixed else old)["try_forward_proposed_mixed"]
                # Scheduler run_batch executes the selected proposal then publishes and resolves its next round.
                with torch.inference_mode():
                    report = test.replay(backend, 2, j, j % 2 == 0)
                    worker, schedule, q = spy["worker"], spy["schedule"], spy["q"]
                    desired = torch.tensor([1+(j+3*i)%8 for i in range(2)])
                    raw = torch.cat(ref.body_tokens(schedule.input_ids)[1],dim=-1)
                    normalized = worker.draft_model.hidden_norm(worker.draft_model.fc(raw)).repeat(1,5)
                    live = torch.cat((torch.ones(q,dtype=torch.bool),(torch.arange(8)<desired[:,None]).flatten()))
                    slots = schedule.out_cache_loc[live]
                    expected_k=spy["original_pool"].clone()
                    expected_v=spy["original_pool"].clone()
                    expected_k[slots]=(normalized+spy["positions"][:,None].to(normalized.dtype))[live]
                    expected_v[slots]=(2*normalized)[live]
                    pool=worker.draft_model_runner.token_to_kv_pool
                    assert torch.equal(expected_k,pool.k_buffer[0]) and torch.equal(expected_v,pool.v_buffer[0])
                    assert qa.projection._prefill_context.get() is None and qa.projection._draft_kv_helper.get() is None
                if scenario in ("head_hook", "global_hook", "wrapped_head", "compiled_head"):
                    assert spy["width_seen"] == ref.V
                if "handle" in spy:
                    spy["handle"].remove()
                local=fixed and scenario in ("long", "short", "context_off", "local_hook",
                                             "rank_fc_hook", "unsupported_norm")
                fast=fixed and scenario in ("long","head_off")
                assert spy["request"] == local and spy["projected"] == fast, (backend,scenario,fixed,spy["projected"])
                sw=math.ceil(ref.V/size)
                assert spy["head_width"] == (sw if local else ref.V) and spy["hidden_width"] == (8 if fast else 40)
                assert spy["collectives"] == ([[len(lengths),sw],[16,2]] if local else [[len(lengths)+16,sw]])
                # The original fixture checked prompt/bonus ids, accept lengths, future-map filtering and rows.
                key=(backend,scenario)
                digest=sha256(expected_k.view(torch.int16).numpy().tobytes()+expected_v.view(torch.int16).numpy().tobytes()).hexdigest()
                if fixed:
                    assert pairs[key] == (report,digest)
                else:
                    pairs[key]=(report,digest)
                observations.append(dict(backend=backend,case=scenario,fixed=fixed,rows=spy["rows"],
                    head_width=spy["head_width"],hidden_width=spy["hidden_width"],collectives=spy["collectives"],
                    committed=report[2],cache_words=2*expected_k.numel()))
    # On discrepant head availability or instrumentation the native cohort vote
    # preserves the proposal for ordinary verification on all ranks.
    for backend in ("FULL", "BREAKABLE"):
        for scenario in ("rank_head", "rank_head_hook"):
            lengths = [130*size+5,7]
            spy.clear()
            spy["collectives"] = []
            fixed = True
            target_module.try_forward_proposed_mixed = new["try_forward_proposed_mixed"]
            try:
                with torch.inference_mode():
                    test.replay(backend,2,1,False)
            except AssertionError as exc:
                assert str(exc) == "unexpected standalone", str(exc)
            else:
                raise AssertionError("divergent cohort reached combined target")
            assert "rows" not in spy and not spy["collectives"]
            assert spy["worker"]._pending_proposal_token is not None
            observations.append(dict(backend=backend,decline=scenario))
    if rank == 0:
        Path(folder,"out.json").write_text(json.dumps(observations))


def main():
    by_world={}
    for size in (4,8):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(check_rank,args=(size,folder),nprocs=size,join=True)
            results=json.loads(Path(folder,"out.json").read_text())
            declines = [r for r in results if "decline" in r]
            results = [r for r in results if "decline" not in r]
            by_world[size]=dict(rounds=len(results), reference_pairs=len(results)//2, agreed_declines=declines,
                                per_rank_cache_words=sum(r["cache_words"] for r in results),
                                projection_hits=sum(r["hidden_width"]==8 for r in results),
                                local_head_hits=sum(r["head_width"]<43 for r in results),
                                examples=[r for r in results if r["backend"]=="FULL" and r["case"]=="long"])
    print(json.dumps(dict(torch=torch.__version__,by_world=by_world,executed_parent_sha256=sha256(parent_mixed()).hexdigest(),
                          real_gloo=True,physical_H200=False),indent=2))


if __name__ == "__main__":
    main()
