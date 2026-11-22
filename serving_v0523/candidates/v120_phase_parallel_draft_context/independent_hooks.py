"""Two real Gloo peers compare the production verify helper to native rank-local hooks."""
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


SCENARIOS = (
    'clean', 'prehook', 'posthook', 'instance_forward', 'replace_fc',
    'replace_weight', 'projector_override', 'hidden_norm_hook', 'global_pre_hook',
    'deterministic', 'noncontiguous', 'autocast', 'autocast_fp32',
)


def production_model(source):
    path = source / 'models/dflash.py'
    model = next(n for n in ast.parse(path.read_text()).body if getattr(n, 'name', '') == 'DFlashDraftModel')
    project = next(n for n in model.body if getattr(n, 'name', '') == 'project_target_hidden')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[project], type_ignores=[]), str(path), 'exec'), ns)
    cls = type('DFlashDraftModel', (nn.Module,), {'project_target_hidden': ns['project_target_hidden']})
    stub = types.ModuleType('sglang.srt.models.dflash')
    stub.DFlashDraftModel = cls
    sys.modules[stub.__name__] = stub
    return cls


def load_helper(source):
    path = source / 'layers/dflash_context_projection.py'
    spec = importlib.util.spec_from_file_location('review_projection', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Group:
    def __init__(self, rank):
        self.rank_in_group = rank
        self.world_size = 2
        self.cpu_group = dist.group.WORLD
        self.device_group = dist.group.WORLD
        self.calls = 0

    def all_gather(self, local, dim):
        assert dim == -1
        self.calls += 1
        # Gloo writes on a different thread. Allocate ordinary CPU destinations.
        with torch.inference_mode(False), torch.no_grad():
            x = local.detach().clone()
            tensors = [torch.empty_like(x) for _ in range(self.world_size)]
            dist.all_gather(tensors, x, group=self.device_group)
        return torch.cat(tensors, dim)


def run(rank, rendezvous, output_dir, source):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=12))
    helper = load_helper(source)
    model = production_model(source)
    group = Group(rank)
    results = []
    with torch.inference_mode():
        for owner in (0, 1):
            for mode in SCENARIOS:
                features, channels, rows = 40, 8, 7
                draft = model()
                draft.quantizable = False
                draft.num_context_features = 5
                draft.config = types.SimpleNamespace(hidden_size=channels)
                draft.fc = nn.Linear(features, channels, bias=False, dtype=torch.bfloat16)
                draft.fc.weight.copy_((torch.arange(channels * features).reshape(channels, -1) % 11 - 5) / 16)
                draft.hidden_norm = nn.Identity()
                with patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)), \
                        patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)):
                    handle = helper.prepare_dflash_verify_projection(draft, group)
                assert handle is not None, (rank, mode, 'startup')
                base = torch.arange(rows * 2 * features).reshape(rows, 2 * features)
                inputs = ((base % 7) - 3).to(torch.bfloat16)[:, ::2]
                if mode != 'noncontiguous':
                    inputs = inputs.contiguous()
                hook_calls = []
                def record(output):
                    hook_calls.append(1)
                    return output
                fc = draft.fc
                with ExitStack() as edits:
                    if rank == owner:
                        if mode == 'prehook':
                            edits.enter_context(fc.register_forward_pre_hook(lambda _, ins: (record(ins[0] + 2),)))
                        elif mode == 'posthook':
                            edits.enter_context(fc.register_forward_hook(lambda _, ins, out: record(out + 3)))
                        elif mode == 'instance_forward':
                            orig = fc.forward
                            fc.forward = lambda x: record(orig(x + 1) + 5)
                        elif mode == 'replace_fc':
                            draft.fc = nn.Linear(features, channels, bias=False, dtype=torch.bfloat16)
                            draft.fc.weight.copy_(fc.weight + 1)
                        elif mode == 'replace_weight':
                            fc.weight = nn.Parameter(fc.weight + 2, requires_grad=False)
                        elif mode == 'projector_override':
                            orig = draft.project_target_hidden
                            draft.project_target_hidden = lambda x: record(orig(x) * 3)
                        elif mode == 'hidden_norm_hook':
                            edits.enter_context(draft.hidden_norm.register_forward_hook(lambda _, ins, out: record(out - 2)))
                        elif mode == 'global_pre_hook':
                            def change(mod, ins):
                                if mod is fc:
                                    return (record(ins[0] - 2),)
                            edits.enter_context(nn.modules.module.register_module_forward_pre_hook(change))
                        elif mode in ('autocast', 'autocast_fp32'):
                            edits.enter_context(torch.autocast('cpu', dtype=torch.bfloat16))
                            if mode == 'autocast_fp32':
                                inputs = inputs.float()
                        elif mode == 'deterministic':
                            torch.use_deterministic_algorithms(True)
                            edits.callback(torch.use_deterministic_algorithms, False)
                    reference = draft.project_target_hidden(inputs)
                    expected_calls = len(hook_calls)
                    hook_calls.clear()
                    before = group.calls
                    actual = helper.project_dflash_verify(handle, inputs)
                    assert torch.equal(reference, actual), (rank, owner, mode, (actual - reference).abs().max())
                    assert len(hook_calls) == expected_calls, (rank, owner, mode, hook_calls, expected_calls)
                    assert group.calls == before + 1, (rank, owner, mode, 'collective count', group.calls - before)
                results.append([owner, mode])
    (Path(output_dir) / ('rank-' + str(rank))).write_text(json.dumps(results))
    dist.destroy_process_group()


if __name__ == '__main__':
    source = Path(sys.argv[1]).resolve() / 'python/sglang/srt'
    with tempfile.TemporaryDirectory(prefix='v120-hook-review-') as tmp:
        mp.spawn(run, args=(str(Path(tmp) / 'rdzv'), tmp, source), nprocs=2, join=True)
        a, b = [json.loads((Path(tmp) / ('rank-' + str(r))).read_text()) for r in (0, 1)]
        assert a == b and len(a) == 2 * len(SCENARIOS)
    print(json.dumps({'framework': torch.__version__, 'real_gloo_ranks': 2,
                      'rank_local_cases': 2 * len(a), 'one_collective_per_case_per_rank': True,
                      'scenarios': SCENARIOS,
                      'production_helper_sha256': hashlib.sha256((source / 'layers/dflash_context_projection.py').read_bytes()).hexdigest()}, indent=2))
