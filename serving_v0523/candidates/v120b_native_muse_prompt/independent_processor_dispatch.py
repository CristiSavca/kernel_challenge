"""Regression for the public Muse text processor's actual inheritance and worker discovery."""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
PIN = '297e54fd105e974cdf401e5b4d20253a22c16f90'


def pinned(rel):
    return subprocess.check_output(['git', '-C', ROOT / 'sglang', 'show', PIN + ':python/sglang/srt/' + rel], text=True)


def class_node(rel, name):
    return next(x for x in ast.parse(pinned(rel)).body if getattr(x, 'name', '') == name)


def define(nodes, env):
    exec('from __future__ import annotations\n' + ast.unparse(ast.Module(body=nodes, type_ignores=[])), env)


def add_stub(name, **kwargs):
    mod = types.ModuleType(name)
    mod.__dict__.update(kwargs)
    sys.modules[name] = mod


def from_patch():
    p = ROOT / 'candidates/v120_phase_parallel_draft_context/patch.diff'
    body = p.read_text().split('diff --git a/python/sglang/srt/layers/dflash_context_projection.py ')[1].split('\ndiff --git ')[0]
    return '\n'.join(line[1:] for line in body.splitlines() if line.startswith('+') and not line.startswith('+++')) + '\n'


def exercise(source):
    ns = {'nn': nn, 'torch': torch}
    base = class_node('layers/logits_processor.py', 'LogitsProcessor')
    # Retain production forward/hidden methods for identity guards; external GPU setup is unnecessary.
    base.body = [
        x for x in base.body if isinstance(x, ast.FunctionDef) and x.name in ('forward', '_get_pruned_states', '_get_hidden_states_to_store', '_get_last_hidden_states_to_store', '_get_logits', '_check_aux_hidden_states')]
    define([base], ns)
    cls = ns['LogitsProcessor']
    cls.__init__ = lambda self, *a, **k: nn.Module.__init__(self)
    ns['OSSLogitsProcessor'] = cls
    muse = class_node('models/muse_spark.py', 'MuseSpark1xLogitsProcessor')
    assert {n.name for n in muse.body if isinstance(n, ast.FunctionDef)} == {'__init__', '_compute_lm_head'}
    define([muse], ns)
    muse_cls = ns[muse.name]
    text = class_node('models/muse_spark.py', 'MuseSparkTextForCausalLM')
    ctor = next(n for n in text.body if getattr(n, 'name', '') == '__init__')
    assign = next(n for n in ast.walk(ctor) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'self.logits_processor')
    omni = class_node('models/muse_spark_omni.py', 'MuseSparkForConditionalGeneration')
    assert ast.unparse(omni.bases[0]) == text.name
    assert 'super().__init__(config.text_config, quant_config, prefix)' in ast.unparse(omni)
    assert 'super().forward(' in ast.unparse(omni)
    for path in ('models/muse_spark_omni.py', 'layers/muse_spark/muse_spark_audio.py', 'layers/muse_spark/muse_spark_navit.py'):
        assert 'LogitsProcessor' not in pinned(path)
    target = nn.Module()
    ns.update(self=target, config=types.SimpleNamespace(output_multiplier=0.17677669529663687))
    define([assign], ns)
    processor = target.logits_processor
    assert type(processor) is muse_cls and isinstance(processor, cls) and type(processor) is not cls
    worker = class_node('speculative/dflash_worker_v2.py', 'DFlashWorkerV2')
    discover = next(n for n in worker.body if getattr(n, 'name', '') == '_defer_target_verify_gather')
    define([discover], ns)
    fixture = types.SimpleNamespace(target_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(model=target)))
    assert ns[discover.name](fixture) is processor and processor.defer_verify_gather
    target.extra = cls()
    assert ns[discover.name](fixture) is None
    del target.extra
    class Draft(nn.Module):
        quantizable = False
        def project_target_hidden(self, x):
            return x
    draft = Draft()
    draft.fc = nn.Linear(40, 8, bias=False, dtype=torch.bfloat16)
    add_stub('sglang.srt.models.dflash', DFlashDraftModel=Draft)
    add_stub('sglang.srt.layers.logits_processor', LogitsProcessor=cls)
    add_stub('sglang.srt.models.muse_spark', MuseSpark1xLogitsProcessor=muse_cls)
    helper = types.ModuleType('candidate_projection')
    exec(source, helper.__dict__)
    helper._agree = lambda acceptable, group: bool(acceptable)
    group = types.SimpleNamespace(world_size=2)
    processor.keep_last_hidden_states = False
    processor.expected_aux_hidden_states = None
    with patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)), patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)):
        def eligible(processor):
            return helper.prepare_dflash_sp_prefill(draft, target, processor, group) is not None
        base_yes, muse_yes = eligible(cls()), eligible(processor)
        class Altered(muse_cls):
            def forward(self, *args, **kwargs):
                return None
        custom = eligible(Altered(None))
        overridden = []
        if muse_yes:
            handle = helper.prepare_dflash_sp_prefill(draft, target, processor, group)
            rows = torch.ones((128, 40), dtype=torch.bfloat16)
            with helper.dflash_sp_prefill_scope(handle), patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
                project = lambda: helper.project_dflash_prefill_shard(rows, 256, group, processor, True)
                assert torch.equal(project(), draft.fc(rows))
                for name in ('forward', '_get_pruned_states', '_get_hidden_states_to_store', '_get_last_hidden_states_to_store'):
                    setattr(processor, name, lambda *params: None)
                    assert project() is None, name
                    overridden.append(name)
                    delattr(processor, name)
    assert base_yes and not custom
    return {'base_processor_eligible':base_yes,'muse_processor_eligible':muse_yes,'overridden_forward_eligible':custom,
            'disabled_instance_overrides':overridden,'real_dflash_worker_discovers_muse':True,'multiple_processors_safely_disabled':True,
            'head_class':type(processor).__name__,'only_muse_overrides':['__init__','_compute_lm_head'],
            'source_sha256':hashlib.sha256(source.encode()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, help='Also test a corrected stage')
    args = parser.parse_args()
    result = {'frozen_v120':exercise(from_patch())}
    assert not result['frozen_v120']['muse_processor_eligible']
    if args.stage:
        result['fixed'] = exercise((args.stage / 'python/sglang/srt/layers/dflash_context_projection.py').read_text())
        assert result['fixed']['muse_processor_eligible']
    print(json.dumps(result, indent=2))
