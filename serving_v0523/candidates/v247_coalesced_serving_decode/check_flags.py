"""Check the real msgspec CLI registration and positional fields on the complete frozen release."""
import ast
import hashlib
import importlib.util
import inspect
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
m = json.loads((HERE / 'validation.json').read_bytes())
assert m['candidate'] == 'V247'
digest = hashlib.sha256((HERE / 'patch.diff').read_bytes()).hexdigest()
assert digest == m['patch_sha256']
assert subprocess.check_output(['git','-C',str(STAGE),'write-tree'],text=True).strip() == m['tree']
fixture = HERE.parent / 'v176_integrated_prompt_decode/independent_registration.py'
spec = importlib.util.spec_from_file_location('registration',fixture)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
original_source = cli.source
hashes = {}
def read(tree, path):
    text = original_source(tree, path)
    if tree == m['tree']:
        staged = STAGE / 'python/sglang/srt' / path
        if staged.exists():
            assert staged.read_bytes() == text.encode(), path
        hashes[path] = hashlib.sha256(text.encode()).hexdigest()
    return text

cli.source = read
for group,extra in (('spec','speculative_dflash_mixed_prefill'),('schedule','max_prefill_tokens')):
    typ, old = cli.FLAG_FIELDS[group]
    cli.FLAG_FIELDS[group] = typ, old + (extra,)
order, parser, _, defaults, namespaces = cli.parser(m['tree'])
util = sys.modules['sglang.srt.arg_groups.arg_utils']
graph = sys.modules['sglang.srt.arg_groups.fields.exec_'].ExecGraph
util.add_cli_args_from_dataclass(parser, graph, fields=['cuda_graph_max_bs_prefill'])
old = cli.parser(m['base'])[0]
assert list(order[:len(old)]) == list(old)
before = cli.parser(m['parent_tree'])[0]
assert list(order) == list(before)
assert all(defaults[k] is False and namespaces[k] == 'spec'
           for k in ('speculative_dflash_mixed_prefill','speculative_dflash_sp_prefill_context'))
expected = dict(cli.EXPECTED, speculative_dflash_sp_prefill_context=True, speculative_dflash_mixed_prefill=True,
                max_prefill_tokens=16640, cuda_graph_max_bs_prefill=16640)
flags = m['launch_flags']
assert set(expected) == {k[2:].replace('-','_') for k in flags}
count = 0
for sequence in permutations(flags):
    argv = []
    for key in sequence:
        argv.append(key)
        if flags[key]:
            argv.append(flags[key])
    actual = vars(parser.parse_args(argv))
    assert all(type(actual[k]) == type(v) and actual[k] == v for k,v in expected.items()),sequence
    count += 1
assert count == 40320

fields = (
    ('model_executor/forward_batch_info.py','ForwardBatch',
     ('mixed_verify_logits_indices','mixed_verify_sharded_logits','dflash_verify_raw_buffer','dflash_prefill_raw_buffer',
      'muse_decode_column_gather','mixed_verify_is_causal','mixed_verify_prompt_count')),
    ('layers/logits_processor.py','LogitsMetadata',('mixed_verify_logits_indices','mixed_verify_sharded_logits')),
    ('layers/logits_processor.py','LogitsProcessorOutput',
     ('dflash_prefill_context_done','dflash_prefill_kv_projected','dflash_verify_fc_shard','dflash_verify_logit_scale')),
)
signatures = {}
for path, name, additions in fields:
    baseline = cli.declarations(read(m['base'],path),name)
    current = cli.declarations(read(m['tree'],path),name)
    assert [ast.dump(x) for x in baseline] == [ast.dump(x) for x in current[:len(baseline)]]
    assert tuple(x.target.id for x in current[len(baseline):]) == additions
    # Preserve real field declarations. Only the accelerator-free enum placeholder is required.
    declarations = '\n'.join('    ' + ast.unparse(x) for x in current)
    native = cli.module('abi_' + name,'from __future__ import annotations\nfrom dataclasses import dataclass\n'
        'class CaptureHiddenMode:\n    NULL=object()\n@dataclass\nclass ' + name + ':\n' + declarations)
    cls = getattr(native,name)
    positional = [object() for _ in current]
    all_explicit, legacy = cls(*positional), cls(*positional[:len(baseline)])
    assert [getattr(all_explicit,n.target.id) for n in current] == positional
    assert [getattr(legacy,n.target.id) for n in baseline] == positional[:len(baseline)]
    new_defaults = {k:getattr(legacy,k) for k in additions}
    if name == 'ForwardBatch':
        assert type(new_defaults['mixed_verify_prompt_count']) is int and new_defaults['mixed_verify_prompt_count'] == 0
    if name == 'LogitsProcessorOutput':
        assert new_defaults['dflash_verify_logit_scale'] is None
    assert list(inspect.signature(cls).parameters) == [n.target.id for n in current]
    signatures[name] = dict(legacy=len(baseline), current=len(current), added_defaults=new_defaults)

print(json.dumps(dict(candidate=m['candidate'],tree=m['tree'],patch_sha256=digest,flags=flags,parsed_cases=count,
    parsed_values=expected,msgspec=cli.msgspec.__version__,original_positional_fields=len(old),current_positional_fields=len(order),
    dataclasses=signatures,source_sha256=hashes,physical_h200=False),indent=2))
