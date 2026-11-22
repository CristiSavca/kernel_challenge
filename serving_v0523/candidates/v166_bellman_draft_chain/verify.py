"""Validate the staged tree, stock sampling/verification ownership, export, and CLI."""
import argparse
import ast
import fnmatch
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType


HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()
BASE='297e54fd105e974cdf401e5b4d20253a22c16f90'
PARENT='b5a37c2923f8d1ab24c1c5cf7920a21e2c086774'
P='python/sglang/'
CUDA=P+'kernels/ops/speculative/dflash.py'
MODEL=P+'srt/models/dflash.py'
SPEC=P+'srt/arg_groups/fields/spec.py'
ORDER=P+'srt/arg_groups/field_order.py'
WORKER=P+'srt/speculative/dflash_worker_v2.py'
flag='--speculative-dflash-prefix-value'
key=flag[2:].replace('-','_')
sha=lambda b:hashlib.sha256(b).hexdigest()

def target(node,name):return next(n for n in ast.walk(node) if getattr(n,'name', None)==name)

def cli(contents):
    import msgspec
    # Load the real pinned argument utility and complete Spec field declaration.
    # Its choices module defines plain constants and has no serving dependencies.
    with tempfile.TemporaryDirectory(prefix='v166-cli-') as tmp:
        def load(name,buf):
            f=Path(tmp)/(name.rsplit('.',1)[-1]+'.py');f.write_bytes(buf)
            spec=importlib.util.spec_from_file_location(name,f);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
        prefix='sglang.srt.arg_groups'
        helper=load(prefix+'.arg_utils',contents(P+'srt/arg_groups/arg_utils.py'))
        load(prefix+'.choices',contents(P+'srt/arg_groups/choices.py'))
        spec=load(prefix+'.fields.spec',contents(SPEC)).Spec
        parser=argparse.ArgumentParser()
        helper.add_cli_args_from_dataclass(parser,spec,fields=[key])
        assert getattr(spec(),key) is False
        assert getattr(parser.parse_args([]),key) is False and getattr(parser.parse_args([flag]),key) is True
        return msgspec.__version__


meta=json.loads((HERE.parent/'v64_humming_grouped_sequence_parallel/validation.json').read_text())
parent=(HERE.parent/'v64_humming_grouped_sequence_parallel/patch.diff').read_bytes()
assert sha(parent)==meta['sha256']
full=(HERE/'patch.diff').read_bytes();delta=(HERE/'delta.diff').read_bytes()
live=json.loads(subprocess.check_output(['curl','-fsSL','https://openkernelchallenge.dev/api/v1/serving']))
assert live['base']['commit']==BASE;rules=live['rules']
scored=json.loads((HERE.parent.parent/'runs/abd217baf0ed49fd.json').read_text())
flags=scored['serving_flags']|{flag:''}
denied=[f for group in live['launch_flags']['denied'].values() for f in group]
assert not any(fnmatch.fnmatchcase(f,d) for f in flags for d in denied)
with tempfile.TemporaryDirectory() as tmp:
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'))
    def git(*args,data=None):return subprocess.check_output(['git','-C',STAGE,*args],env=env,input=data)
    def apply(tree,diff,reverse=False):
        git('read-tree',tree);git('apply','--cached','--whitespace=error-all',*(['--reverse'] if reverse else []),'-',data=diff);return git('write-tree').decode().strip()
    base_tree=git('rev-parse',BASE+'^{tree}').decode().strip();assert apply(BASE,parent)==PARENT
    tree=apply(BASE,full);assert apply(PARENT,delta)==tree
    assert apply(tree,full,True)==base_tree and apply(tree,delta,True)==PARENT
    def data(p):return git('show',tree+':'+p)
    paths=git('diff','--name-only',BASE,tree).decode().splitlines()
    changed=git('diff','--name-only',PARENT,tree).decode().splitlines();assert set(changed)=={CUDA,MODEL,ORDER,SPEC}
    for path in paths:
        assert path.startswith(rules['allowed_prefix']) and not any(path.startswith(x) for x in rules['forbidden_prefixes'])
        assert path.endswith(tuple(rules['allowed_suffixes'])) and data(path)==(STAGE/path).read_bytes()
        if path.endswith('.py'):compile(data(path),path,'exec')
    def methods(tree_id,path):
        t=ast.parse(git('show',tree_id+':'+path));return {x.name:ast.dump(x) for x in t.body if isinstance(x,ast.FunctionDef)}
    old=methods(PARENT,CUDA);now=methods(tree,CUDA)
    assert {k for k in old.keys()|now.keys() if old.get(k)!=now.get(k)}=={'_selector_walk_kernel','selector_walk_triton','_prefix_value_plan'}
    worker=ast.parse(data(WORKER));folded=target(target(worker,'_SelectorDraftSampler'),'__call__');eager=target(worker,'_propose_selector_block')
    assert all('self.selector.sample_path(' in ast.unparse(n) for n in (folded,eager))
    assert data(WORKER)==git('show',PARENT+':'+WORKER)
    model=ast.parse(data(MODEL));ctor=target(target(model,'DFlash2DraftModel'),'__init__');walk=target(target(model,'CandidateSelector'),'sample_path')
    assert 'prefix_value=get_spec().speculative_dflash_prefix_value' in ast.unparse(ctor) and 'prefix_value=self.prefix_value' in ast.unparse(walk)
    origin=ast.literal_eval(next(n.value for n in ast.parse(git('show',PARENT+':'+ORDER)).body if isinstance(n,ast.Assign)))
    final=ast.literal_eval(next(n.value for n in ast.parse(data(ORDER)).body if isinstance(n,ast.Assign)))
    assert tuple(final)==(*origin,key)
    cli_version=cli(data)
    digest=sha(data(CUDA))
    assert all(json.loads((HERE/name).read_text())['source_sha256']==digest for name in ('sm90.json','interpreter.json'))
added=[l[1:] for l in full.decode().splitlines() if l.startswith('+') and not l.startswith('+++ ')]
longest=max(map(len,added));assert len(full)<=rules['max_patch_bytes'] and longest<=rules['max_line_chars']
assert not any(v in l for l in added for v in rules['forbidden_words'])
assert not any(l.startswith((b'old mode ',b'new mode ',b'GIT binary patch',b'Binary files ',b'rename from ',b'rename to ')) for l in full.splitlines())
print(json.dumps(dict(candidate='V166',base=BASE,parent='V64',parent_sha256=sha(parent),parent_tree=PARENT,tree=tree,
                      patch_sha256=sha(full),delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),paths=paths,
                      changed_from_parent=changed,launch_flags=flags,fixed_flag_patterns=len(denied),source_sha256=digest,
                      official_export='okc serve diff .',official_rules='https://openkernelchallenge.dev/api/v1/serving',strict_roundtrips=True,
                      maximum_added_line_chars=longest,original_fields_preserved=len(origin),real_cli_msgspec=cli_version,
                      physical_h200=False,submitted=False),indent=2))
