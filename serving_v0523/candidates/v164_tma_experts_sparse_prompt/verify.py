"""Rebuild from the two frozen parents and check phase boundaries and patch rules."""
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()
a=HERE.parent/'v161_tma_expert_channels'
b=HERE.parent/'v163_sparse_prompt_exchange'
meta_a,meta_b=[json.loads((p/'validation.json').read_text()) for p in (a,b)]
sha=lambda data:hashlib.sha256(data).hexdigest()
get=lambda p:(p/'patch.diff').read_bytes()
full,delta=get(HERE),(HERE/'delta.diff').read_bytes()
live=json.loads(subprocess.check_output(['curl','-fsSL','https://openkernelchallenge.dev/api/v1/serving']))
base=live['base']['commit'];assert base==meta_a['pinned_commit']==meta_b['base']
rules=live['rules'];flags=meta_a['launch_flags']|{'--muse-sparse-prompt-exchange':''}
assert flags==meta_b['launch_flags']
denied=[f for g in live['launch_flags']['denied'].values() for f in g]
assert not any(fnmatch.fnmatchcase(f,ban) for f in flags for ban in denied)
assert sha(get(a))==meta_a['patch_sha256'] and sha(get(b))==meta_b['patch_sha256']
def methods(data):
    tree=ast.parse(data)
    return {f.name:ast.dump(f,include_attributes=False) for f in ast.walk(tree) if isinstance(f,ast.FunctionDef)}
with tempfile.TemporaryDirectory(prefix='v164-verify-') as scratch:
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(scratch)/'index'))
    def git(*args,data=None):
        return subprocess.check_output(['git','-C',STAGE,*args],env=env,input=data)
    def apply(tree,patch,rev=False):
        git('read-tree',tree)
        git('apply','--cached','--whitespace=error-all',*(['--reverse'] if rev else []),'-',data=patch)
        return git('write-tree').decode().strip()
    tree_a=apply(base,get(a));tree_b=apply(base,get(b));assert (tree_a,tree_b)==(meta_a['tree'],meta_b['tree'])
    result=apply(base,full);assert apply(tree_a,delta)==result
    assert apply(result,full,True)==git('rev-parse',base+'^{tree}').decode().strip()
    assert apply(result,delta,True)==tree_a
    assert result==subprocess.check_output(['git','-C',STAGE,'write-tree'],text=True).strip()
    paths=git('diff','--name-only',base,result).decode().splitlines()
    changed=git('diff','--name-only',tree_a,result).decode().splitlines()
    assert changed==meta_b['changed_from_parent']
    def read(tree,p):return git('show',f'{tree}:{p}')
    for p in paths:
        assert p.startswith(rules['allowed_prefix']) and not any(p.startswith(d) for d in rules['forbidden_prefixes'])
        assert p.endswith(tuple(rules['allowed_suffixes']))
        data=read(result,p)
        assert data and data==(STAGE/p).read_bytes() and git('ls-tree',result,p).startswith((b'100644 blob ',b'100755 blob '))
        if p.endswith('.py'):compile(data,p,'exec')
    assert set(paths)==set(meta_a['paths'])|set(meta_b['paths'])
    for p in changed:
        if p.endswith(('field_order.py','fields/parallel.py','sparse_prompt_exchange.py')):
            assert read(result,p)==read(tree_b,p)
    hum=next(p for p in changed if p.endswith('muse_spark_humming_moe.py'))
    x=methods(read(tree_a,hum));y=methods(read(result,hum))
    altered={k for k in x.keys()|y.keys() if x.get(k)!=y.get(k)}
    assert altered=={'humming_grouped_expert_outputs','_compact_prefill_windows','compact_prefill_and_combine'}
    core=next(p for p in changed if p.endswith('models/muse_spark.py'))
    assert methods(read(tree_b,core))['_forward_sp_replicated_experts']==methods(read(result,core))['_forward_sp_replicated_experts']
    def model_methods(data):
        return {(cls.name,f.name):ast.dump(f,include_attributes=False) for cls in ast.parse(data).body
                if isinstance(cls,ast.ClassDef) for f in cls.body if isinstance(f,ast.FunctionDef)}
    u=model_methods(read(tree_a,core));v=model_methods(read(result,core))
    assert {k for k in u.keys()|v.keys() if u.get(k)!=v.get(k)}=={('MuseSpark1xMoE','_forward_sp_replicated_experts')}
    source_hashes={p:sha(read(result,p)) for p in changed+list(meta_a['source_hashes'])}
    for p,digest in meta_a['source_hashes'].items():
        if p.endswith('.cuh'):assert source_hashes[p]==digest
assert len(full)<=rules['max_patch_bytes']
added=[l[1:] for l in full.decode().splitlines() if l.startswith('+') and not l.startswith('+++ ')]
assert not any(v in line for v in rules['forbidden_words'] for line in added)
assert not any(l.startswith((b'old mode ',b'new mode ',b'GIT binary patch',b'Binary files ',b'rename from ',b'rename to ')) for l in full.splitlines())
longest=max(map(len,added));assert longest<=rules['max_line_chars']
print(json.dumps(dict(candidate='V164',base=base,parent='V161',parent_sha256=sha(get(a)),parent_tree=tree_a,
    prompt_parent='V163',prompt_parent_sha256=sha(get(b)),prompt_parent_tree=tree_b,
    tree=result,patch_sha256=sha(full),bytes=len(full),delta_sha256=sha(delta),delta_bytes=len(delta),launch_flags=flags,
    paths=paths,changed_from_parent=changed,source_sha256=source_hashes,unmodified_v161_cuh_and_prepare_dispatch=True,
    v163_prompt_model_method_and_cli_fields_identical=True,v161_changed_humming_functions=sorted(altered),
    official_export='okc serve diff .',strict_full_delta_apply_reverse=True,official_rules='https://openkernelchallenge.dev/api/v1/serving',
    max_added_line_chars=longest,fixed_flag_patterns=len(denied),python_compiled=sum(p.endswith('.py') for p in paths),
    physical_H200=False,judge_submitted=False),indent=2))
