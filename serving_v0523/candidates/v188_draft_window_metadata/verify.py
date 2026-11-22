"""Validate the frozen export and replay V68 + V188 through a temporary Git index."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here=Path(__file__).resolve().parent
manifest=json.loads((here/'validation.json').read_text())
repo=Path(sys.argv[1])
sha=lambda b:hashlib.sha256(b).hexdigest()
patch=(here/'patch.diff').read_bytes()
parent=(here.parent/'v68_grouped_decode_verify/patch.diff').read_bytes()
delta=(here/'delta.diff').read_bytes()
assert [sha(v) for v in (parent,patch,delta)]==[manifest[k] for k in ('parent_sha256','patch_sha256','delta_sha256')]
rules=json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv)>2 else subprocess.check_output(
    ['curl','-fsSL',manifest['rules_url']]))
r=rules['rules']; assert rules['base']['commit']==manifest['base']
with tempfile.TemporaryDirectory() as wd:
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(wd)/'index'))
    def git(*args,data=None):
        return subprocess.check_output(['git','-C',str(repo),*args],input=data,env=env)
    def apply(tree,diff,reverse=False):
        git('read-tree',tree)
        git('apply','--cached','--whitespace=error-all',*(['--reverse'] if reverse else []),'-',data=diff)
        return git('write-tree').decode().strip()
    base=manifest['base'];old=apply(base,parent);new=apply(base,patch)
    assert old==manifest['parent_tree'] and new==manifest['tree'] and apply(old,delta)==new
    assert apply(new,delta,True)==old and apply(new,patch,True)==git('rev-parse',base+'^{tree}').decode().strip()
    paths=git('diff','--name-only',base,new).decode().splitlines()
    assert paths==manifest['paths'] and git('diff','--name-only',old,new).decode().splitlines()==manifest['changed_from_parent']
    assert len(patch)<=r['max_patch_bytes']
    for p in paths:
        assert p.startswith(r['allowed_prefix']) and not p.startswith(tuple(r['forbidden_prefixes'])) and p.endswith(tuple(r['allowed_suffixes']))
        if p.endswith('.py'):compile(git('show',new+':'+p),p,'exec')
    added=[l[1:] for l in patch.decode().splitlines() if l.startswith('+') and not l.startswith('+++')]
    assert max(map(len,added))<=r['max_line_chars'] and not any(x in l for l in added for x in r['forbidden_words'])
    flags=json.loads((here.parent.parent/'runs/04b01f958aa5487c.json').read_text())['result']['serving_flags']
    assert flags==manifest['launch_flags']
    assert not any(fnmatch.fnmatchcase(k,v) for k in flags for v in sum(rules['launch_flags']['denied'].values(),[]))
    for name in ('sm90','interpreter'):
        receipt=json.loads((here/(name+'.json')).read_text())
        assert receipt['metadata_sha256']==sha(git('show',new+':python/sglang/kernels/ops/attention/metadata.py'))
    assert len(json.loads((here/'sm90.json').read_text())['specializations'])==20
    print('Verified',new,len(patch),sha(patch))
