"""Reconstruct this one-file V195 extension and verify kernel/candidate receipts and the public contract."""
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
PARENT=HERE.parent/'v195_fused_expert_affine_verify'
BASE='297e54fd105e974cdf401e5b4d20253a22c16f90'
PARENT_SHA='dd9ebfcbf2f8b14f66e7d1ed63fafd1c4c9d441766f076ee0cbb06f835ebc0f7'
PATH='python/sglang/kernels/ops/attention/paged_verify_attention.py'
PUBLIC='https://openkernelchallenge.dev/api/v1/serving'
def sha(x):return hashlib.sha256(x).hexdigest()
def methods(src):return {n.name:n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)}

def main():
    repo=Path(sys.argv[1]);service=json.loads(Path(sys.argv[2]).read_bytes()) if len(sys.argv)>2 else json.loads(subprocess.check_output(['curl','-fsSL',PUBLIC]))
    pm=json.loads((PARENT/'validation.json').read_bytes());parent_patch=(PARENT/'patch.diff').read_bytes()
    full,delta=[(HERE/n).read_bytes() for n in ('patch.diff','delta.diff')]
    assert sha(parent_patch)==PARENT_SHA and pm['base']==service['base']['commit']==BASE
    with tempfile.TemporaryDirectory() as tmp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'))
        def git(*args,data=None):return subprocess.check_output(['git','-C',str(repo),*args],input=data,env=env)
        def apply(tree,patch,rev=False):
            git('read-tree',tree);git('apply','--cached','--whitespace=error-all',*(['--reverse'] if rev else []),'-',data=patch)
            return git('write-tree').decode().strip()
        parent=apply(BASE,parent_patch);tree=apply(BASE,full)
        assert parent==pm['tree'] and apply(tree,full,True)==git('rev-parse',BASE+'^{tree}').decode().strip()
        assert apply(parent,delta)==tree and apply(tree,delta,True)==parent
        changes=git('diff','--name-only',parent,tree).decode().splitlines();assert changes==[PATH]
        paths=git('diff','--name-only',BASE,tree).decode().splitlines();assert paths==pm['paths']
        def blob(t,p):return git('show',t+':'+p)
        old,new=blob(parent,PATH),blob(tree,PATH);a,b=methods(old),methods(new)
        assert set(b)-set(a)=={'_tile_origin_mask','_next_tile_run'}
        assert [name for name in a if ast.dump(a[name])!=ast.dump(b[name])]==['_accumulate_tiles','_partition_attention']
        def loop(node):
            return ast.dump(next(n for n in node.body if isinstance(n,ast.For)),include_attributes=False)
        assert loop(a['_accumulate_tiles'])==loop(b['_accumulate_tiles'])
        r=service['rules']
        for f in paths:
            assert f.startswith(r['allowed_prefix']) and not f.startswith(tuple(r['forbidden_prefixes']))
            assert f.endswith(tuple(r['allowed_suffixes']))
            if f.endswith('.py'):compile(blob(tree,f),f,'exec')
        added=[l[1:] for l in full.decode().splitlines() if l.startswith('+') and not l.startswith('+++')]
        assert len(full)<=r['max_patch_bytes'] and max(map(len,added))<=r['max_line_chars']
        assert not any(w in l for w in r['forbidden_words'] for l in added)
        denied=sum(service['launch_flags']['denied'].values(),[])
        assert not any(fnmatch.fnmatchcase(f,g) for f in pm['launch_flags'] for g in denied)
        tests=json.loads((HERE/'interpreter.json').read_bytes())
        assert tests['sha256']==sha(new) and tests['parent_sha256']==sha(old)
        assert tests['numerical']['indexed_to_direct']>0 and tests['numerical']['direct_to_indexed']>0
        bins=json.loads((HERE/'sm90.json').read_bytes())['records']
        versions={'V195':sha(old),'V196':sha(new)}
        for row in bins:
            if row['variant'] in versions:assert row['source_sha256']==versions[row['variant']]
        primary=[v for v in bins if v['variant']=='V196' and v['aligned'] and v['enabled']]
        assert len(primary)==12 and len(bins)==38 and all(v['spill']==[0,0] for v in primary)
        return dict(candidate='V196',parent='V195',base=BASE,parent_tree=parent,parent_sha256=PARENT_SHA,tree=tree,
            official_export='okc serve diff .',bytes=len(full),patch_sha256=sha(full),delta_bytes=len(delta),delta_sha256=sha(delta),
            paths=paths,changed_from_parent=changes,source_sha256=sha(new),prior_source_sha256=sha(old),launch_flags=pm['launch_flags'],
            strict_full_relative_forward_reverse=True,identical_ordered_tile_compute_AST=True,
            evidence_source_bound=['sm90.json','interpreter.json'],sm90_cases=len(bins),aligned_primary_cases=len(primary),
            denied_flag_patterns=len(denied),max_added_line_chars=max(map(len,added)),rules_url=PUBLIC,
            physical_H200=False,submitted=False)
if __name__=='__main__':print(json.dumps(main(),indent=2))
