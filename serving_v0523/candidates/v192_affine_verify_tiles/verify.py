"""Verify official V192 export, V190 parent, kernel receipts and public rules using a private Git index."""
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
PARENT=HERE.parent/'v190_grouped_decode_draft_window'
BASE='297e54fd105e974cdf401e5b4d20253a22c16f90'
PARENT_SHA='f08f7a1f790d20b7c59e69a617131f89baf5d18114366767f31c6b8affa4ba88'
K='python/sglang/kernels/ops/attention/'
F='python/sglang/srt/layers/attention/flashattention_backend.py'
URL='https://openkernelchallenge.dev/api/v1/serving'
def sha(b):return hashlib.sha256(b).hexdigest()
def structure(source):
    return {n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def main():
    repo=Path(sys.argv[1])
    service=json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv)>2 else json.loads(subprocess.check_output(['curl','-fsSL',URL]))
    pm=json.loads((PARENT/'validation.json').read_bytes())
    parent_patch=(PARENT/'patch.diff').read_bytes()
    full,delta=[(HERE/f).read_bytes() for f in ('patch.diff','delta.diff')]
    assert sha(parent_patch)==PARENT_SHA and service['base']['commit']==BASE
    with tempfile.TemporaryDirectory() as tmp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'))
        def git(*args,data=None):return subprocess.check_output(['git','-C',str(repo),*args],input=data,env=env)
        def apply(tree,patch,rev=False):
            git('read-tree',tree)
            git('apply','--cached','--whitespace=error-all',*(['--reverse'] if rev else []),'-',data=patch)
            return git('write-tree').decode().strip()
        base_tree=git('rev-parse',BASE+'^{tree}').decode().strip()
        parent=apply(BASE,parent_patch);tree=apply(BASE,full)
        assert parent==pm['tree'] and apply(tree,full,True)==base_tree
        assert apply(parent,delta)==tree and apply(tree,delta,True)==parent
        changes=git('diff','--name-only',parent,tree).decode().splitlines()
        assert changes==[K+'metadata.py',K+'paged_verify_attention.py',F]
        files=git('diff','--name-only',BASE,tree).decode().splitlines()
        assert files==pm['paths']
        def blob(t,f):return git('show',t+':'+f)
        r=service['rules']
        for f in files:
            assert f.startswith(r['allowed_prefix']) and not f.startswith(tuple(r['forbidden_prefixes']))
            assert f.endswith(tuple(r['allowed_suffixes']))
            if f.endswith('.py'):compile(blob(tree,f),f,'exec')
        added=[s[1:] for s in full.decode().splitlines() if s.startswith('+') and not s.startswith('+++')]
        assert len(full)<=r['max_patch_bytes'] and max(map(len,added))<=r['max_line_chars']
        assert not any(w in s for w in r['forbidden_words'] for s in added)
        denied=sum(service['launch_flags']['denied'].values(),[])
        flags=pm['launch_flags'];assert not any(fnmatch.fnmatchcase(f,g) for f in flags for g in denied)
        def definitions(t,f):return structure(blob(t,f))
        def cls(t,n):return definitions(t,F)[n]
        b,c=[cls(t,'FlashAttentionMetadata') for t in (parent,tree)]
        old=[i.target.id for i in b.body if isinstance(i,ast.AnnAssign)]
        new=[i.target.id for i in c.body if isinstance(i,ast.AnnAssign)]
        assert new==old+['verify_tile_origins']
        def methods(t):
            return {i.name:ast.dump(i,include_attributes=False) for i in cls(t,'FlashAttentionBackend').body if isinstance(i,ast.FunctionDef)}
        b,c=methods(parent),methods(tree);changed_methods=sorted(m for m in b if b[m]!=c[m])
        assert changed_methods==['_bind_metadata_buffers','_set_decode_page_metadata','forward_extend','init_cuda_graph_state']
        for name in ('_merge_partitions','_plan','_tile_geometry'):
            left=definitions(parent,K+'paged_verify_attention.py')[name]
            right=definitions(tree,K+'paged_verify_attention.py')[name]
            assert ast.dump(left)==ast.dump(right)
        shas={Path(f).name:sha(blob(tree,f)) for f in changes}
        shas['incumbent.py']=sha(blob(parent,K+'paged_verify_attention.py'))
        received=json.loads((HERE/'interpreter.json').read_text())
        assert all(shas[f]==h for f,h in received['source_sha256'].items())
        aot=json.loads((HERE/'sm90.json').read_text())['tests']
        assert len(aot)==32
        for a in aot:
            kind=a['label'].split('/')[0]
            filename={'V190':'incumbent.py','cert':'metadata.py'}.get(kind,'paged_verify_attention.py')
            assert a['source_sha256']==shas[filename]
            if kind=='direct':assert a['tma_ops']==4 and a['spill_bytes']==[0,0]
        return dict(candidate='V192',parent='V190',base=BASE,parent_tree=parent,parent_sha256=PARENT_SHA,tree=tree,
            official_export='okc serve diff .',bytes=len(full),patch_sha256=sha(full),delta_bytes=len(delta),
            delta_sha256=sha(delta),paths=files,changed_from_parent=changes,source_hashes=shas,
            launch_flags=flags,changed_backend_methods=changed_methods,strict_full_relative_forward_reverse=True,
            denied_flag_patterns=len(denied),max_added_line_chars=max(map(len,added)),rules_url=URL,
            sm90_cases=len(aot),interpreter_cases=received['attention_invocations'],physical_H200=False,submitted=False)
if __name__=='__main__':print(json.dumps(main(),indent=2))
