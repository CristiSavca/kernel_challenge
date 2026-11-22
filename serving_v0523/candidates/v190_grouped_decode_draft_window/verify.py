"""Rebuild the release in both directions and confirm that its two CUDA mechanisms retain their source."""
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
P=HERE.parent/'v189_decode_grouped_mma'
D=HERE.parent/'v188_draft_window_metadata'
BASE='297e54fd105e974cdf401e5b4d20253a22c16f90'
P_SHA='18cf2315fa01dc568e28b178f92dad69386191a23ba4834a6616b65fe6eac597'
D_SHA='cf8190f5901a87ca6f6e8effa2f6151998c86f7af47835d23a68c45aa5764471'
K='python/sglang/kernels/ops/attention/metadata.py'
B='python/sglang/srt/layers/attention/flashattention_backend.py'
URL='https://openkernelchallenge.dev/api/v1/serving'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(repo, service):
    pm,dm=[json.loads((folder/'validation.json').read_text()) for folder in (P,D)]
    assert pm['patch_sha256']==P_SHA and dm['patch_sha256']==D_SHA
    assert service['base']['commit']==BASE and pm['base']==dm['base']==BASE
    with tempfile.TemporaryDirectory() as wd:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(wd)/'index'))
        def git(*args,data=None):
            return subprocess.check_output(['git','-C',str(repo),*args],input=data,env=env)
        def apply(tree,patch,reverse=False):
            git('read-tree',tree)
            git('apply','--cached','--whitespace=error-all',*(['--reverse'] if reverse else []),'-',data=patch)
            return git('write-tree').decode().strip()
        full,delta=[(HERE/p).read_bytes() for p in ('patch.diff','delta.diff')]
        p,d=[(x/'patch.diff').read_bytes() for x in (P,D)]
        donor=(D/'delta.diff').read_bytes()
        assert digest(p)==P_SHA and digest(d)==D_SHA and digest(donor)==dm['delta_sha256']
        parent=apply(BASE,p);source=apply(BASE,d)
        assert parent==pm['tree'] and source==dm['tree']
        baseline=apply(BASE,(HERE.parent/'v68_grouped_decode_verify/patch.diff').read_bytes())
        assert apply(baseline,donor)==source and apply(source,donor,True)==baseline
        current=apply(parent,donor)
        assert apply(current,donor,True)==parent and apply(parent,delta)==current and apply(current,delta,True)==parent
        assert apply(BASE,full)==current and apply(current,full,True)==git('rev-parse',BASE+'^{tree}').decode().strip()
        paths=git('diff','--name-only',BASE,current).decode().splitlines()
        changes=git('diff','--name-only',parent,current).decode().splitlines()
        assert len(paths)==28 and changes==[K,B]
        def blob(t,p):return git('show',t+':'+p)
        def method(t,name):return ast.dump(next(n for n in ast.walk(ast.parse(blob(t,B)))
            if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        assert blob(current,K)==blob(source,K)
        shared=['_draft_graph_window','_set_decode_page_metadata','_apply_cuda_graph_metadata']
        assert all(method(current,m)==method(source,m) for m in shared)
        preserved=['forward_extend','forward_decode','muse_verify_fp8_store','_capture_unified_verify_metadata']
        assert all(method(current,m)==method(parent,m) for m in preserved)
        for f,sha in pm['changed_from_parent'].items():
            assert digest(blob(current,f))==sha
        r=service['rules']
        for f in paths:
            assert f.startswith(r['allowed_prefix']) and not f.startswith(tuple(r['forbidden_prefixes']))
            assert f.endswith(tuple(r['allowed_suffixes']))
            if f.endswith('.py'):compile(blob(current,f),f,'exec')
        added=[l[1:] for l in full.decode().splitlines() if l.startswith('+') and not l.startswith('+++')]
        assert len(full)<=r['max_patch_bytes'] and max(map(len,added))<=r['max_line_chars']
        assert not any(x in l for l in added for x in r['forbidden_words'])
        assert not any(l.startswith((b'rename ',b'GIT binary patch',b'Binary files ')) for l in full.splitlines())
        flags=pm['launch_flags'];denied=sum(service['launch_flags']['denied'].values(),[])
        assert flags==dict(dm['launch_flags'],**{'--speculative-dflash-sp-prefill-context':''})
        assert not any(fnmatch.fnmatchcase(k,pattern) for k in flags for pattern in denied)
        suites={}
        for out,old in [('replay',D),('decode',P),('cpu',P)]:
            now=(HERE/(out+'.json')).read_bytes();assert json.loads(now)==json.loads((old/(out+'.json')).read_bytes())
            suites[out]=digest(now)
        compile_receipts={}
        for directory,filename in [(D,'sm90.json'),(D,'interpreter.json'),(P,'sm90a.json')]:
            compile_receipts[directory.name+'/'+filename]=digest((directory/filename).read_bytes())
        return dict(candidate='V190',base=BASE,parent='V189',parent_tree=parent,parent_sha256=P_SHA,
            metadata_donor='V188',metadata_donor_tree=source,metadata_donor_sha256=D_SHA,metadata_donor_delta_sha256=digest(donor),
            tree=current,official_export='okc serve diff .',bytes=len(full),patch_sha256=digest(full),delta_bytes=len(delta),
            delta_sha256=digest(delta),paths=paths,changed_from_parent=changes,launch_flags=flags,denied_flag_patterns=len(denied),
            max_added_line_chars=max(map(len,added)),strict_full_relative_and_donor_reconstruction=True,
            inherited_target_backend_methods=preserved,inherited_draft_backend_methods=shared,
            identical_kernel_source=dict(metadata=digest(blob(current,K)),expert=pm['changed_from_parent']),
            rerun_suites=suites,inherited_codegen_receipts=compile_receipts,physical_H200=False,submitted=False,
            matched_control=dm['control'],rules_url=URL)


if __name__=='__main__':
    rules=json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv)>2 else subprocess.check_output(['curl','-fsSL',URL]))
    print(json.dumps(verify(Path(sys.argv[1]),rules),indent=2))
