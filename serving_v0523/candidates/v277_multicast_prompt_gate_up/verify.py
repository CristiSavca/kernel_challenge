"""Reconstruct both exports, check the current public contract and connect the independent receipts."""
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1]).resolve()
parent=here.parent/"v248_shared_verify_raster"
previous=json.loads((parent/"validation.json").read_bytes())
contract=json.loads(Path(sys.argv[2]).read_bytes()) if len(sys.argv)>2 else json.loads(
    subprocess.check_output(["curl","-fsSL","https://openkernelchallenge.dev/api/v1/serving"]))
limits=contract["rules"]
base=contract["base"]["commit"]
def digest(data):
    return hashlib.sha256(data).hexdigest()

with tempfile.TemporaryDirectory() as tmp:
    obj=Path(tmp)/"objects"
    obj.mkdir()
    objects=subprocess.check_output(["git","-C",stage,"rev-parse","--absolute-git-dir"],text=True).strip()+"/objects"
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/"index"),GIT_OBJECT_DIRECTORY=str(obj),GIT_ALTERNATE_OBJECT_DIRECTORIES=objects)
    def git(*args, stdin=None):
        return subprocess.check_output(["git","-C",stage,*args],input=stdin,env=env)
    def apply(tree,patch,reverse=False):
        git("read-tree",tree)
        git("apply","--cached","--whitespace=error-all",*(["-R"] if reverse else []),"-",stdin=patch)
        return git("write-tree").decode().strip()
    patch=(here/"patch.diff").read_bytes()
    delta=(here/"delta.diff").read_bytes()
    pp=(parent/"patch.diff").read_bytes()
    prev=apply(base,pp)
    assert prev==previous["tree"] and digest(pp)==previous["patch_sha256"]
    tree=apply(base,patch)
    assert apply(prev,delta)==tree and apply(tree,delta,True)==prev
    assert apply(tree,patch,True)==git("rev-parse",base+"^{tree}").decode().strip()
    changed=git("diff","--name-only",prev,tree).decode().splitlines()
    filelist=git("diff","--name-only",base,tree).decode().splitlines()
    assert len(changed)==4 and len(patch)<=limits["max_patch_bytes"]
    for path in filelist:
        assert path.startswith(limits["allowed_prefix"]) and not path.startswith(tuple(limits["forbidden_prefixes"]))
        assert path.endswith(tuple(limits["allowed_suffixes"]))
        if path.endswith(".py"): ast.parse(git("show",tree+":"+path))
    plus=[x[1:] for x in patch.decode().splitlines() if x.startswith("+") and not x.startswith("+++")]
    assert max(map(len,plus))<=limits["max_line_chars"] and all(w not in x for x in plus for w in limits["forbidden_words"])
    assert not any(patch.startswith(t) or b"\n"+t in patch for t in [b"rename ",b"old mode ",b"new mode ",b"GIT binary patch",b"Binary files "])
    flags=previous["launch_flags"]
    denied=[p for group in contract["launch_flags"]["denied"].values() for p in group]
    assert not any(fnmatch.fnmatchcase(k,p) for k in flags for p in denied)
    sources={p:digest(git("show",tree+":"+p)) for p in changed}
    assert [Path(p).name for p in changed]==["decode_gate_up.cuh","decode_gate_up_launch.cuh","decode_gate_up.py","prompt_multicast.py"]
    machine=json.loads((here/"machine.json").read_bytes())
    assert machine["images"][1]["source_sha256"]==json.loads((here/"independent_fence.json").read_bytes())["images"][0]["sha256"]
    assert git("diff","--no-renames","-U1",base,tree)==patch
    print(json.dumps(dict(candidate="V277",parent="V248",base=base,parent_tree=prev,parent_sha256=digest(pp),tree=tree,
        patch_sha256=digest(patch),delta_sha256=digest(delta),bytes=len(patch),delta_bytes=len(delta),changed_from_parent=changed,
        paths=filelist,sources=sources,launch_flags=flags,added_line_maximum=max(map(len,plus)),strict_apply_reverse_full_delta=True,
        official_export="okc serve diff with git diff.context=1",GPU=False,submitted=False),indent=2))

