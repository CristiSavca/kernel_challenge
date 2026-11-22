"""Rebuild the cumulative export and the donor delta using isolated Git objects."""
from contextlib import contextmanager
from pathlib import Path
import ast
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
SG=HERE.parents[1]/"sglang"
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
DONOR=next(HERE.parent.glob("v68_*"))
BASE="python/sglang/srt/mem_cache/"
COMMON=BASE+"pool_host/common.py"
MHA=BASE+"pool_host/mha.py"
def digest(d):return hashlib.sha256(d).hexdigest()

@contextmanager
def candidate():
    with tempfile.TemporaryDirectory() as scratch:
        temp=Path(scratch)
        (temp/"objects").mkdir()
        gitdir=subprocess.check_output(["git","-C",SG,"rev-parse","--absolute-git-dir"],text=True).strip()
        env=dict(os.environ,GIT_INDEX_FILE=str(temp/"index"),GIT_OBJECT_DIRECTORY=str(temp/"objects"),
                 GIT_ALTERNATE_OBJECT_DIRECTORIES=str(Path(gitdir)/"objects"))
        def git(*args,data=None):
            return subprocess.check_output(["git","-C",SG,*args],input=data,env=env)
        git("read-tree",PIN)
        old=(DONOR/"patch.diff").read_bytes();new=(HERE/"patch.diff").read_bytes();delta=(HERE/"delta.diff").read_bytes()
        git("apply","--cached","-",data=old)
        parent=git("write-tree").decode().strip()
        assert parent==json.loads((DONOR/"validation.json").read_bytes())["trees"]["v68"]
        git("apply","--cached","-",data=delta)
        tree=git("write-tree").decode().strip()
        git("apply","--reverse","--cached","-",data=delta)
        assert git("write-tree").decode().strip()==parent
        git("read-tree",PIN)
        git("apply","--cached","-",data=new)
        assert git("write-tree").decode().strip()==tree
        git("apply","--reverse","--cached","-",data=new)
        assert git("write-tree").strip()==git("rev-parse",PIN+"^{tree}").strip()
        yield parent,tree,git

if __name__=="__main__":
    contract=json.loads(Path(sys.argv[1]).read_bytes())
    with candidate() as (parent,tree,git):
        changed=git("diff","--name-only",parent,tree).decode().splitlines()
        assert set(changed)=={COMMON,MHA}
        paths=git("diff","--name-only",PIN,tree).decode().splitlines()
        assert all(p.startswith("python/sglang/") and "/model_loader/" not in p for p in paths)
        for p in paths:
            if p.endswith(".py"): ast.parse(git("show",tree+":"+p))
        full=(HERE/"patch.diff").read_bytes();delta=(HERE/"delta.diff").read_bytes()
        maxlen=max(map(len,[l for l in full.decode().splitlines() if l.startswith("+") and not l.startswith("+++")]))-1
        assert maxlen<=400 and len(full)<=contract["rules"]["max_patch_bytes"] and PIN==contract["base"]["commit"]
        flags=json.loads((HERE/"flags.json").read_bytes())
        print(json.dumps(dict(candidate="V298",parent="V68",base=PIN,parent_tree=parent,tree=tree,paths=paths,changed=changed,
            source_sha256={p:digest(git("show",tree+":"+p)) for p in changed},patch_sha256=digest(full),
            delta_sha256=digest(delta),parent_sha256=digest((DONOR/"patch.diff").read_bytes()),bytes=len(full),
            delta_bytes=len(delta),max_added_line=maxlen,official_byte_limit=contract["rules"]["max_patch_bytes"],
            launch_flags=flags,export="okc serve diff .",strict_reconstruction=True,physical_H200=False,submitted=False),indent=2))
