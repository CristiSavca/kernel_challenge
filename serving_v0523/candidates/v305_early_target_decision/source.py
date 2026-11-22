"""Resolve exact V302 files without mutating the working checkout or persistent Git object store."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
CP=HERE.parent
ROOT=HERE.parents[2]
PARENT=CP/"v302_decode_verify_partitions"

@contextmanager
def sources():
    repo=ROOT/"serving_v0523/sglang"
    gd=subprocess.check_output(["git","-C",repo,"rev-parse","--absolute-git-dir"],text=True).strip()
    r=json.loads((PARENT/"validation.json").read_bytes())
    with tempfile.TemporaryDirectory() as td:
        obj=Path(td)/"objects";obj.mkdir()
        env=dict(os.environ,GIT_INDEX_FILE=td+"/index",GIT_OBJECT_DIRECTORY=str(obj),GIT_ALTERNATE_OBJECT_DIRECTORIES=gd+"/objects")
        def git(*cmd,data=None):return subprocess.check_output(["git","-C",repo,*cmd],env=env,input=data)
        git("read-tree",r["base"]);git("apply","--cached","--whitespace=error-all","-",data=(PARENT/"patch.diff").read_bytes())
        assert git("write-tree").decode().strip()==r["tree"]
        yield lambda name:git("show",":python/sglang/"+name).decode()


if __name__=="__main__":
    import sys
    with sources() as get:print(get(sys.argv[1]))
