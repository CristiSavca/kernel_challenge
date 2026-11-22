"""Run the released verifier wrapper and native expert dispatch tests on the same applied tree."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


here = Path(__file__).resolve().parent
root = Path(sys.argv[1]).resolve()
project = here.parent
v295 = next(project.glob("v295_*/validation.json"))
v297 = next(project.glob("v297_*/independent_dispatch.py"))
v301 = next(project.glob("v301_*/independent_runtime.py"))
base = json.loads(v295.read_bytes())["base"]
combined = subprocess.check_output(["git", "-C", root, "write-tree"], text=True).strip()
def git(*args, data=None):
    return subprocess.check_output(["git", "-C", root, *args], input=data)
def capture(call):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        call()
    return json.loads(out.getvalue())

with tempfile.TemporaryDirectory() as tmp:
    objects = Path(tmp) / "objects"
    objects.mkdir()
    stage_git = git("rev-parse", "--absolute-git-dir").decode().strip()
    os.environ.update(GIT_INDEX_FILE=str(Path(tmp)/"index"), GIT_OBJECT_DIRECTORY=str(objects),
                      GIT_ALTERNATE_OBJECT_DIRECTORIES=str(Path(stage_git)/"objects"))
    git("read-tree",base)
    git("apply","--cached","--whitespace=error-all","-",data=(v295.parent/"patch.diff").read_bytes())
    earlier = git("write-tree").decode().strip()
    expert = runpy.run_path(str(v297))
    fn = expert["main"]
    assert fn.__globals__["PARENT"] == earlier
    fn.__globals__["FINAL"] = combined
    result = capture(fn)
    # The geometry replay rebuilds its own read-only V297 parent.
    verifier = runpy.run_path(str(v301))
    attn = capture(lambda: verifier["main"](root))
assert result["candidate_tree"]==combined and result["calls"]==56
assert attn==json.loads((v301.parent/"independent_runtime.json").read_text())
src = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (v297,v301)}
print(json.dumps(dict(tree=combined,source_sha256=src,experts=result,verification=attn,
    scope="Combined source under the normal attention launch wrapper and preparation/routing/projection dispatch. Expert math and device launches are modeled. Kernel outputs and Hopper builds are verified separately."),indent=2))
