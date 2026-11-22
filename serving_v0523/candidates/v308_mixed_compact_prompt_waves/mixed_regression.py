"""Re-run the V204 parent's mixed scheduler and logits processor fixtures on the combined tree."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

stage = Path(sys.argv[1]).resolve()
directory = Path(__file__).resolve().parent
candidate = next(directory.parent.glob("v202_*"))
head = next(directory.parent.glob("v203_*"))
tree = subprocess.check_output(["git","-C",stage,"write-tree"],text=True).strip()
reports = {}
# The fixtures read the pinned source directly. Materialize on demand, including
# original files outside the candidate diff, in a disposable source directory.
with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp).resolve()
    # Prefill the fixture's literal source dependencies from the combined tree.
    fixture_names = set()
    for directory_ in candidate.parent.iterdir():
        match = re.match(r"v(\d+)_", directory_.name)
        if match and 154 <= int(match[1]) <= 203:
            for script in directory_.glob("*.py"):
                fixture_names.update(re.findall(r"([a-zA-Z_][a-zA-Z_0-9]*\.py)\b", script.read_text()))
    listing = subprocess.check_output(["git","-C",stage,"ls-tree","-r","--name-only",tree,"--","python/sglang"],text=True)
    for name in listing.splitlines():
        path = temp/name
        if path.name != "__init__.py" and path.name in fixture_names:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(subprocess.check_output(["git","-C",stage,"show",tree+":"+name]))
    for name, fixture in (("rounds",candidate/"check_scheduler_rounds.py"),("heads",head/"check_graph_head.py")):
        for _ in range(100):
            run = subprocess.run([sys.executable,"-B",fixture,str(temp)],capture_output=True,text=True)
            if run.returncode == 0:
                reports[name]=json.loads(run.stdout)
                break
            hit = re.search(r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'",run.stderr)
            if hit is None:
                raise RuntimeError(run.stderr)
            path = Path(hit[1])
            # A missing intermediate directory before '..' also prevents POSIX open.
            path.parent.mkdir(parents=True,exist_ok=True)
            path = path.resolve()
            rel = path.relative_to(temp).as_posix()
            assert rel.startswith("python/sglang/") and not path.exists(), run.stderr
            path.write_bytes(subprocess.check_output(["git","-C",stage,"show",tree+":"+rel]))
        else:
            raise RuntimeError("Could not assemble mixed regression sources")
print(json.dumps(dict(tree=tree,**reports),indent=2))
