"""Check the reconstructed V150 tree and official export against its scored lineage."""
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(stage):
    here = Path(__file__).resolve().parent
    parent = here.parent / "v147_integrated_experts_prefill"
    receipt = json.loads((parent / "validation.json").read_text())
    full = (here / "patch.diff").read_bytes()
    delta = (here / "delta.diff").read_bytes()
    old = (parent / "patch.diff").read_bytes()
    assert sha(old) == receipt["patch_sha256"]
    with tempfile.TemporaryDirectory() as temp:
        env = dict(os.environ,GIT_INDEX_FILE=str(Path(temp) / "index"))
        def git(*args, data=None):
            return subprocess.check_output(["git","-C",str(stage),*args],env=env,input=data)
        def apply(tree, patch, reverse=False):
            git("read-tree",tree)
            git("apply","--cached","--whitespace=error",*(["-R"] if reverse else []),"-",data=patch)
            return git("write-tree").decode().strip()
        base = receipt["base"]
        base_tree = git("rev-parse",base + "^{tree}").decode().strip()
        baseline = apply(base,old)
        assert baseline == receipt["tree"]
        tree = apply(base,full)
        assert apply(baseline,delta) == tree
        assert apply(tree,full,True) == base_tree and apply(tree,delta,True) == baseline
        assert subprocess.check_output(["git","-C",str(stage),"write-tree"]).decode().strip() == tree
        names = git("diff","--name-status","--no-renames",base,tree).decode().splitlines()
        paths = []
        for record in names:
            status,path = record.split("\t")
            assert status in ("M","A") and path.startswith("python/sglang/")
            assert "/srt/model_loader/" not in path
            data = git("show",tree + ":" + path)
            assert data == (stage / path).read_bytes()
            if path.endswith(".py"):
                ast.parse(data,filename=path)
            paths.append(path)
        assert b"GIT binary patch" not in full and len(full) <= 400000
        longest = max(len(line[1:].decode()) for line in full.splitlines() if line.startswith(b"+") and not line.startswith(b"+++"))
        assert longest <= 400
        changed = git("diff","--name-only",baseline,tree).decode().splitlines()
        assert len(changed) == 3
        flags = receipt["launch_flags"]
        fixed = (here.parent.parent / "flags").read_text().split("# Fixed,")[1].split("\n",1)[1]
        patterns = [s.strip() for s in fixed.replace("#","").replace("\n","").split(",") if s.strip()]
        assert not any(fnmatch.fnmatchcase(flag,pattern) for flag in flags for pattern in patterns)
        source = sha((stage / next(p for p in paths if p.endswith("/sp_owner_projection.py"))).read_bytes())
        for name in ("cpu_result.json","interpreter.json"):
            assert json.loads((here/name).read_text())["source_sha256"] == source
        bins = json.loads((here/"sm90.json").read_text())["cases"]
        ptx = json.loads((here/"ptx.json").read_text())
        assert all(item["source_sha256"] == source for item in bins)
        assert {i["cubin_sha256"] for i in bins} == {i["cubin_sha256"] for i in ptx}
    return dict(candidate="V150",parent="V147",base=base,parent_tree=baseline,tree=tree,
                parent_sha256=sha(old),patch_sha256=sha(full),bytes=len(full),delta_bytes=len(delta),
                delta_sha256=sha(delta),source_sha256=source,paths=paths,changed_from_parent=changed,
                launch_flags=flags,fixed_flag_patterns=len(patterns),longest_added_source_line=longest,
                full_and_delta_apply_reverse=True,official_export="okc serve diff .",
                physical_H200=False,submitted=False)


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])),indent=2))
