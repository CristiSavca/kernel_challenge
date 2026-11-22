"""Validate the official export, V163 delta, unchanged arithmetic and live source rules."""
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
V163 = HERE.parent / "v163_sparse_prompt_exchange"
parent = json.loads((V163 / "validation.json").read_text())
live = json.loads(subprocess.check_output(["curl", "-fsSL", "https://openkernelchallenge.dev/api/v1/serving"]))
base = live["base"]["commit"]
assert base == parent["base"]
rules = live["rules"]
sha = lambda data: hashlib.sha256(data).hexdigest()
delta = (HERE / "delta.diff").read_bytes()
full = (HERE / "patch.diff").read_bytes()
original = (V163 / "patch.diff").read_bytes()
assert sha(original) == parent["patch_sha256"]
with tempfile.TemporaryDirectory() as scratch:
    env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch) / "index"))
    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", STAGE, *args], env=env, input=data)
    def apply(tree, buf, rev=False):
        git("read-tree", tree)
        git("apply", "--cached", "--whitespace=error-all", *(["-R"] if rev else []), "-", data=buf)
        return git("write-tree").decode().strip()
    before = apply(base, original)
    tree = apply(base, full)
    assert before == parent["tree"] and apply(before, delta) == tree
    assert apply(tree, full, True) == git("rev-parse", base+"^{tree}").decode().strip()
    assert apply(tree, delta, True) == before
    diff_paths = lambda tree_a: git("diff", "--name-only", tree_a, tree).decode().splitlines()
    paths, changed = diff_paths(base), diff_paths(before)
    assert set(changed) == {
        "python/sglang/kernels/ops/moe/muse_spark_sparse_return.py",
        "python/sglang/srt/layers/muse_spark/sparse_prompt_exchange.py",
        "python/sglang/srt/models/muse_spark.py",
    }
    hashes = {}
    for name in paths:
        assert name.startswith(rules["allowed_prefix"])
        assert not any(name.startswith(p) for p in rules["forbidden_prefixes"])
        assert Path(name).suffix in rules["allowed_suffixes"]
        assert git("ls-tree", tree, name).startswith((b"100644 blob ", b"100755 blob "))
        data = git("show", f"{tree}:{name}")
        assert data and data == (STAGE / name).read_bytes()
        compile(data, name, "exec")
        if name in changed:
            hashes[name] = sha(data)
assert len(full) <= rules["max_patch_bytes"]
lines = full.decode().splitlines()
assert not any(l.startswith(("rename from ", "rename to ", "old mode ", "new mode ", "GIT binary patch", "Binary files")) for l in lines)
added = [l[1:] for l in lines if l.startswith("+") and not l.startswith("+++ ")]
assert max(map(len, added)) <= rules["max_line_chars"]
assert not any(w in l for l in added for w in rules["forbidden_words"])
flags = parent["launch_flags"]
denied = [p for group in live["launch_flags"]["denied"].values() for p in group]
assert not any(fnmatch.fnmatchcase(f,p) for f in flags for p in denied)
print(json.dumps(dict(candidate="V167", base=base, parent="V163", parent_tree=before,
    parent_sha256=sha(original), tree=tree, patch_sha256=sha(full), bytes=len(full),
    delta_sha256=sha(delta), delta_bytes=len(delta), paths=paths, changed_from_parent=changed,
    source_sha256=hashes, launch_flags=flags, export="okc serve diff", all_roundtrips=True,
    python_units_compiled=len(paths), fixed_flag_patterns=len(denied), max_line=max(map(len,added)),
    official_rules="https://openkernelchallenge.dev/api/v1/serving", physical_H200=False), indent=2))
