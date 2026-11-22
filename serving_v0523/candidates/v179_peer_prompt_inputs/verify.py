"""Reconstruct the full and incremental exports and validate public serving rules."""

import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
PARENT = HERE.parent / "v176_integrated_prompt_decode"
URL = "https://openkernelchallenge.dev/api/v1/serving"
EXTRA = {"python/sglang/kernels/ops/moe/muse_spark_peer_rows.py",
         "python/sglang/srt/layers/muse_spark/peer_prompt_inputs.py",
         "python/sglang/srt/layers/muse_spark/sparse_prompt_exchange.py",
         "python/sglang/srt/models/muse_spark.py"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


previous = json.loads((PARENT / "validation.json").read_bytes())
ruleset = json.loads(subprocess.check_output(["curl", "-fsSL", URL]))
rules = ruleset["rules"]
base = ruleset["base"]["commit"]
assert base == previous["base"]
full, delta, ancestor = [(d / f).read_bytes() for d, f in (
    (HERE, "patch.diff"), (HERE, "delta.diff"), (PARENT, "patch.diff"))]
assert sha(ancestor) == previous["patch_sha256"]
with tempfile.TemporaryDirectory() as tmp:
    env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", STAGE, *args], env=env, input=data)

    def apply(tree, patch, reverse=False):
        git("read-tree", tree)
        git("apply", "--cached", "--whitespace=error-all", *(["-R"] if reverse else []), "-", data=patch)
        return git("write-tree").decode().strip()

    start, tree = apply(base, ancestor), apply(base, full)
    assert start == previous["tree"] and apply(start, delta) == tree
    assert apply(tree, delta, True) == start and apply(tree, full, True) == git("rev-parse", base + "^{tree}").decode().strip()
    assert set(git("diff", "--name-only", start, tree).decode().splitlines()) == EXTRA
    paths = git("diff", "--name-only", base, tree).decode().splitlines()
    for p in paths:
        assert p.startswith(rules["allowed_prefix"]) and not any(p.startswith(i) for i in rules["forbidden_prefixes"])
        assert p.endswith(tuple(rules["allowed_suffixes"]))
        existing = git("ls-tree", base, p)
        expected_mode = existing.split()[0] if existing else b"100644"
        assert expected_mode in (b"100644", b"100755")
        assert git("ls-tree", tree, p).startswith(expected_mode + b" blob ")
        source = git("show", tree + ":" + p)
        assert source == (STAGE / p).read_bytes()
        if p.endswith(".py"):
            compile(source, p, "exec")
    kernel = "python/sglang/kernels/ops/moe/muse_spark_peer_rows.py"
    kernel_sha = sha(git("show", tree + ":" + kernel))
lines = full.decode().splitlines()
assert not any(l.startswith(("rename from ", "rename to ", "old mode ", "new mode ", "GIT binary patch", "Binary files")) for l in lines)
added = [l[1:] for l in lines if l.startswith("+") and not l.startswith("+++ ")]
assert max(map(len, added)) <= rules["max_line_chars"] and len(full) <= rules["max_patch_bytes"]
assert not any(bad in line for line in added for bad in rules["forbidden_words"])
flags = previous["launch_flags"]
denied = [p for patterns in ruleset["launch_flags"]["denied"].values() for p in patterns]
assert not any(fnmatch.fnmatchcase(f, p) for f in flags for p in denied)
print(json.dumps(dict(candidate="V179", parent="V176", base=base, parent_tree=start, parent_sha256=sha(ancestor),
    tree=tree, patch_sha256=sha(full), delta_sha256=sha(delta), bytes=len(full), delta_bytes=len(delta),
    paths=paths, changed_from_parent=sorted(EXTRA), peer_kernel_sha256=kernel_sha, launch_flags=flags,
    checked_flag_patterns=len(denied), max_added_line_chars=max(map(len,added)), official_export="okc serve diff",
    official_rules=URL, strict_apply_reverse_full_delta=True, submitted=False, physical_H200=False), indent=2))
