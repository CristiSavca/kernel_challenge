"""Reconstruct the V254 full/delta patches on the public pin and check the published rules."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "v68_grouped_decode_verify"
STAGE = Path(sys.argv[1]).resolve()
RULES_URL = "https://openkernelchallenge.dev/api/v1/serving"
CHANGED = {
    "python/sglang/kernels/ops/moe/muse_spark_routing.py",
    "python/sglang/kernels/ops/moe/muse_spark_routing_jit.py",
    "python/sglang/kernels/jit/csrc/muse_spark/routing.cuh",
    "python/sglang/srt/models/muse_spark.py",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


previous = json.loads((PARENT / "validation.json").read_bytes())
contract = json.loads(subprocess.check_output(["curl", "-fsSL", RULES_URL]))
rules = contract["rules"]
pin = contract["base"]["commit"]
assert pin == json.loads((PARENT.parent / "v64_humming_grouped_sequence_parallel/validation.json").read_bytes())["base"]
parent, full, delta = [p.read_bytes() for p in (
    PARENT / "patch.diff", HERE / "patch.diff", HERE / "delta.diff")]
assert sha(parent) == previous["sha256"]

with tempfile.TemporaryDirectory() as scratch:
    env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}

    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", str(STAGE), *args], input=data, env=env)

    def apply(tree, patch, reverse=False):
        git("read-tree", tree)
        reverse_args = ["--reverse"] if reverse else []
        git("apply", "--cached", "--whitespace=error-all", *reverse_args, "-", data=patch)
        return git("write-tree").decode().strip()

    native = git("rev-parse", pin + "^{tree}").decode().strip()
    before, tree = apply(pin, parent), apply(pin, full)
    assert before == previous["trees"]["v68"] and apply(before, delta) == tree
    assert apply(tree, delta, True) == before and apply(tree, full, True) == native
    assert set(git("diff", "--name-only", before, tree).decode().splitlines()) == CHANGED
    paths = git("diff", "--name-only", native, tree).decode().splitlines()
    for path in paths:
        assert path.startswith(rules["allowed_prefix"]) and path.endswith(tuple(rules["allowed_suffixes"]))
        assert not any(path.startswith(prefix) for prefix in rules["forbidden_prefixes"])
        old, new = git("ls-tree", native, path), git("ls-tree", tree, path)
        mode = old.split()[0] if old else b"100644"
        assert mode in {b"100644", b"100755"} and new.startswith(mode + b" blob ")
        content = git("show", tree + ":" + path)
        assert content and b"\0" not in content and content == (STAGE / path).read_bytes()
        if path.endswith(".py"):
            compile(content, path, "exec")

lines = full.decode().splitlines()
assert not any(line.startswith(("rename from ", "rename to ", "old mode ", "new mode ", "GIT binary patch", "Binary files")) for line in lines)
added = [line[1:] for line in lines if line.startswith("+") and not line.startswith("+++ ")]
assert max(map(len, added)) <= rules["max_line_chars"] and len(full) <= rules["max_patch_bytes"]
assert not any(word in line for line in added for word in rules["forbidden_words"])
control = json.loads((HERE.parents[1] / "runs/04b01f958aa5487c.json").read_bytes())
assert control["mode"] == "check" and control["result"]["passed"]
flags = control["serving_flags"]
denied = [p for patterns in contract["launch_flags"]["denied"].values() for p in patterns]
assert not any(fnmatch.fnmatchcase(flag, pat) for flag in flags for pat in denied)
print(json.dumps(dict(candidate="V254", parent="V68", base=pin, parent_tree=before, tree=tree,
    parent_sha256=sha(parent), patch_sha256=sha(full), delta_sha256=sha(delta), bytes=len(full),
    delta_bytes=len(delta), paths=paths, changed_from_parent=sorted(CHANGED), launch_flags=flags,
    forbidden_flag_patterns=len(denied), max_added_line_chars=max(map(len, added)),
    strict_apply_reverse_full_delta=True, official_export="okc serve diff", official_rules=RULES_URL,
    physical_H200=False, submitted=False), indent=2))
