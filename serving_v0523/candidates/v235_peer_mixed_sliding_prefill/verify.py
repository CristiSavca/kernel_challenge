"""Reconstruct the V235 full/delta patches on the public pin and check the published rules."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "v233_peer_prompt_overflow"
STAGE = Path(sys.argv[1]).resolve()
RULES_URL = "https://openkernelchallenge.dev/api/v1/serving"
DONOR = HERE.parent / "v234_causal_mixed_sliding_prefill"
donor = json.loads((DONOR / "validation.json").read_bytes())
CHANGED = set(donor["changed_from_parent"])


def sha(data):
    return hashlib.sha256(data).hexdigest()


previous = json.loads((PARENT / "validation.json").read_bytes())
contract = json.loads(subprocess.check_output(["curl", "-fsSL", RULES_URL]))
rules = contract["rules"]
pin = contract["base"]["commit"]
assert pin == previous["base"]
parent, full, delta = [p.read_bytes() for p in (
    PARENT / "patch.diff", HERE / "patch.diff", HERE / "delta.diff")]
assert sha(parent) == previous["patch_sha256"]

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
    assert before == previous["tree"] and apply(before, delta) == tree
    assert apply(tree, delta, True) == before and apply(tree, full, True) == native
    assert set(git("diff", "--name-only", before, tree).decode().splitlines()) == CHANGED
    from itertools import permutations
    members = {k: next(HERE.parent.glob(f"v{k}_*")) for k in (229,230,231,232,234)}
    member_info = {k: json.loads((p/"validation.json").read_bytes()) for k,p in members.items()}
    base_229 = member_info[229]["tree"]
    assert apply(pin, (members[229]/"patch.diff").read_bytes()) == base_229
    for k in (230,231,232,234):
        d = member_info[k]
        assert sha((members[k]/"patch.diff").read_bytes()) == d["patch_sha256"]
        assert sha((members[k]/"delta.diff").read_bytes()) == d["delta_sha256"]
        assert apply(pin, (members[k]/"patch.diff").read_bytes()) == d["tree"]
        ancestor = member_info[232 if k==234 else 229]["tree"]
        assert d["parent_tree"] == ancestor and apply(ancestor,(members[k]/"delta.diff").read_bytes()) == d["tree"]
    schedules = [order for order in permutations((230,231,232,234)) if order.index(232)<order.index(234)]
    assert len(schedules) == 12
    for schedule in schedules:
        tip = base_229
        for k in schedule:
            tip = apply(tip,(members[k]/"delta.diff").read_bytes())
        assert tip == tree
        for k in reversed(schedule):
            tip = apply(tip,(members[k]/"delta.diff").read_bytes(),True)
        assert tip == base_229
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
flags = previous["launch_flags"]
denied = [p for patterns in contract["launch_flags"]["denied"].values() for p in patterns]
assert not any(fnmatch.fnmatchcase(flag, pat) for flag in flags for pat in denied)
print(json.dumps(dict(candidate="V235", parent="V233", base=pin, parent_tree=before, tree=tree,
    parent_sha256=sha(parent), patch_sha256=sha(full), delta_sha256=sha(delta), bytes=len(full),
    delta_bytes=len(delta), paths=paths, changed_from_parent=sorted(CHANGED), launch_flags=flags,
    forbidden_flag_patterns=len(denied), max_added_line_chars=max(map(len, added)),
    strict_apply_reverse_full_delta=True, official_export="okc serve diff", official_rules=RULES_URL,
    donor_sha256=donor["patch_sha256"], legal_apply_orders=len(schedules), byte_budget=rules["max_patch_bytes"], physical_H200=False, submitted=False), indent=2))
