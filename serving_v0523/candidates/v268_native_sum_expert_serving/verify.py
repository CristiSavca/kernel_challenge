"""Reconstruct the combined serving patch from the judge pin and the native output reduction."""
import ast
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from itertools import permutations
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
ENDPOINT = "https://openkernelchallenge.dev/api/v1/serving"
PARTS = (257, 258, 260, 261, 263)
MODEL = "python/sglang/srt/models/muse_spark.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def files(num):
    folder = next(HERE.parent.glob(f"v{num}_*"))
    return folder, json.loads((folder / "validation.json").read_bytes())


contract = json.loads(subprocess.check_output(["curl", "-fsSL", ENDPOINT]))
rules, pin = contract["rules"], contract["base"]["commit"]
m = {n: files(n) for n in (248, 256, *PARTS)}
full, delta, integration = [(HERE / f).read_bytes() for f in ("patch.diff", "delta.diff", "integration.diff")]
with tempfile.TemporaryDirectory() as tmp:
    env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp)/"index")}

    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", str(STAGE), *args], env=env, input=data)

    def apply(tree, patch, reverse=False):
        git("read-tree", tree)
        opts = (["--reverse"] if reverse else [])
        git("apply", "--cached", "--whitespace=error-all", *opts, "-", data=patch)
        return git("write-tree").decode().strip()

    pin_tree = git("rev-parse", pin + "^{tree}").decode().strip()
    base = m[248][1]["tree"]
    flags = m[248][1]["launch_flags"]
    pieces = {}
    for n, (folder, manifest) in m.items():
        source = (folder / "patch.diff").read_bytes()
        assert manifest["base"] == pin and digest(source) == manifest["patch_sha256"]
        assert apply(pin, source) == manifest["tree"] and apply(manifest["tree"], source, True) == pin_tree
        if n == 257:
            # V257 recorded just the four shared server options. Its arithmetic delta changes no configuration.
            assert len(manifest["launch_flags"]) == 4 and manifest["launch_flags"].items() <= flags.items()
        else:
            assert flags == manifest["launch_flags"]
        if n in PARTS:
            pieces[n] = (folder / "delta.diff").read_bytes()
            parent = m[261 if n == 263 else 256][1]["tree"]
            assert digest(pieces[n]) == manifest["delta_sha256"] and manifest["parent_tree"] == parent
            assert apply(parent, pieces[n]) == manifest["tree"] and apply(manifest["tree"], pieces[n], True) == parent
    tree = apply(pin, full)
    assert apply(tree, full, True) == pin_tree and apply(base, delta) == tree and apply(tree, delta, True) == base
    # All component edits land mechanically. Only the QKV method and weight guards differ.
    mechanical = apply(tree, integration, True)
    assert apply(mechanical, integration) == tree
    orders = [p for p in permutations(PARTS) if p.index(261) < p.index(263)]
    for order in orders:
        mixed = base
        for n in order:
            mixed = apply(mixed, pieces[n])
        assert mixed == mechanical
        for n in reversed(order):
            mixed = apply(mixed, pieces[n], reverse=True)
        assert mixed == base
    paths = git("diff", "--name-only", pin_tree, tree).decode().splitlines()
    changed = git("diff", "--name-only", base, tree).decode().splitlines()
    integrated = git("diff", "--name-only", mechanical, tree).decode().splitlines()
    assert set(changed) == {p for n in PARTS for p in m[n][1]["changed_from_parent"]}
    native_o = "python/sglang/srt/layers/layernorm_sp.py"
    def read_fn(ref, name):
        module = ast.parse(git("show", ref+":"+native_o))
        return ast.dump(next(n for n in module.body if getattr(n,"name","")==name))
    assert read_fn(base, "row_parallel_gbar_matmul") == read_fn(tree, "row_parallel_gbar_matmul")
    assert not git("ls-tree", tree, "python/sglang/srt/layers/communication/sp_attention_return.py")
    assert set(integrated) == {native_o,"python/sglang/srt/layers/communication/sp_half_qkv.py"}
    for path in paths:
        assert path.startswith(rules["allowed_prefix"]) and path.endswith(tuple(rules["allowed_suffixes"]))
        assert not any(path.startswith(p) for p in rules["forbidden_prefixes"])
        old, new = git("ls-tree", pin_tree, path), git("ls-tree", tree, path)
        mode = old.split()[0] if old else b"100644"
        assert mode in (b"100644", b"100755") and new.startswith(mode + b" blob ")
        data = git("show", tree + ":" + path)
        assert data and b"\0" not in data and data == (STAGE / path).read_bytes()
        if path.endswith(".py"):
            compile(data, path, "exec")
    assert git("diff", "-U1", "--no-renames", pin, tree) == full
lines = full.decode().splitlines()
assert not any(s.startswith(("rename from ", "rename to ", "old mode ", "new mode ", "GIT binary patch", "Binary files")) for s in lines)
added = [s[1:] for s in lines if s.startswith("+") and not s.startswith("+++ ")]
assert len(full) <= rules["max_patch_bytes"] and max(map(len, added)) <= rules["max_line_chars"]
assert not any(word in line for line in added for word in rules["forbidden_words"])
patterns = [p for group in contract["launch_flags"]["denied"].values() for p in group]
assert not any(fnmatch.fnmatchcase(flag, pattern) for flag in flags for pattern in patterns)
print(json.dumps(dict(candidate="V268", parent="V248", base=pin, parent_tree=base, tree=tree,
                      parent_sha256=m[248][1]["patch_sha256"], bytes=len(full), delta_bytes=len(delta),
                      patch_sha256=digest(full), delta_sha256=digest(delta), integration_sha256=digest(integration),
                      integration_bytes=len(integration), integration_files=integrated, mechanical_tree=mechanical,
                      paths=paths, changed_from_parent=changed, legal_apply_orders=len(orders),
                      launch_flags=flags, max_added_line_chars=max(map(len, added)), forbidden_flag_patterns=len(patterns),
                      strict_apply_reverse_full_delta=True, official_export="okc serve diff",
                      official_rules=ENDPOINT, byte_budget=rules["max_patch_bytes"], physical_H200=False,
                      submitted=False),indent=2))
