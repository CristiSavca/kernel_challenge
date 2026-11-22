"""Check both exact patch orders and the public serving constraints, using a scratch index."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
stage = Path(sys.argv[1])
anchor = here.parent / "v176_integrated_prompt_decode"
paired = here.parent / "v177_grouped_gate_up_epilogue"
peer = here.parent / "v179_peer_prompt_inputs"
base = "297e54fd105e974cdf401e5b4d20253a22c16f90"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(folder, file):
    return (folder / file).read_bytes()


with tempfile.TemporaryDirectory() as scratch:
    env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch) / "index"))

    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", stage, *args], input=data, env=env)

    def apply(tree, patch, reverse=False):
        git("read-tree", tree)
        args = ["--reverse"] if reverse else []
        git("apply", "--cached", "--whitespace=error-all", *args, "-", data=patch)
        return git("write-tree").decode().strip()

    def changed(first, second):
        return git("diff", "--name-only", first, second).decode().splitlines()

    originals = {p: json.loads(read(p, "validation.json")) for p in [anchor, paired, peer]}
    for path, manifest in originals.items():
        patch = read(path, "patch.diff")
        assert apply(base, patch) == manifest["tree"] and sha(patch) == manifest["patch_sha256"]

    full, delta = [read(here, f) for f in ("patch.diff", "delta.diff")]
    tree = apply(base, full)
    assert delta == read(peer, "delta.diff")
    bank, prompt = originals[paired]["tree"], originals[peer]["tree"]
    anchor_tree = originals[anchor]["tree"]
    assert apply(bank, delta) == apply(prompt, read(paired, "delta.diff")) == tree
    assert apply(tree, delta, True) == bank
    assert apply(tree, read(paired, "delta.diff"), True) == prompt
    assert apply(tree, full, True) == git("rev-parse", base + "^{tree}").decode().strip()

    donor_paths = {}
    for folder, manifest in [(paired, originals[paired]), (peer, originals[peer])]:
        donor_tree = manifest["tree"]
        mod = changed(anchor_tree, donor_tree)
        assert len(mod) == 4 and set(mod) == set(manifest["changed_from_parent"])
        assert apply(anchor_tree, read(folder, "delta.diff")) == donor_tree
        assert apply(donor_tree, read(folder, "delta.diff"), True) == anchor_tree
        assert manifest["parent_tree"] == anchor_tree and manifest["parent_sha256"] == originals[anchor]["patch_sha256"]
        donor_paths[manifest["candidate"]] = {}
        for name in mod:
            content = git("show", donor_tree + ":" + name)
            assert content == git("show", tree + ":" + name) == (stage / name).read_bytes()
            donor_paths[manifest["candidate"]][name] = sha(content)
    left, right = map(set, donor_paths.values())
    assert not left & right and set(changed(anchor_tree, tree)) == left | right

    if len(sys.argv) > 2:
        config = json.loads(Path(sys.argv[2]).read_bytes())
    else:
        config = json.loads(subprocess.check_output(["curl", "-fsSL", "https://openkernelchallenge.dev/api/v1/serving"]))
    rules = config["rules"]
    assert config["base"]["commit"] == base and len(full) <= rules["max_patch_bytes"]
    forbidden_headers = (b"rename ", b"old mode ", b"new mode ", b"GIT binary patch", b"Binary files ")
    assert not any(line.startswith(forbidden_headers) for line in full.splitlines())
    paths = changed(base, tree)
    for name in paths:
        assert name.startswith(rules["allowed_prefix"]) and not name.startswith(tuple(rules["forbidden_prefixes"]))
        assert name.endswith(tuple(rules["allowed_suffixes"]))
        content = git("show", tree + ":" + name)
        assert content == (stage / name).read_bytes()
        if name.endswith(".py"):
            compile(content, name, "exec")
    additions = [x[1:] for x in full.decode().splitlines() if x.startswith("+") and not x.startswith("+++")]
    longest = max(map(len, additions))
    assert longest <= rules["max_line_chars"] and not any(w in line for w in rules["forbidden_words"] for line in additions)
    denied = [v for group in config["launch_flags"]["denied"].values() for v in group]
    flags = originals[paired]["launch_flags"]
    assert all(m["launch_flags"] == flags for m in originals.values())
    assert not any(fnmatch.fnmatchcase(k, v) for k in flags for v in denied)
    joint = json.loads(read(here, "joint.json"))
    for rank in (4, 8):
        assert joint["gloo"][str(rank)]["joint"]["peer_w13_return_calls"] == rank
    print(json.dumps(dict(
        candidate="V181", base=base, parent="V177", parent_sha256=originals[paired]["patch_sha256"],
        parent_tree=bank, donor="V179", donor_sha256=originals[peer]["patch_sha256"], donor_tree=prompt,
        anchor="V176", anchor_sha256=originals[anchor]["patch_sha256"], anchor_tree=anchor_tree,
        tree=tree, patch_sha256=sha(full), bytes=len(full), delta_sha256=sha(delta), delta_bytes=len(delta),
        launch_flags=flags, paths=paths, donor_files=donor_paths, checked_flag_patterns=len(denied),
        max_added_line_chars=longest, official_export="okc serve diff .", strict_apply_reverse_full_delta=True,
        same_tree_both_donor_orders=True, changed_from_parent=changed(bank, tree), joint_receipt_sha256=sha(read(here, "joint.json")),
        hardware_measured=False, submitted=False
    ), indent=2))
