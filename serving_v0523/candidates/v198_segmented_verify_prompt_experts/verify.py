"""Verify the official patch, both donor orders, their payloads and the serving contract."""
from fnmatch import fnmatchcase
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
URL = "https://openkernelchallenge.dev/api/v1/serving"
COMMON = HERE.parent / "v195_fused_expert_affine_verify"
PROMPT = HERE.parent / "v197_grouped_prompt_gate_up"
ATTENTION = HERE.parent / "v196_segmented_verify_tma"


def digest(data):
    return sha256(data).hexdigest()


def load_export(folder):
    meta = json.loads((folder / "validation.json").read_bytes())
    full = (folder / "patch.diff").read_bytes()
    delta = (folder / "delta.diff").read_bytes()
    assert digest(full) == meta["patch_sha256"] and digest(delta) == meta["delta_sha256"]
    return meta, full, delta


contract = json.loads(subprocess.check_output(["curl", "-fsSL", URL]))
rules = contract["rules"]
pin = contract["base"]["commit"]
common, prompt, attention = [load_export(f) for f in (COMMON, PROMPT, ATTENTION)]
flags = common[0]["launch_flags"]
for donor in (prompt, attention):
    assert donor[0]["base"] == common[0]["base"] == pin
    assert donor[0]["launch_flags"] == flags
    assert donor[0]["parent_tree"] == common[0]["tree"]
full = (HERE / "patch.diff").read_bytes()
delta = (HERE / "delta.diff").read_bytes()
with tempfile.TemporaryDirectory() as tmp:
    env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp)/"index")}

    def git(*args, data=None):
        return subprocess.check_output(["git","-C",str(STAGE),*args],input=data,env=env)

    def apply(tree, data, reverse=False):
        git("read-tree", tree)
        git("apply","--cached","--whitespace=error-all",*(["--reverse"] if reverse else []),"-",data=data)
        return git("write-tree").decode().strip()

    native = git("rev-parse",pin+"^{tree}").decode().strip()
    before = apply(pin, common[1])
    assert before == common[0]["tree"]
    result = apply(pin, full)
    assert apply(result, full, True) == native
    signed = {}
    for donor, sibling in ((prompt, attention), (attention, prompt)):
        target = donor[0]["tree"]
        assert apply(pin, donor[1]) == target
        assert apply(target, donor[1], True) == native
        assert apply(before, donor[2]) == target and apply(target, donor[2], True) == before
        assert apply(target, sibling[2]) == result and apply(result,sibling[2],True) == target
        changes = git("diff","--name-only",before,target).decode().splitlines()
        signed[donor[0]["candidate"]] = {path:digest(git("show",target+":"+path)) for path in changes}
        for path, expected in signed[donor[0]["candidate"]].items():
            assert digest(git("show",result+":"+path)) == expected == digest((STAGE/path).read_bytes())
    assert signed["V196"].keys() == {"python/sglang/kernels/ops/attention/paged_verify_attention.py"}
    assert signed["V196"].keys().isdisjoint(signed["V197"])
    assert set(git("diff","--name-only",before,result).decode().splitlines()) == signed["V196"].keys() | signed["V197"].keys()
    assert apply(prompt[0]["tree"],delta) == result and apply(result,delta,True) == prompt[0]["tree"]
    assert git("write-tree").decode().strip() == prompt[0]["tree"]
    paths = git("diff","--name-only",native,result).decode().splitlines()
    for path in paths:
        assert path.startswith(rules["allowed_prefix"]) and not path.startswith(tuple(rules["forbidden_prefixes"]))
        assert path.endswith(tuple(rules["allowed_suffixes"]))
        old, new = git("ls-tree",native,path), git("ls-tree",result,path)
        mode = old.split()[0] if old else b"100644"
        assert mode in {b"100644",b"100755"} and new.startswith(mode+b" blob ")
        content = git("show", result+":"+path)
        assert content == (STAGE/path).read_bytes() and content and b"\0" not in content
        if path.endswith(".py"):
            compile(content, path, "exec")

lines = full.decode().splitlines()
assert not any(line.startswith(("rename ","old mode ","new mode ","GIT binary patch","Binary files ")) for line in lines)
added = [line[1:] for line in lines if line.startswith("+") and not line.startswith("+++ ")]
assert len(full) <= rules["max_patch_bytes"] and max(map(len,added)) <= rules["max_line_chars"]
assert not any(word in line for word in rules["forbidden_words"] for line in added)
denied = [p for patterns in contract["launch_flags"]["denied"].values() for p in patterns]
assert not any(fnmatchcase(flag,pattern) for flag in flags for pattern in denied)
print(json.dumps(dict(candidate="V198", base=pin, parent="V197", common_parent="V195", attention_donor="V196",
    parent_tree=prompt[0]["tree"], parent_sha256=digest(prompt[1]), common_tree=before,
    tree=result, bytes=len(full), patch_sha256=digest(full), delta_sha256=digest(delta), delta_bytes=len(delta),
    launch_flags=flags, paths=paths, changed_from_parent=sorted(signed["V196"]), exact_donor_sources=signed,
    strict_apply_reverse_full_delta=True, strict_both_donor_orders=True, source_and_modes_verified=True,
    official_export="okc serve diff", official_rules=URL, checked_denied_flag_patterns=len(denied),
    max_added_line_chars=max(map(len,added)), physical_h200=False, submitted=False),indent=2))
