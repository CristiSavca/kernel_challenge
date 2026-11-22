"""Validate the official V241 export, every component and the public source contract."""
import fnmatch
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
ENDPOINT = "https://openkernelchallenge.dev/api/v1/serving"
PARENT = 235
DONORS = (236, 237, 239, 240)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def files(version):
    folder = next(HERE.parent.glob(f"v{version}_*"))
    return folder, json.loads((folder/"validation.json").read_bytes())


contract = json.loads(subprocess.check_output(["curl","-fsSL",ENDPOINT]))
rules, pin = contract["rules"], contract["base"]["commit"]
manifests = {v: files(v) for v in (PARENT,*DONORS)}
parent_dir, parent_manifest = manifests[PARENT]
pieces = {}
full, change = [(HERE/name).read_bytes() for name in ("patch.diff", "delta.diff")]
with tempfile.TemporaryDirectory() as scratch:
    env = dict(os.environ,GIT_INDEX_FILE=str(Path(scratch)/"index"))

    def git(*args, data=None):
        return subprocess.check_output(["git","-C",str(STAGE),*args],input=data,env=env)

    def apply(tree, patch, reverse=False):
        git("read-tree",tree)
        git("apply","--cached","--whitespace=error-all",*(["--reverse"] if reverse else []),"-",data=patch)
        return git("write-tree").decode().strip()

    native = git("rev-parse", pin+"^{tree}").decode().strip()
    for v,(folder,m) in manifests.items():
        blob = (folder/"patch.diff").read_bytes()
        assert m["base"] == pin and sha(blob)==m["patch_sha256"]
        assert apply(pin,blob)==m["tree"] and apply(m["tree"],blob,True)==native
        assert m["launch_flags"]==parent_manifest["launch_flags"]
        if v != PARENT:
            delta = (folder/"delta.diff").read_bytes()
            base = manifests[239 if v==240 else PARENT][1]["tree"]
            assert base == m["parent_tree"] and sha(delta)==m["delta_sha256"]
            assert apply(base,delta)==m["tree"] and apply(m["tree"],delta,True)==base
            pieces[v] = delta
    before = parent_manifest["tree"]
    tree = apply(pin,full)
    assert apply(tree,full,True)==native and apply(before,change)==tree and apply(tree,change,True)==before
    orders = [o for o in permutations(DONORS) if o.index(239)<o.index(240)]
    for order in orders:
        pos = before
        for v in order:
            pos=apply(pos,pieces[v])
        assert pos==tree
        for v in reversed(order):
            pos=apply(pos,pieces[v],True)
        assert pos==before
    paths=git("diff","--name-only",native,tree).decode().splitlines()
    changed=git("diff","--name-only",before,tree).decode().splitlines()
    expected={p for v in DONORS for p in manifests[v][1]["changed_from_parent"]}
    assert set(changed)==expected
    for path in paths:
        assert path.startswith(rules["allowed_prefix"]) and path.endswith(tuple(rules["allowed_suffixes"]))
        assert not any(path.startswith(prefix) for prefix in rules["forbidden_prefixes"])
        old,new=git("ls-tree",native,path),git("ls-tree",tree,path)
        mode=old.split()[0] if old else b"100644"
        assert mode in (b"100644",b"100755") and new.startswith(mode+b" blob ")
        data=git("show",tree+":"+path)
        assert data and bytes(1) not in data and data==(STAGE/path).read_bytes()
        if path.endswith(".py"):
            compile(data,path,"exec")
        if path in expected:
            owners = {v for v in DONORS if path in manifests[v][1]["changed_from_parent"]}
            if 240 in owners:
                owners.discard(239)
            if len(owners)==1:
                owner=owners.pop()
                assert data==git("show",manifests[owner][1]["tree"]+":"+path),(owner,path)
    # The release differs from a three-context export only in unchanged surrounding lines.
    default=git("-c","diff.context=3","diff","--no-renames",pin,tree)
    assert apply(pin,default)==tree
lines=full.decode().splitlines()
assert not any(s.startswith(("rename from ","rename to ","old mode ","new mode ","GIT binary patch","Binary files")) for s in lines)
added=[s[1:] for s in lines if s.startswith("+") and not s.startswith("+++ ")]
assert max(map(len,added))<=rules["max_line_chars"] and len(full)<=rules["max_patch_bytes"]
assert not any(word in line for line in added for word in rules["forbidden_words"])
denied = [p for patterns in contract["launch_flags"]["denied"].values() for p in patterns]
flags = parent_manifest["launch_flags"]
assert not any(fnmatch.fnmatchcase(flag,pat) for flag in flags for pat in denied)
print(json.dumps(dict(candidate="V241",parent="V235",base=pin,parent_tree=before,tree=tree,parent_sha256=parent_manifest["patch_sha256"],
    patch_sha256=sha(full),delta_sha256=sha(change),bytes=len(full),delta_bytes=len(change),paths=paths,changed_from_parent=changed,
    launch_flags=flags,forbidden_flag_patterns=len(denied),max_added_line_chars=max(map(len,added)),strict_apply_reverse_full_delta=True,
    official_export="okc serve diff",export_context_lines=2,three_context_identical_tree=True,three_context_bytes=len(default),
    legal_apply_orders=len(orders),donor_patches={f"V{v}":manifests[v][1]["patch_sha256"] for v in DONORS},
    official_rules=ENDPOINT,byte_budget=rules["max_patch_bytes"],physical_H200=False,submitted=False),indent=2))

