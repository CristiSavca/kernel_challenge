"""Check the public patch, standalone export, fallback and real Hopper machine evidence."""
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
name, prior = "V302", "V299"
parent = next(here.parent.glob(prior.lower()+"_*"))
stage = Path(sys.argv[1]).resolve()
rules_url = "https://openkernelchallenge.dev/api/v1/serving"
contract = json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv) > 2 else
                      subprocess.check_output(["curl", "-fsSL", rules_url]))
limits = contract["rules"]
base = contract["base"]["commit"]
before = json.loads((parent/"validation.json").read_bytes())
changed_expected = {
    "python/sglang/kernels/jit/csrc/muse_spark/grouped_decode_mma_pair.cuh",
    "python/sglang/srt/layers/muse_spark/decode_mma_pair.py",
    "python/sglang/srt/layers/muse_spark/residual_prompt_mma.py",
    "python/sglang/kernels/ops/attention/paged_verify_attention.py",
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

with tempfile.TemporaryDirectory() as scratch:
    obj = Path(scratch) / "objects"
    obj.mkdir()
    source = subprocess.check_output(["git", "-C", stage, "rev-parse", "--absolute-git-dir"], text=True).strip()
    env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch)/"index"), GIT_OBJECT_DIRECTORY=str(obj),
               GIT_ALTERNATE_OBJECT_DIRECTORIES=source+"/objects")
    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", stage, *args], env=env, input=data)

    def apply(tree, patch, reverse=False):
        git("read-tree", tree)
        git("apply", "--cached", "--whitespace=error-all", *(["-R"] if reverse else []), "-", data=patch)
        return git("write-tree").decode().strip()

    old = (parent/"patch.diff").read_bytes()
    full = (here/"patch.diff").read_bytes()
    delta = (here/"delta.diff").read_bytes()
    native = git("rev-parse", base+"^{tree}").decode().strip()
    previous, tree = apply(base, old), apply(base, full)
    assert before["base"] == base and sha(old) == before["patch_sha256"] and previous == before["tree"]
    assert apply(previous,delta) == tree and apply(tree,delta,True) == previous
    assert apply(tree,full,True) == native and git("diff","--no-renames","-U1",base,tree) == full
    changed = git("diff","--name-only",previous,tree).decode().splitlines()
    assert set(changed) == changed_expected
    paths = git("diff","--name-only",base,tree).decode().splitlines()
    for path in paths:
        assert path.startswith(limits["allowed_prefix"]) and not path.startswith(tuple(limits["forbidden_prefixes"]))
        assert path.endswith(tuple(limits["allowed_suffixes"]))
        entry = git("ls-tree",tree,path)
        old_entry = git("ls-tree",base,path)
        mode = old_entry.split()[0] if old_entry else b"100644"
        assert mode in (b"100644", b"100755") and entry.startswith(mode+b" blob ")
        data = git("show",tree+":"+path)
        assert data and b"\0" not in data
        if path.endswith(".py"):
            ast.parse(data)
    added = [line[1:] for line in full.decode().splitlines() if line.startswith("+") and not line.startswith("+++")]
    assert len(full) <= limits["max_patch_bytes"] and max(map(len,added)) <= limits["max_line_chars"]
    assert not any(w in line for line in added for w in limits["forbidden_words"])
    assert not any(line.startswith(("rename ", "old mode ", "new mode ", "GIT binary patch", "Binary files"))
                   for line in full.decode().splitlines())
    flags = before["launch_flags"]
    denied = [pattern for patterns in contract["launch_flags"]["denied"].values() for pattern in patterns]
    assert not any(fnmatch.fnmatchcase(flag, pat) for flag in flags for pat in denied)
    assert dict((row.split()[0], " ".join(row.split()[1:])) for row in (here/"flags").read_text().splitlines()) == flags
    sources = {p: sha(git("show",tree+":"+p)) for p in changed}
    def read(n):
        return json.loads((here/(n+".json")).read_bytes())
    closure=read("expert_closure")
    assert closure["tree"] == tree and closure["all_cubins_and_sass_identical"] and len(closure["cases"]) == 12
    assert closure["source_sha256"] == sources
    machine=json.loads((parent/"machine.json").read_bytes())
    for stem,image in closure["cases"].items():
        if stem in machine["m8"]:
            target=machine["m8"][stem]["selected"]
            assert image["cubin"] == target["cubin"] and image["sass"] == target["sass"]
        else:
            target=machine["unchanged_machine_schedules"][stem]
            assert image["cubin"] == target["cubin_sha256"] and image["sass"] == target["sass_sha256"]
    assert (here/"startup.json").read_bytes() == (parent/"startup.json").read_bytes()
    attn=next(p for p in changed if p.endswith("/paged_verify_attention.py"))
    a301=next(here.parent.glob("v301_*"))
    donor=apply(base,(a301/"patch.diff").read_bytes())
    assert git("show",tree+":"+attn) == git("show",donor+":"+attn)
    attn_data={}
    for name in ("sm90","capture_sm90","independent_runtime","interpreter"):
        contents=(a301/(name+".json")).read_bytes()
        d=json.loads(contents)
        assert d["source_sha256"]["candidate"] == sources[attn]
        attn_data[name]=sha(contents)
        if name=="sm90": assert len(d["builds"])==54
        if name=="capture_sm90": assert d["capture_geometry"] and len(d["builds"])==48
        if name=="interpreter": assert len(d["cases"])==30
    integrated=read("integrated_dispatch")
    assert integrated["tree"]==tree and integrated["experts"]["calls"]==56
    assert read("attention_runtime")["source_sha256"]["candidate"] == sources[attn]
    evidence={p.name:sha(p.read_bytes()) for p in sorted(here.glob("*.json")) if p.stem not in ("validation","cleanup")}
    print(json.dumps(dict(candidate="V302",parent="V299",base=base,parent_tree=previous,tree=tree,
        parent_sha256=sha(old),patch_sha256=sha(full),delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),
        paths=paths,changed_from_parent=changed,sources=sources,evidence_sha256=evidence,launch_flags=flags,
        max_added_line_chars=max(map(len,added)),byte_budget=limits["max_patch_bytes"],attention_donor_tree=donor,
        attention_evidence_sha256=attn_data,strict_apply_reverse_full_delta=True,official_export="okc serve diff, Git diff context 1",
        official_rules=rules_url,physical_H200=False,submitted=False),indent=2))
