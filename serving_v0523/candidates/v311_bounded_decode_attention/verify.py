"""Check the exported patch, public contract and host evidence from the exact tree."""
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
name, prior = "V311", "V204"
parent = next(here.parent.glob(prior.lower()+"_*"))
stage = Path(sys.argv[1]).resolve()
rules_url = "https://openkernelchallenge.dev/api/v1/serving"
contract = json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv) > 2 else
                      subprocess.check_output(["curl", "-fsSL", rules_url]))
limits = contract["rules"]
base = contract["base"]["commit"]
before = json.loads((parent/"validation.json").read_bytes())
read_parent_flags = json.loads((parent / "flags.json").read_bytes())["flags"]
prefix = "python/sglang/"
attention = prefix + "kernels/ops/attention/paged_verify_attention.py"
changed_expected = {attention}


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
    sources = {p: sha(git("show",tree+":"+p)) for p in changed}
    def read(n):
        return json.loads((here/(n+".json")).read_bytes())
    flags_file = read("flags")
    assert flags == flags_file == read_parent_flags and len(flags) == 6
    dispatch = read("dispatch")
    assert sources[attention] == dispatch["sources_sha256"]["v311"]
    assert sha(git("show",previous+":"+attention)) == dispatch["sources_sha256"]["v204"]
    assert dispatch["counts"]["dispatch_pairs"] == 3000
    assert dispatch["counts"]["identical_partial_intersections"] > 60000
    assert dispatch["counts"]["full_forward_extend_pairs"] == 40
    oracle = read("interpreter")
    hopper = read("sm90")
    source_pair = dict(parent=dispatch["sources_sha256"]["v204"], candidate=sources[attention])
    assert oracle["source_sha256"] == hopper["source_sha256"] == source_pair
    assert oracle["total"]["bf16_words"] == 236544 and oracle["total"]["fp32_words"] == 1946880
    assert len(hopper["builds"]) == 50
    serving = here.parents[1]
    job = "ad275bf67d9a4a0a"
    receipt = serving / "runs" / (job + ".json")
    if receipt.exists():
        scored = json.loads(receipt.read_bytes())
        assert scored["result"]["passed"] and scored["result"]["patch"]["sha256"] == sha(old)
        assert scored["serving_flags"] == flags
    else:
        public = json.loads((serving / "results.json").read_bytes())
        row, = [run for run in public["jobs"] if run["id"] == job]
        assert (row["candidate"], row["kind"], row["result"]) == ("V204", "scored", "ok")
        assert row["flags"] == flags and public["patches"]["V204"]["sha256"] == sha(old)
    assert len(paths) == len(before["paths"])
    evidence = {p.name:sha(p.read_bytes()) for p in sorted(here.glob("*.json"))
                if p.stem not in ("validation","judge_observation","cleanup") and not p.stem.endswith("cleanup")}
    print(json.dumps(dict(candidate=name,parent=prior,base=base,parent_tree=previous,tree=tree,
        parent_sha256=sha(old),patch_sha256=sha(full),delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),
        paths=paths,changed_from_parent=changed,sources=sources,evidence_sha256=evidence,launch_flags=flags,
        max_added_line_chars=max(map(len,added)),byte_budget=limits["max_patch_bytes"],strict_apply_reverse_full_delta=True,
        official_export="okc serve diff, Git diff context 1",official_rules=rules_url,physical_H200=False,submitted=False),indent=2))
