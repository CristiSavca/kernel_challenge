"""Check the public patch, reversible export, runtime partition geometry and Hopper compiler evidence."""
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
name, prior = "V301", "V297"
parent = next(here.parent.glob(prior.lower()+"_*"))
stage = Path(sys.argv[1]).resolve()
rules_url = "https://openkernelchallenge.dev/api/v1/serving"
contract = json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv) > 2 else
                      subprocess.check_output(["curl", "-fsSL", rules_url]))
limits = contract["rules"]
base = contract["base"]["commit"]
before = json.loads((parent/"validation.json").read_bytes())
kernel = "python/sglang/kernels/ops/attention/paged_verify_attention.py"
changed_expected = {kernel}

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
    builds, numeric, runtime = [read(n) for n in ("sm90", "interpreter", "independent_runtime")]
    source_pair = dict(parent=sha(git("show", previous+":"+kernel)), candidate=sources[kernel])
    assert numeric["source_sha256"] == builds["source_sha256"] == source_pair
    assert all(runtime["source_sha256"][k] == digest for k,digest in source_pair.items())
    backend = "python/sglang/srt/layers/attention/flashattention_backend.py"
    assert runtime["source_sha256"]["backend"] == sha(git("show", tree+":"+backend))
    assert len(builds["builds"]) == 54 and all(
        image["spill_store_load_bytes"] == [0,0] for key,image in builds["builds"].items() if key.endswith("/candidate"))
    assert builds["builds"]["b4/w-1/c131072/partition/direct/parent"]["spill_store_load_bytes"] == [28,36]
    for b in (4,16,32):
        prefix = f"b{b}/w2047/c131072/partition/direct/"
        parent_build = builds["builds"][prefix+"parent"]; current = builds["builds"][prefix+"candidate"]
        for key in ("sass_instructions","registers","static_branches","global_loads","tensor_map_loads"):
            assert parent_build[key] == current[key], (b,key)
    capture = read("capture_sm90")
    assert capture["capture_geometry"] and capture["source_sha256"] == source_pair and len(capture["builds"]) == 48
    assert capture["page_table_pitch"]==131088 and (capture["full_descriptor_rows"], capture["sliding_descriptor_rows"])==(5005915,1251479)
    assert not read("capacity_sensitivity")["kernel_build_differences_by_metric"]
    images=capture["builds"]
    for b,branch_counts in ((1,(30,14)),(4,(44,17))):
        prefix=f"b{b}/w-1/c131088/partition/direct/"
        assert tuple(images[prefix+ver]["static_branches"] for ver in ("parent","candidate")) == branch_counts
    for ver in ("parent","candidate"):
        assert images["b16/w-1/c131088/partition/direct/"+ver]["spill_store_load_bytes"] == [32,44]
    assert len(numeric["cases"]) == 30 and numeric["total"]["dense_words"] == 184320
    assert numeric["max_abs_fp64"] < .01 and runtime["counts"]["identical_intersections"] == 68955
    assert runtime["public_max_old_full_scratch"] < runtime["public_min_mixed_budget"]
    assert before["launch_flags"] == flags
    assert len(paths) == len(before["paths"]) == 61
    evidence={p.name:sha(p.read_bytes()) for p in sorted(here.glob("*.json")) if p.stem not in ("validation","cleanup") and not p.stem.endswith("_cleanup")}
    print(json.dumps(dict(candidate=name,parent=prior,base=base,parent_tree=previous,tree=tree,
        parent_sha256=sha(old),patch_sha256=sha(full),delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),
        paths=paths,changed_from_parent=changed,sources=sources,evidence_sha256=evidence,launch_flags=flags,
        max_added_line_chars=max(map(len,added)),byte_budget=limits["max_patch_bytes"],
        strict_apply_reverse_full_delta=True,official_export="okc serve diff, Git diff context 1",official_rules=rules_url,
        physical_H200=False,submitted=False), indent=2))
