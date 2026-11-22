"""Validate V289 against the public contract and reconstruct both patch directions."""
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
parent = here.parent / "v284_rank_major_routed_pool"
stage = Path(sys.argv[1]).resolve()
rules_url = "https://openkernelchallenge.dev/api/v1/serving"
contract = json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv) > 2 else
                      subprocess.check_output(["curl", "-fsSL", rules_url]))
limits = contract["rules"]
base = contract["base"]["commit"]
before = json.loads((parent / "validation.json").read_bytes())
kernel = "python/sglang/kernels/jit/csrc/muse_spark/ranked_pool_norm.cuh"
changed_expected = {"python/sglang/srt/" + p for p in ("layers/muse_spark/muse_spark_fp4_moe.py",
    "layers/muse_spark/muse_spark_humming_moe.py","models/muse_spark.py")}

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
    sources = {p:sha(git("show",tree+":"+p)) for p in changed}
    def result(name):
        return json.loads((here/(name+".json")).read_text())
    replay, capture, parsed = (result(n) for n in ("replay","capture_gloo","flags"))
    for hashes in (replay["hashes"], capture["sources"]):
        assert all(sha(git("show",tree+":"+p)) == digest for p,digest in hashes.items())
    assert replay["early_plans"] == 14 and replay["compared_calls"] == 34
    assert capture["rank_conditions"] == 204
    assert parsed["tree"] == tree and parsed["parent"] == previous and parsed["flags"] == flags
    assert parsed["parsed_orderings"] == 720
    assert sha(git("show",tree+":"+kernel)) == before["sources"][kernel]
    hpath = next(p for p in changed if p.endswith("humming_moe.py"))
    def function(ref, name):
        module = ast.parse(git("show",ref+":"+hpath))
        return next(n for n in module.body if getattr(n,"name",None) == name)
    assert ast.dump(function(tree,"_plan_grouped_decode")) == ast.dump(function(previous,"_plan_grouped_decode"))
    def launches(n):
        return [ast.dump(c) for c in ast.walk(n) if isinstance(c,ast.Call) and isinstance(c.func,ast.Subscript)
            and isinstance(c.func.value,ast.Name) and c.func.value.id == "_plan_grouped_decode"]
    assert len(launches(function(tree,"fill_decode_plan"))) == 1
    assert launches(function(tree,"fill_decode_plan")) == launches(function(previous,"humming_grouped_decode_outputs"))
    assert ast.dump(function(tree,"prepare_ranked_pool")) == ast.dump(function(previous,"prepare_ranked_pool"))
    print(json.dumps(dict(candidate="V289",parent="V284",base=base,parent_tree=previous,tree=tree,
        parent_sha256=sha(old),patch_sha256=sha(full),delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),
        paths=paths,changed_from_parent=changed,sources=sources,launch_flags=flags,added_line_maximum=max(map(len,added)),
        strict_apply_reverse_full_delta=True,official_export="okc serve diff, Git diff context 1",official_rules=rules_url,
        planner_kernel_and_launch_unchanged=True,ranked_norm_device_source_unchanged=True,
        physical_H200=False,submitted=False),indent=2))
