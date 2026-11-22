"""Check the public patch, reversible export, conditional wave geometry and native build evidence."""
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
name, prior = "V306", "V197"
parent = next(here.parent.glob(prior.lower()+"_*"))
stage = Path(sys.argv[1]).resolve()
rules_url = "https://openkernelchallenge.dev/api/v1/serving"
contract = json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv) > 2 else
                      subprocess.check_output(["curl", "-fsSL", rules_url]))
limits = contract["rules"]
base = contract["base"]["commit"]
before = json.loads((parent/"validation.json").read_bytes())
prefix = "python/sglang/"
layers = prefix+"srt/layers/muse_spark/"
wave = layers+"humming_prefill_waves.py"
binding = layers+"decode_gate_up.py"
native_launcher = prefix+"kernels/jit/csrc/muse_spark/decode_gate_up_launch.cuh"
changed_expected = {native_launcher, binding, wave, layers+"muse_spark_fp4_moe.py", layers+"muse_spark_humming_moe.py",
    prefix+"kernels/ops/moe/muse_spark_post_expert_finalize_triton.py", prefix+"srt/environ.py",
    prefix+"srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py"}

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
    replay, interpreter, compiled, ffi = [read(k) for k in ("replay","packer_interpreter","packer_sm90","generated_ffi")]
    assert replay["wave_sha256"] == interpreter["source_sha256"] == read("work")["source_sha256"] == sources[wave]
    assert len(compiled["cases"]) == 18
    for entry in compiled["cases"]:
        assert entry["source_sha256"] == sources[wave] and len(entry["cubin_sha256"]) == 64
        assert any("0 bytes spill stores, 0 bytes spill loads" in line for line in entry["assembler_stats"])
    assert len(ffi["exports"]) == 7 and ffi["launcher_sha256"] == read("epilogue")["launcher_sha256"] == sources[native_launcher]
    assert ffi["python_sha256"] == read("binding")["source_sha256"] == sources[binding]
    assert replay["counts"]["cases"]==720 and interpreter["summary"]["verified_payload_words"]==21739116
    assert read("entry")["passed"] and read("eager_prompt")["bf16_output_components"]==5772800
    assert all(x["mean_waves"][1]==1 for x in read("work")["shapes"])
    evidence = {p.name:sha(p.read_bytes()) for p in sorted(here.glob("*.json"))
                if p.stem not in ("validation","judge_observation","cleanup") and not p.stem.endswith("cleanup")}
    print(json.dumps(dict(candidate=name,parent=prior,base=base,parent_tree=previous,tree=tree,
        parent_sha256=sha(old),patch_sha256=sha(full),delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),
        paths=paths,changed_from_parent=changed,sources=sources,evidence_sha256=evidence,launch_flags=flags,
        max_added_line_chars=max(map(len,added)),byte_budget=limits["max_patch_bytes"],strict_apply_reverse_full_delta=True,
        official_export="okc serve diff, Git diff context 1",official_rules=rules_url,physical_H200=False,submitted=False),indent=2))
