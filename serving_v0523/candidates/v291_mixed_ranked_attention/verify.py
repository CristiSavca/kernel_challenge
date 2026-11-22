"""Validate V291 against the public contract and bind its three attention donors."""
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
parent = here.parent / "v290_mixed_ranked_peer_serving"
stage = Path(sys.argv[1]).resolve()
rules_url = "https://openkernelchallenge.dev/api/v1/serving"
contract = json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv) > 2 else
                      subprocess.check_output(["curl", "-fsSL", rules_url]))
limits = contract["rules"]
base = contract["base"]["commit"]
before = json.loads((parent / "validation.json").read_bytes())
field = "python/sglang/srt/model_executor/forward_batch_info.py"
changed_expected = {field, *["python/sglang/"+p for p in (
    "srt/layers/attention/direct_prompt_kv.py","srt/layers/attention/flashattention_backend.py",
    "srt/model_executor/runner/prefill_cuda_graph_runner.py","srt/speculative/dflash_mixed_forward.py",
    "kernels/ops/attention/paged_verify_attention.py")]}

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
    components = result("components")
    component_patches, component_deltas = {}, {}
    last = None
    for version in (233,235,239,240):
        donor = next(here.parent.glob("v"+str(version)+"_*"))
        dv = json.loads((donor/"validation.json").read_bytes())
        dp = (donor/"patch.diff").read_bytes()
        dt = apply(base,dp)
        assert dt == components[str(version)] == dv["tree"] and sha(dp)==dv["patch_sha256"]
        component_patches[version] = sha(dp)
        if last is not None:
            dd = (donor/"delta.diff").read_bytes()
            assert apply(last,dd)==dt and sha(dd)==dv["delta_sha256"]
            component_deltas[version]=sha(dd)
        last=dt
    for p in changed_expected - {field}:
        assert git("show",tree+":"+p) == git("show",last+":"+p)
    actual=git("show",tree+":"+field)
    ranked=b"    muse_ranked_pool: bool = False\n"
    assert actual.count(ranked)==1 and actual.replace(ranked,b"")==git("show",last+":"+field)
    def batch_fields(t):
        node=next(n for n in ast.parse(git("show",t+":"+field)).body if getattr(n,"name",None)=="ForwardBatch")
        return [n.target.id for n in node.body if isinstance(n,ast.AnnAssign)]
    assert batch_fields(tree) == batch_fields(previous) + ["mixed_verify_is_causal","mixed_verify_prompt_count"]
    evidence = {}
    for name in ("flags","server_config","peer_capacity","paged_tail","full_attention","certificate",
                 "independent_causal_attention","attention_abi","ranked_model","transitions","work"):
        data = result(name)
        if "tree" in data:
            assert data["tree"] == tree
        if name in ("server_config","peer_capacity","paged_tail","certificate"):
            assert data["patch_sha256"] == sha(full)
        if name == "full_attention":
            assert data["candidates"]["V291"] == sha(full)
        hashes = data.get("hashes", data.get("source_sha256", data.get("sources", {})))
        if name == "independent_causal_attention":
            assert data["patch_sha256"]["V291"] == sha(full)
            hashes = hashes["V291"]
        for path,digest in hashes.items():
            path = path if path.startswith("python/sglang/") else "python/sglang/srt/"+path
            assert sha(git("show",tree+":"+path)) == digest, path
        evidence[name]=sha((here/(name+".json")).read_bytes())
    assert result("flags")["parsed_flag_orderings"] == 40320
    assert result("paged_tail")["total_tail_scalars"] == 409600
    assert result("full_attention")["sampled_attention_scalars"] == 23040
    print(json.dumps(dict(candidate="V291",parent="V290",base=base,parent_tree=previous,tree=tree,
        parent_sha256=sha(old),patch_sha256=sha(full),delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),
        paths=paths,changed_from_parent=changed,sources=sources,evidence_sha256=evidence,component_patches=component_patches,component_deltas=component_deltas,launch_flags=flags,
        max_added_line_chars=max(map(len,added)),byte_budget=limits["max_patch_bytes"],
        strict_apply_reverse_full_delta=True,official_export="okc serve diff, Git diff context 1",official_rules=rules_url,
        physical_H200=False,submitted=False),indent=2))
