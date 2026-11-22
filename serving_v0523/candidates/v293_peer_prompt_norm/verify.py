"""Validate either peer-normalization export and match code and evidence to its public tree."""
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent if len(sys.argv)<4 else Path(sys.argv[3]).resolve()
combined = here.name.startswith("v296_")
name, prior = ("V296","V295") if combined else ("V293","V291")
parent = next(here.parent.glob(prior.lower()+"_*"))
stage = Path(sys.argv[1]).resolve()
rules_url = "https://openkernelchallenge.dev/api/v1/serving"
contract = json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv) > 2 else
                      subprocess.check_output(["curl", "-fsSL", rules_url]))
limits = contract["rules"]
base = contract["base"]["commit"]
before = json.loads((parent/"validation.json").read_bytes())
peer = "python/sglang/srt/layers/muse_spark/peer_prompt_pack.py"
humming=peer.replace("peer_prompt_pack.py","muse_spark_humming_moe.py")
changed_expected={peer,humming,"python/sglang/srt/models/muse_spark.py"}

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
    only = here.parent/"v293_peer_prompt_norm"
    numeric=json.loads((only/"numeric.json").read_bytes())
    compiled=json.loads((here.parent/"v284_rank_major_routed_pool/independent_native_sm90a.json").read_bytes())
    cuda="python/sglang/kernels/jit/csrc/muse_spark/ranked_pool_norm.cuh"
    assert sha(git("show",tree+":"+cuda))==numeric["unchanged_cuda_sha256"]==compiled["files"]["ranked_pool_norm.cuh"]
    assert numeric["words"]==21738788 and numeric["mismatches"]==0
    def definition(ver,path,name):
        nodes=ast.parse(git("show",ver+":"+path)).body
        return ast.dump(next(n for n in nodes if getattr(n,"name",None)==name),include_attributes=False)
    assert definition(tree,humming,"_ranked_pool_module")==definition(previous,humming,"_ranked_pool_module")
    setup=result("native_setup")
    assert setup["source"]==sources[peer] and sum(c["normalization"] for c in setup["cases"])==3
    if combined:
        basic=result("norm_publication")
        second=result("independent_norm_routes")
        graph=result("graph_norm")
        assert graph["tree"]==tree
        for hashes in (second["sources"],graph["source_sha256"]):
            for p,digest in hashes.items():
                assert sha(git("show",tree+":"+p))==digest,p
        assert basic["counts"]["native_norms"]+basic["counts"]["into_slab"]==276
        assert basic["counts"]["veto_slab_clones"]>0 and basic["mismatches"]==0
        wrong=result("independent_norm_alias")
        assert wrong["exit_code"] and "17088" in wrong["detected"]
        late=result("independent_norm_late_helper")
        decode,parsed,config=(result(n) for n in ("ordinary_decode","flags","server_config"))
        for hashes in (late["sources"],decode["hashes"],config["source_sha256"]):
            for p,digest in hashes.items():
                p=p if p.startswith("python/sglang/") else "python/sglang/srt/"+p
                assert sha(git("show",tree+":"+p))==digest,p
        assert late["counts"]["calls"]==12 and decode["compared_calls"]==34
        assert parsed["tree"]==config["tree"]==tree and parsed["parsed_flag_orderings"]==40320
        assert parsed["launch_flags"]==flags and sha(full)==config["patch_sha256"] and len(config["cases"])==17
        receipts=("norm_publication","independent_norm_routes","independent_norm_alias","graph_norm",
                  "independent_norm_late_helper","native_setup","ordinary_decode","flags","server_config")
    else:
        basic,admission=result("transport_norm"),result("admission")
        for p,digest in admission["sources"].items():
            assert sha(git("show",tree+":"+p))==digest,p
        assert basic["words_compared"]==6307840 and basic["mismatches"]==0
        assert admission["accepted"]==["frozen","plain"] and len(admission["declined"])>=24
        receipts=("transport_norm","admission","numeric","work","native_setup")
    evidence={n:sha((here/(n+".json")).read_bytes()) for n in receipts}
    print(json.dumps(dict(candidate=name,parent=prior,base=base,parent_tree=previous,tree=tree,
        parent_sha256=sha(old),patch_sha256=sha(full),delta_sha256=sha(delta),bytes=len(full),delta_bytes=len(delta),
        paths=paths,changed_from_parent=changed,sources=sources,evidence_sha256=evidence,unchanged_cuda=numeric["unchanged_cuda_sha256"],launch_flags=flags,
        max_added_line_chars=max(map(len,added)),byte_budget=limits["max_patch_bytes"],
        strict_apply_reverse_full_delta=True,official_export="okc serve diff, Git diff context 1",official_rules=rules_url,
        physical_H200=False,submitted=False),indent=2))
