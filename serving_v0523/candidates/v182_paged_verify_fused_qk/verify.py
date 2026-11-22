"""Check the full V180/V136 integration against pinned sources and official rules."""
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

base = "297e54fd105e974cdf401e5b4d20253a22c16f90"
here = Path(__file__).resolve().parent
stage = Path(sys.argv[1])
donor = here.parent / "v136_fused_verify_qk_rope_cache"
parent = here.parent / "v180_paged_verify_direct_prompt"
m, v = [json.loads((folder/"validation.json").read_bytes()) for folder in [parent, donor]]
full, delta = [(here / name).read_bytes() for name in ("patch.diff","delta.diff")]
dpatch = (donor/"delta.diff").read_bytes()
sha = lambda data: hashlib.sha256(data).hexdigest()
url = "https://openkernelchallenge.dev/api/v1/serving"
published = json.loads(Path(sys.argv[2]).read_bytes() if len(sys.argv)>2 else subprocess.check_output(["curl","-fsSL",url]))
assert published["base"]["commit"] == base
rules = published["rules"]


def line_edits(data):
    return [line for line in data.splitlines() if line.startswith((b"+",b"-")) and not line.startswith((b"+++",b"---"))]


with tempfile.TemporaryDirectory() as scratch:
    env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch)/"index"))

    def git(*args, data=None):
        return subprocess.check_output(["git","-C",stage,*args],env=env,input=data)

    def apply(tree, patch, reverse=False):
        git("read-tree",tree)
        git("apply","--cached","--whitespace=error-all",*(["--reverse"] if reverse else []),"-",data=patch)
        return git("write-tree").decode().strip()

    v68_patch = next(here.parent.glob("v68_*/patch.diff")).read_bytes()
    assert sha(v68_patch) == m["parent_sha256"] and apply(base,v68_patch) == m["parent_tree"]
    for candidate, report in [(donor,v),(parent,m)]:
        original = (candidate/"patch.diff").read_bytes()
        assert sha(original) == report["patch_sha256"] and apply(base,original) == report["tree"]
    tree = apply(base,full)
    assert apply(m["tree"],delta) == apply(m["tree"],dpatch) == tree
    assert apply(tree,delta,True) == apply(tree,dpatch,True) == m["tree"]
    assert apply(tree,full,True) == git("rev-parse",base+"^{tree}").decode().strip()
    assert line_edits(delta) == line_edits(dpatch)
    paths = git("diff","--name-only",base,tree).decode().splitlines()
    changed = git("diff","--name-only",m["tree"],tree).decode().splitlines()
    assert len(changed) == 4 and set(changed) == set(v["changed_from_parent"])
    for p in paths:
        assert p.startswith(rules["allowed_prefix"]) and not p.startswith(tuple(rules["forbidden_prefixes"]))
        assert p.endswith(tuple(rules["allowed_suffixes"]))
        prior_mode = git("ls-tree",base,p).split()
        assert git("ls-tree",tree,p).startswith((prior_mode[0] if prior_mode else b"100644")+b" blob ")
        data = git("show",tree+":"+p)
        assert data == (stage/p).read_bytes()
        if p.endswith(".py"):
            compile(data,p,"exec")

    back = "python/sglang/srt/layers/attention/flashattention_backend.py"
    model = "python/sglang/srt/models/muse_spark.py"
    kernel = "python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py"
    identical = {}
    def method(t,path,cls,name):
        module = ast.parse(git("show",t+":"+path))
        body = next(node.body for node in module.body if getattr(node,"name",None)==cls)
        item = next(node for node in body if getattr(node,"name",None)==name)
        return ast.dump(item)
    for path, cls, key, origin in [(back,"FlashAttentionBackend","forward_extend",m),
            (back,"FlashAttentionBackend","prepare_paged_mha_query",m),
            (back,"FlashAttentionBackend","muse_verify_fp8_store",v),
            (model,"MuseSpark1xAttention","forward",v)]:
        body = method(tree,path,cls,key)
        assert body == method(origin["tree"],path,cls,key)
        identical[key] = dict(donor=origin["candidate"],ast_sha256=sha(body.encode()))
    assert sha(git("show",tree+":"+kernel)) == v["kernel_sha256"]
    assert git("show",tree+":python/sglang/srt/layers/attention/direct_prompt_kv.py") == git(
        "show",m["tree"]+":python/sglang/srt/layers/attention/direct_prompt_kv.py")
    assert m["launch_flags"] == v["flags"]
    forbidden_headers=(b"rename ",b"old mode ",b"new mode ",b"GIT binary patch",b"Binary files ")
    assert not any(line.startswith(forbidden_headers) for line in full.splitlines())
    added=[line[1:] for line in full.decode().splitlines() if line.startswith("+") and not line.startswith("+++")]
    longest=max(map(len,added))
    assert len(full)<=rules["max_patch_bytes"] and longest<=rules["max_line_chars"]
    assert not any(word in line for word in rules["forbidden_words"] for line in added)
    denied=[word for words in published["launch_flags"]["denied"].values() for word in words]
    assert not any(fnmatch.fnmatchcase(k,w) for k in m["launch_flags"] for w in denied)
    print(json.dumps(dict(candidate="V182", base=base, parent="V180", parent_sha256=m["patch_sha256"],parent_tree=m["tree"],
        donor="V136",donor_tree=v["tree"],donor_sha256=v["patch_sha256"],tree=tree,patch_sha256=sha(full),bytes=len(full),
        delta_sha256=sha(delta),delta_bytes=len(delta),donor_delta_sha256=sha(dpatch),identical_donor_source_edits=len(line_edits(delta)),
        source_edit_digest=sha(b"\n".join(line_edits(delta))),paths=paths,changed_from_parent=changed,identical_methods=identical,
        launch_flags=m["launch_flags"], kernel_sha256=v["kernel_sha256"],bound_sm90_specializations=v["release_sm90_assemblies"],
        bound_interpreter_configurations=v["interpreter_cases"],bound_interpreter_compared_bytes=v["interpreter_bytes"],
        measured_ancestor="V68",measured_comparison_job=m["measured_parent_job"],measured_comparison_users=m["parent_users"],
        measured_comparison_tok_s_per_gpu=m["parent_tok_s_per_gpu"],official_rules=url,official_export="okc serve diff .",
        max_added_line_chars=longest,checked_flag_patterns=len(denied),strict_apply_reverse_full_delta=True,h200=False,submitted=False),indent=2))
