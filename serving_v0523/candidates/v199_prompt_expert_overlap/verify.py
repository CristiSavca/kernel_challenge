"""Verify the frozen export, parent lineage, official static rules and compiled sources."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here, stage = Path(__file__).resolve().parent, Path(sys.argv[1])
parent = here.parent / "v198_segmented_verify_prompt_experts"
base = "297e54fd105e974cdf401e5b4d20253a22c16f90"
sha = lambda b: hashlib.sha256(b).hexdigest()
full, delta, p = [(d/f).read_bytes() for d,f in ((here,"patch.diff"),(here,"delta.diff"),(parent,"patch.diff"))]
pm = json.loads((parent/"validation.json").read_bytes())
assert sha(p) == pm["patch_sha256"]
with tempfile.TemporaryDirectory() as tmp:
    env = {**os.environ,"GIT_INDEX_FILE":str(Path(tmp)/"index")}

    def git(*args,stdin=None):
        return subprocess.check_output(["git","-C",stage,*args],env=env,input=stdin)

    def apply(tree,patch,reverse=False):
        git("read-tree",tree)
        git("apply","--cached","--whitespace=error-all",*(["-R"] if reverse else []),"-",stdin=patch)
        return git("write-tree").decode().strip()

    pt = apply(base,p)
    tree = apply(base,full)
    assert pt == pm["tree"] and apply(pt,delta) == tree and apply(tree,delta,True) == pt
    assert apply(tree,full,True) == git("rev-parse",base+"^{tree}").decode().strip()
    paths = git("diff","--name-only",base,tree).decode().splitlines()
    changed = git("diff","--name-only",pt,tree).decode().splitlines()
    assert len(changed) == 3
    config = json.loads(Path(sys.argv[2]).read_bytes()) if len(sys.argv)>2 else json.loads(
        subprocess.check_output(["curl","-fsSL","https://openkernelchallenge.dev/api/v1/serving"]))
    rules = config["rules"]
    assert base == config["base"]["commit"] and len(full) <= rules["max_patch_bytes"]
    for path in paths:
        assert path.startswith(rules["allowed_prefix"]) and not path.startswith(tuple(rules["forbidden_prefixes"]))
        assert path.endswith(tuple(rules["allowed_suffixes"]))
        data=git("show",tree+":"+path)
        assert len(data)
        if path.endswith(".py"): compile(data,path,"exec")
    assert not any(line.startswith((b"rename ",b"old mode ",b"new mode ",b"GIT binary patch",b"Binary files ")) for line in full.splitlines())
    added=[line[1:] for line in full.decode().splitlines() if line.startswith("+") and not line.startswith("+++")]
    assert max(map(len,added))<=rules["max_line_chars"] and not any(w in x for x in added for w in rules["forbidden_words"])
    denied=[p for ps in config["launch_flags"]["denied"].values() for p in ps]
    flags=pm["launch_flags"]
    assert not any(fnmatch.fnmatchcase(k,v) for k in flags for v in denied)
    artifacts={path:sha(git("show",tree+":"+path)) for path in changed}
    sm=json.loads((here/"sm90a.json").read_bytes())
    dig=lambda leaf: next(d for path,d in artifacts.items() if path.endswith("/"+leaf))
    assert dig("prompt_weight_pipeline.cuh")==json.loads((here/"operands.json").read_text())["header_sha256"]
    assert dig("prompt_weight_pipeline.py")==json.loads((here/"startup.json").read_text())["adapter_sha256"]
    cb=json.loads((here/"independent_callback.json").read_text())
    assert sha(full)==cb["official_patch_sha256"] and dig("muse_spark_humming_moe.py")==cb["sha256"]["muse_spark_humming_moe.py"]
    assert sm["images"]["pipeline"]["register_audit"]["group_sizes"]==[4]*8
    print(json.dumps(dict(candidate="V199",base=base,parent="V198",parent_tree=pt,parent_sha256=sha(p),tree=tree,
        official_export="okc serve diff .",patch_sha256=sha(full),bytes=len(full),delta_sha256=sha(delta),delta_bytes=len(delta),
        paths=paths,changed_from_parent=artifacts,launch_flags=flags,checked_flag_patterns=len(denied),
        max_added_line_chars=max(map(len,added)),strict_apply_reverse_full_delta=True,
        compiled_runtime="NVCC with pre-load Hopper instruction check",humming="0.1.12",physical_H200=False,submitted=False),indent=2))
