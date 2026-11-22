"""Reconstruct both frozen donor orders and validate the official V195 submission-format patch."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1]).resolve()
dirs={v:next(here.parent.glob(f"v{v}_*")) for v in (190,191,192)}
man={v:json.loads((p/"validation.json").read_bytes()) for v,p in dirs.items()}
full={v:(p/"patch.diff").read_bytes() for v,p in dirs.items()}
delta={v:(p/"delta.diff").read_bytes() for v,p in dirs.items() if v!=190}
full[195]=(here/"patch.diff").read_bytes();delta[195]=(here/"delta.diff").read_bytes()
sha=lambda b:hashlib.sha256(b).hexdigest()
for v,p in dirs.items():
    assert sha(full[v])==man[v]["patch_sha256"]
    if v in delta:assert sha(delta[v])==man[v]["delta_sha256"]
base=man[190]["base"]
with tempfile.TemporaryDirectory() as root:
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(root)/"index"))
    def git(*args,data=None):
        return subprocess.check_output(["git","-C",stage,*args],env=env,input=data)
    def apply(tree,direction,blob):
        git("read-tree",tree)
        git("apply","--cached","--whitespace=error-all",*(["-R"] if direction<0 else []),"-",data=blob)
        return git("write-tree").decode().strip()
    tree={v:apply(base,1,b) for v,b in full.items()}
    for v in dirs:assert tree[v]==man[v]["tree"]
    root_tree=git("rev-parse",base+"^{tree}").decode().strip()
    for v in (191,192):
        assert apply(tree[190],1,delta[v])==tree[v] and apply(tree[v],-1,delta[v])==tree[190]
        peer=383-v
        assert apply(tree[peer],1,delta[v])==tree[195] and apply(tree[195],-1,delta[v])==tree[peer]
    assert apply(tree[191],1,delta[195])==tree[195] and apply(tree[195],-1,delta[195])==tree[191]
    assert all(apply(tree[v],-1,b)==root_tree for v,b in full.items())
    phases={}
    for v in (191,192):
        changes=git("diff","--name-only",tree[190],tree[v]).decode().splitlines()
        phases[v]={p:sha(git("show",tree[195]+":"+p)) for p in changes}
        for p,h in phases[v].items():assert h==sha(git("show",tree[v]+":"+p))==sha((stage/p).read_bytes())
    assert set(phases[191]).isdisjoint(phases[192]) and [len(phases[v]) for v in (191,192)]==[5,3]
    changed=git("diff","--name-only",tree[190],tree[195]).decode().splitlines()
    assert set(changed)==set(phases[191])|set(phases[192])
    cfg=json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv)>2 else json.loads(subprocess.check_output(
        ["curl","-fsSL","https://openkernelchallenge.dev/api/v1/serving"]))
    rules=cfg["rules"];flags=man[190]["launch_flags"]
    assert cfg["base"]["commit"]==base and all(m["launch_flags"]==flags for m in man.values())
    denied=[mask for values in cfg["launch_flags"]["denied"].values() for mask in values]
    assert not any(fnmatch.fnmatchcase(k,p) for k in flags for p in denied)
    paths=git("diff","--name-only",base,tree[195]).decode().splitlines()
    for p in paths:
        assert p.startswith(rules["allowed_prefix"]) and not p.startswith(tuple(rules["forbidden_prefixes"]))
        assert p.endswith(tuple(rules["allowed_suffixes"]))
        data=git("show",tree[195]+":"+p); assert data==(stage/p).read_bytes()
        if p.endswith(".py"):compile(data,p,"exec")
    export=full[195]
    additions=[s[1:] for s in export.decode().splitlines() if s.startswith("+") and not s.startswith("+++")]
    assert not any(line.startswith((b"rename ",b"old mode ",b"new mode ",b"GIT binary patch",b"Binary files ")) for line in export.splitlines())
    assert len(export)<=rules["max_patch_bytes"] and max(map(len,additions))<=rules["max_line_chars"]
    assert not any(w in line for w in rules["forbidden_words"] for line in additions)
    print(json.dumps(dict(candidate="V195",base=base,parent="V191",parent_tree=tree[191],parent_sha256=sha(full[191]),
        common_parent="V190",common_sha256=sha(full[190]),attention_donor="V192",attention_sha256=sha(full[192]),
        attention_donor_delta_sha256=sha(delta[192]),tree=tree[195],official_export="okc serve diff .",
        patch_sha256=sha(export),bytes=len(export),delta_bytes=len(delta[195]),delta_sha256=sha(delta[195]),
        paths=paths,changed_from_parent=list(phases[192]),independent_phase_sources=phases,launch_flags=flags,
        checked_denied_flag_patterns=len(denied),max_added_line_chars=max(map(len,additions)),
        strict_both_orders_all_reverse=True,compiled_every_changed_python_module=True,physical_H200=False,submitted=False),indent=2))
