"""Reconstruct both trees, validate the public source rules and count per-rank route resources."""
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


here=Path(__file__).resolve().parent
parent=next(here.parent.glob("v309_*"))
sglang=here.parents[1]/"sglang"
contract=json.loads((parent/"public_contract.json").read_bytes())
rules=contract["rules"]
prior=json.loads((parent/"validation.json").read_bytes())
base=contract["base"]["commit"]
assert base==prior["base"]
sha=lambda data:hashlib.sha256(data).hexdigest()
old=(parent/"patch.diff").read_bytes()
full=(here/"patch.diff").read_bytes()
delta=(here/"delta.diff").read_bytes()
assert sha(old)==prior["patch_sha256"]
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    objects=tmp/"objects"
    objects.mkdir()
    source=subprocess.check_output(["git","-C",sglang,"rev-parse","--absolute-git-dir"],text=True).strip()
    env=dict(os.environ,GIT_INDEX_FILE=str(tmp/"index"),GIT_OBJECT_DIRECTORY=str(objects),
             GIT_ALTERNATE_OBJECT_DIRECTORIES=source+"/objects")
    def git(*args,data=None):
        return subprocess.check_output(["git","-C",sglang,*args],env=env,input=data)
    def apply(tree,patch,reverse=False):
        git("read-tree",tree)
        git("apply","--cached","--whitespace=error-all",*(["-R"] if reverse else []),"-",data=patch)
        return git("write-tree").decode().strip()
    before=apply(base,old)
    tree=apply(base,full)
    assert before==prior["tree"] and apply(before,delta)==tree and apply(tree,delta,True)==before
    assert apply(tree,full,True)==git("rev-parse",base+"^{tree}").decode().strip()
    assert git("diff","--no-renames","-U1",base,tree)==full
    changes=git("diff","--name-only",before,tree).decode().splitlines()
    assert len(changes)==2
    for path in changes:
        assert path.rsplit("/",1)[-1] in {"humming_prompt_route_tiles.py","humming_prefill_waves.py"}
    paths=git("diff","--name-only",base,tree).decode().splitlines()
    for path in paths:
        assert path.startswith(rules["allowed_prefix"]) and not path.startswith(tuple(rules["forbidden_prefixes"]))
        assert path.endswith(tuple(rules["allowed_suffixes"]))
        mode=git("ls-tree",tree,path).split()[0]
        old_mode=git("ls-tree",base,path).split()[:1] or [b"100644"]
        assert [mode]==old_mode and mode in {b"100644",b"100755"}
        source=git("show",tree+":"+path)
        assert b"\0" not in source
        if path.endswith(".py"):
            ast.parse(source)
    lines=full.decode().splitlines()
    added=[line[1:] for line in lines if line.startswith("+") and not line.startswith("+++")]
    longest=max(map(len,added))
    assert longest<=rules["max_line_chars"] and len(full)<=rules["max_patch_bytes"]
    assert not any(word in line for line in added for word in rules["forbidden_words"])
    assert not any(line.startswith(("rename ","old mode ","new mode ","GIT binary patch","Binary files")) for line in lines)
    flags=json.loads((here/"flags.json").read_bytes())
    assert flags==prior["launch_flags"]
    denied=[pattern for patterns in contract["launch_flags"]["denied"].values() for pattern in patterns]
    assert not any(fnmatch.fnmatchcase(flag,pat) for flag in flags for pat in denied)
data=json.loads((parent/"geometry.json").read_bytes())["examples_TP8_BF16"]["16384"]
slots=16384*8
experts=32
waves=data["V309_layouts"]["fused"]["max_waves"]
tiles=(slots+255)//256
old_bytes=slots*(4+8)
new_bytes=4*(slots+experts*(tiles+1)+experts*waves)
print(json.dumps(dict(base=base,parent="V309",parent_tree=before,tree=tree,patch_sha256=sha(full),delta_sha256=sha(delta),
    bytes=len(full),files=len(paths),max_added_line_chars=longest,strict_apply_reverse_full_delta=True,launch_flags=flags,
    common_16k=dict(slots=slots,tile_size=256,tiles=tiles,experts=experts,waves=waves,prior_sorted_ids_and_order_bytes=old_bytes,
        candidate_keys_prefix_and_boundaries_bytes=new_bytes,difference_bytes=old_bytes-new_bytes),
    physical_H200=False,submitted=False),indent=2))
