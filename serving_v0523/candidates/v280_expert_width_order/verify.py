"""Apply both exports in independent Git indexes and check the public rules and linked receipts."""
import ast
import fnmatch
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here=Path(__file__).resolve().parent
parent=here.parent/"v274_four_block_prompt_multicast"
previous=json.loads((parent/"validation.json").read_bytes())
stage=Path(sys.argv[1]).resolve()
contract=json.loads(Path(sys.argv[2]).read_bytes())
base=contract["base"]["commit"]
rules=contract["rules"]
with tempfile.TemporaryDirectory() as tmp:
    obj=Path(tmp)/"objects"
    obj.mkdir()
    alternate=subprocess.check_output(["git","-C",stage,"rev-parse","--absolute-git-dir"],text=True).strip()+"/objects"
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/"index"),GIT_OBJECT_DIRECTORY=str(obj),GIT_ALTERNATE_OBJECT_DIRECTORIES=alternate)
    def git(*args,stdin=None):
        return subprocess.check_output(["git","-C",stage,*args],input=stdin,env=env)
    def apply(tree,patch,reverse=False):
        git("read-tree",tree)
        git("apply","--cached","--whitespace=error-all",*(["-R"] if reverse else []),"-",stdin=patch)
        return git("write-tree").decode().strip()
    delta=(here/"delta.diff").read_bytes()
    full=(here/"patch.diff").read_bytes()
    pp=(parent/"patch.diff").read_bytes()
    base248=(here.parent/"v248_shared_verify_raster/patch.diff").read_bytes()
    d270=(here.parent/"v270_wide_prompt_weight_reuse/delta.diff").read_bytes()
    prev=apply(apply(apply(base,base248),d270),(parent/"delta.diff").read_bytes())
    tree=apply(base,full)
    assert full==(stage.parent/"patch.diff").read_bytes() and prev=="7d806f2f44be5164ecabc78e449671713fbd912b" and sha256(pp).hexdigest()==previous["patch_sha256"]
    assert apply(prev,delta)==tree and apply(tree,delta,True)==prev and apply(tree,full,True)==git("rev-parse",base+"^{tree}").decode().strip()
    changed=git("diff","--name-only",prev,tree).decode().splitlines()
    paths=git("diff","--name-only",base,tree).decode().splitlines()
    assert len(changed)==3 and len(full)<=rules["max_patch_bytes"]
    for path in paths:
        assert path.startswith(rules["allowed_prefix"]) and not path.startswith(tuple(rules["forbidden_prefixes"])) and path.endswith(tuple(rules["allowed_suffixes"]))
        mode=git("ls-tree",tree,path).split()[0]
        original=git("ls-tree",base,path).split()
        assert mode==(original[0] if original else b"100644") and mode in (b"100644",b"100755")
        assert original or git("cat-file","-s",tree+":"+path).strip()!=b"0"
        if path.endswith(".py"): ast.parse(git("show",tree+":"+path))
    added=[s[1:] for s in full.decode().splitlines() if s.startswith("+") and not s.startswith("+++")]
    assert max(map(len,added))<=rules["max_line_chars"] and not any(w in s for s in added for w in rules["forbidden_words"])
    assert not any(b"\n"+w in full for w in (b"rename ",b"old mode ",b"new mode ",b"GIT binary patch",b"Binary files "))
    flags=previous["launch_flags"]
    denied=[g for gs in contract["launch_flags"]["denied"].values() for g in gs]
    assert not any(fnmatch.fnmatchcase(k,g) for k in flags for g in denied)
    sources={p:sha256(git("show",tree+":"+p)).hexdigest() for p in changed}
    def read(name):
        return json.loads((here/(name+".json")).read_bytes())
    assert read("constructor")["patch_sha256"]==sha256(full).hexdigest()
    assert read("width_replay")["production_header_sha256"] in sources.values()
    assert read("sm90a")["static_shared_bytes"]==1024
    print(json.dumps(dict(candidate="V280",parent="V248 + V270 and V274 W2 deltas",base=base,tree=tree,parent_tree=prev,v274_w2_export_sha256=sha256(pp).hexdigest(),
        patch_sha256=sha256(full).hexdigest(),delta_sha256=sha256(delta).hexdigest(),bytes=len(full),delta_bytes=len(delta),
        max_allowed_bytes=rules["max_patch_bytes"],added_line_maximum=max(map(len,added)),changed_from_parent=changed,
        sources=sources,paths=paths,launch_flags=flags,strict_apply_reverse_full_delta=True,official_export="okc serve diff, git diff.context=1",
        live_rules_sha256=sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),gpu=False,submitted=False),indent=2))
