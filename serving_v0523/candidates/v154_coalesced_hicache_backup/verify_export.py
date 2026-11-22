"""Reconstruct both official patches and prove that they share one transport change."""
from pathlib import Path
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tempfile


HERE=Path(__file__).resolve().parent
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
HEADER="python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(stage,candidate,parent,rules,flags):
    full=(candidate/"patch.diff").read_bytes()
    parent_patch=(parent/"patch.diff").read_bytes()
    delta=(HERE/"delta.diff").read_bytes()
    with tempfile.TemporaryDirectory() as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/"index"))
        def git(*args,data=None):
            return subprocess.check_output(["git","-C",stage,*args],env=env,input=data)
        def apply(tree,patch,reverse=False):
            git("read-tree",tree)
            git("apply","--cached","--whitespace=error",*(["-R"] if reverse else []),"-",data=patch)
            return git("write-tree").decode().strip()
        base_tree=git("rev-parse",PIN+"^{tree}").decode().strip()
        parent_tree=apply(PIN,parent_patch)
        tree=apply(PIN,full)
        assert apply(parent_tree,delta)==tree
        assert apply(tree,full,True)==base_tree and apply(tree,delta,True)==parent_tree
        assert git("diff","--name-only",parent_tree,tree).decode().strip()==HEADER
        assert git("diff","--no-ext-diff",parent_tree,tree)==delta
        assert subprocess.check_output(["git","-C",stage,"write-tree"]).decode().strip()==tree
        paths=[]
        for record in git("diff","--no-renames","--name-status",PIN,tree).decode().splitlines():
            status,path=record.split("\t")
            assert status in ("M","A") and path.startswith(rules["allowed_prefix"])
            assert not any(path.startswith(p) for p in rules["forbidden_prefixes"])
            assert Path(path).suffix in rules["allowed_suffixes"]
            contents=git("show",tree+":"+path)
            assert contents and contents==(stage/path).read_bytes()
            if path.endswith(".py"):
                compile(contents,path,"exec")
            paths.append(path)
    longest=0
    for line in full.decode().splitlines():
        if line.startswith("+") and not line.startswith("+++ "):
            assert not any(word in line for word in rules["forbidden_words"])
            longest=max(longest,len(line)-1)
        assert not line.startswith(("old mode ","new mode ","GIT binary patch","Binary files ","rename from ","rename to "))
        if line.startswith("new file mode "):assert line=="new file mode 100644"
    assert longest<=rules["max_line_chars"] and len(full)<=rules["max_patch_bytes"]
    source=sha((stage/HEADER).read_bytes())
    cpu=json.loads((HERE/"native_cpu.json").read_text())
    sm90=json.loads((HERE/"sm90.json").read_text())
    assert source==sm90["source_sha256"]==cpu["source_sha256"]
    return dict(candidate=candidate.name.split("_")[0].replace("v","V",1),parent=parent.name.split("_")[0].upper(),
                base=PIN,tree=tree,parent_tree=parent_tree,bytes=len(full),patch_sha256=sha(full),
                parent_sha256=sha(parent_patch),delta_sha256=sha(delta),delta_bytes=len(delta),
                changed_from_parent=[HEADER],header_sha256=source,paths=paths,launch_flags=flags,
                longest_added_line_chars=longest,strict_apply_full_delta_and_reverse=True,
                export="okc serve diff .",physical_h200=False)


if __name__=="__main__":
    live=json.loads(subprocess.check_output(["curl","-fsSL","https://openkernelchallenge.dev/api/v1/serving"]))
    assert live["base"]["commit"]==PIN
    denials=[f for group in live["launch_flags"]["denied"].values() for f in group]
    flags151=json.loads((HERE.parent/"v151_integrated_verify_scheduler/validation.json").read_text())["launch_flags"]
    flags64=json.loads((HERE.parent/"v147_integrated_experts_prefill/validation.json").read_text())["launch_flags"]
    for idx, (candidate,parent,flags) in enumerate(((HERE,HERE.parent/"v151_integrated_verify_scheduler",flags151),
            (HERE.parent/"v154c_v64_hicache_backup",HERE.parent/"v64_humming_grouped_sequence_parallel",flags64))):
        data=verify(Path(sys.argv[idx+1]),candidate,parent,live["rules"],flags)
        assert not any(fnmatch.fnmatchcase(f,p) for f in flags for p in denials)
        data.update(fixed_flag_patterns=len(denials),official_rules="https://openkernelchallenge.dev/api/v1/serving")
        (candidate/"validation.json").write_text(json.dumps(data,indent=2)+"\n")
        print(data["candidate"],data["patch_sha256"],data["tree"],len(data["paths"]),data["bytes"])
