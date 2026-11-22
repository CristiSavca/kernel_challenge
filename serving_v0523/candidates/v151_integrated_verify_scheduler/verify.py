"""Reconstruct V151 and check method identity across its two source lineages."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from fnmatch import fnmatchcase
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE="297e54fd105e974cdf401e5b4d20253a22c16f90"
PARENT="cd089ab0b54e9fa2ee4c1cfdcb3f354370748d85"


def sha(b):return hashlib.sha256(b).hexdigest()


def verify(stage):
    patch=(HERE/"patch.diff").read_bytes()
    delta=(HERE/"delta.diff").read_bytes()
    parent_dir=HERE.parent/"v147_integrated_experts_prefill"
    parent=(parent_dir/"patch.diff").read_bytes()
    score=json.loads((HERE.parent.parent/"runs/abd217baf0ed49fd.json").read_text())
    assert sha(parent)==json.loads((parent_dir/"validation.json").read_text())["patch_sha256"]
    flags=score["serving_flags"]|{"--speculative-dflash-share-verify-schedule":""}
    fixed=(HERE.parent.parent/"flags").read_text().split("# Fixed,")[1].split("\n",1)[1]
    patterns=[v.strip() for v in fixed.replace("#","").replace("\n","").split(",") if v.strip()]
    assert not any(fnmatchcase(flag,pattern) for flag in flags for pattern in patterns)
    with tempfile.TemporaryDirectory() as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/"index"))
        def git(*args,content=None):
            return subprocess.check_output(["git","-C",stage,*args],env=env,input=content)
        def apply(base,payload,reverse=False):
            git("read-tree",base)
            git("apply","--cached","--whitespace=error",*(["-R"] if reverse else []),"-",content=payload)
            return git("write-tree").decode().strip()
        baseline=git("rev-parse",BASE+"^{tree}").decode().strip()
        target=apply(BASE,patch)
        assert apply(target,patch,True)==baseline and apply(BASE,parent)==PARENT
        assert apply(PARENT,delta)==target and apply(target,delta,True)==PARENT
        changed=git("diff","--name-only",PARENT,target).decode().splitlines()
        paths=git("diff","--name-only",BASE,target).decode().splitlines()
        sources={}
        for name in paths:
            assert name.startswith("python/sglang/") and "/model_loader/" not in name and name.endswith((".py",".json",".cuh"))
            data=git("show",target+":"+name)
            assert data==(stage/name).read_bytes()
            sources[name]=sha(data)
            if name.endswith(".py"):compile(data,name,"exec")
        import ast
        be="python/sglang/srt/layers/attention/flashattention_backend.py"
        mo="python/sglang/srt/models/muse_spark.py"
        def method(tree,filename,cls,name):
            module=ast.parse(git("show",tree+":"+filename))
            cl=next(n for n in module.body if getattr(n,"name","")==cls)
            return ast.dump(next(n for n in cl.body if getattr(n,"name","")==name))
        other=(HERE.parent/"v132_verify_scheduler_metadata/patch.diff").read_bytes()
        old=apply(BASE,other)
        assert method(old,be,"FlashAttentionBackend","_verify_schedule")==method(target,be,"FlashAttentionBackend","_verify_schedule")
        for filename,cls,name in ((be,"FlashAttentionBackend","muse_verify_fp8_store"),(be,"FlashAttentionBackend","get_paged_mha_kv_cache"),
                (be,"FlashAttentionBackend","prepare_paged_mha_query"),(mo,"MuseSpark1xAttention","forward")):
            assert method(PARENT,filename,cls,name)==method(target,filename,cls,name)
    added=max(len(line)-1 for line in patch.splitlines() if line.startswith(b"+") and not line.startswith(b"+++"))
    assert len(paths)==19 and len(changed)==4 and added<=400 and len(patch)<400000
    return dict(candidate="V151",base=BASE,parent="V147",parent_tree=PARENT,parent_sha256=sha(parent),
                patch_sha256=sha(patch),delta_sha256=sha(delta),tree=target,bytes=len(patch),delta_bytes=len(delta),
                paths=paths,changed_from_parent=changed,source_sha256=sources,launch_flags=flags,fixed_flag_patterns=len(patterns),
                maximum_added_line_bytes=added,python_compiled=sum(name.endswith(".py") for name in paths),strict_forward_reverse=True,
                scheduler_method_identical_V132=True,V147_normalization_forward_and_KV_writer_selection_identical=True,
                official_export="okc serve diff .",physical_H200=False,judge_submitted=False)


if __name__=="__main__":
    print(json.dumps(verify(Path(sys.argv[1]).resolve()),indent=2))
