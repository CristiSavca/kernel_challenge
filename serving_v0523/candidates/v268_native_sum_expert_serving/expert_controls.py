"""Verify that V268's complete expert kernels and public dispatch bodies match the V267 controls."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
V267 = HERE.parent / "v267_pipelined_expert_serving"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((HERE/"validation.json").read_bytes())
    previous = json.loads((V267/"validation.json").read_bytes())
    receipts = json.loads((V267/"expert_sources.json").read_bytes())
    assert receipts["tree"]==previous["tree"] and previous["patch_sha256"]==receipts["patch_sha256"]
    fixed = list(receipts["byte_identical"]) + ["python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py",
                                               "python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py"]
    model = "python/sglang/srt/models/muse_spark.py"
    def methods(text):
        cls=next(node for node in ast.parse(text).body if getattr(node,"name","")=="MuseSpark1xMoE")
        return {name:ast.dump(next(node for node in cls.body if getattr(node,"name","")==name))
                for name in ["forward","_forward_sp_experts","_forward_sp_replicated_experts"]}
    with tempfile.TemporaryDirectory() as tmp:
        env={**os.environ,"GIT_INDEX_FILE":str(Path(tmp)/"index")}
        def git(*args,input=None):
            return subprocess.check_output(["git","-C",STAGE,*args],input=input,env=env)
        for folder,receipt in [(HERE,manifest),(V267,previous)]:
            full=(folder/"patch.diff").read_bytes()
            assert sha(full)==receipt["patch_sha256"]
            git("read-tree",receipt["base"])
            git("apply","--cached","--whitespace=error-all","-",input=full)
            assert git("write-tree").decode().strip()==receipt["tree"]
        files={}
        for path in fixed:
            a=git("show",manifest["tree"]+":"+path)
            b=git("show",previous["tree"]+":"+path)
            assert a==b and a==(STAGE/path).read_bytes(),path
            files[path]=sha(a)
        pub=methods(git("show",manifest["tree"]+":"+model))
        assert pub==methods(git("show",previous["tree"]+":"+model))
        return dict(full_sha256=manifest["patch_sha256"],control_sha256=previous["patch_sha256"],
                    byte_identical_files=files,model_method_AST_sha256={k:sha(v.encode()) for k,v in pub.items()},
                    V267_sources_sha256=sha((V267/"expert_sources.json").read_bytes()),physical_H200=False)
if __name__=="__main__":
    print(json.dumps(main(),indent=2))
