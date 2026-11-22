"""Reconstruct the comparison trees and verify the integrated expert device sources."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()
CANDIDATES=HERE.parent


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    controls={n:next(CANDIDATES.glob(f"v{n}_*")) for n in (258,261,263,267)}
    manifests={n:json.loads((folder/"validation.json").read_bytes()) for n,folder in controls.items()}
    sources={}
    with tempfile.TemporaryDirectory() as temp:
        objects=Path(temp)/"objects"
        objects.mkdir()
        native=subprocess.check_output(["git","-C",STAGE,"rev-parse","--git-path","objects"],text=True).strip()
        git_env={**os.environ,"GIT_INDEX_FILE":str(Path(temp)/"index"),"GIT_OBJECT_DIRECTORY":str(objects),
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES":str((STAGE/native).resolve())}
        def git(*args, data=None):
            return subprocess.check_output(["git","-C",STAGE,*args],input=data,env=git_env)
        for n,folder in controls.items():
            manifest=manifests[n]; patch=(folder/"patch.diff").read_bytes()
            assert digest(patch)==manifest["patch_sha256"]
            git("read-tree",manifest["base"])
            git("apply","--cached","--whitespace=error-all","-",data=patch)
            assert git("write-tree").decode().strip()==manifest["tree"]
        def source(n,path):
            return git("show",manifests[n]["tree"]+":"+path)
        cpu="python/sglang/srt/layers/muse_spark/"
        gpu="python/sglang/kernels/jit/csrc/muse_spark/"
        for n,path in ((258,"python/sglang/kernels/ops/moe/muse_spark_peer_pack.py"),(258,cpu+"peer_prompt_pack.py"),
                (263,gpu+"decode_gate_up.cuh"),(263,gpu+"decode_gate_up_launch.cuh"),(263,gpu+"live_decode_reduce.cuh"),
                (263,gpu+"residual_prompt_mma.cuh"),(263,gpu+"grouped_decode_mma_pair.cuh"),(263,cpu+"decode_gate_up.py"),
                (263,cpu+"live_decode_reduce.py"),(263,cpu+"decode_mma_pair.py"),(263,cpu+"residual_prompt_mma.py")):
            control=source(n,path)
            assert source(267,path)==control and (STAGE/path).read_bytes()==control,path
            sources[path]={"control":n,"sha256":digest(control)}
        path=cpu+"muse_spark_humming_moe.py"
        def body(n,name):
            return ast.dump(next(node for node in ast.parse(source(n,path)).body if getattr(node,"name","")==name))
        functions={}
        for n,names in ((258,("humming_grouped_expert_outputs","prompt_route_order","_pack_prompt_routes")),
                        (261,("_decode_route_expert","_plan_grouped_decode","_stage_grouped_decode","_gather_grouped_decode")),
                        (263,("humming_grouped_decode_outputs","humming_decode_eligible","prepare_humming_nvfp4"))):
            for name in names:
                old=body(n,name)
                assert body(267,name)==old,name
                functions[name]={"control":n,"ast_sha256":digest(old.encode())}
        return dict(patch_sha256=manifests[267]["patch_sha256"],tree=manifests[267]["tree"],
                    byte_identical=sources,identical_python_functions=functions,
                    qualified_hardware_runs=[],unchanged_SM90_code_coverage={"V258_stable_prompt":22,
                    "V261_fused_decode":22,"V263_token_major_code_and_controls":7})
if __name__=="__main__":
    print(json.dumps(main(),indent=2))
