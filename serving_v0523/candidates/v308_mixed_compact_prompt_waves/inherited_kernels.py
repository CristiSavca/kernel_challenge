"""Bind the V308 device source and its public wrapper registrations to independently checked parents."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
stage = Path(sys.argv[1]).resolve()
v204 = next(here.parent.glob("v204_*"))
v306 = next(here.parent.glob("v306_*"))
pin = json.loads((v204/"validation.json").read_bytes())["base"]
jit = "python/sglang/kernels/jit/csrc/muse_spark/"
layers = "python/sglang/srt/layers/muse_spark/"
kernels = [jit+"decode_gate_up.cuh",jit+"decode_gate_up_launch.cuh",layers+"decode_gate_up.py",
           "python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py"]
down = [jit+"prompt_weight_pipeline.cuh",layers+"prompt_weight_pipeline.py",layers+"decode_mma_pair.py"]
sha = lambda data: hashlib.sha256(data).hexdigest()
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    (p/"objects").mkdir()
    public_git = subprocess.check_output(["git","-C",here.parents[1]/"sglang","rev-parse","--absolute-git-dir"],text=True).strip()
    env = dict(os.environ,GIT_INDEX_FILE=str(p/"index"),GIT_OBJECT_DIRECTORY=str(p/"objects"),
               GIT_ALTERNATE_OBJECT_DIRECTORIES=public_git+"/objects")
    def git(*args):
        return subprocess.check_output(["git","-C",stage,*args],env=env)
    def compare_source(directory, checks):
        git("read-tree",pin)
        git("apply","--cached","--whitespace=error-all",directory/"patch.diff")
        tree = git("write-tree").decode().strip()
        assert tree == json.loads((directory/"validation.json").read_bytes())["tree"]
        result={}
        for path in checks:
            content = (stage/path).read_bytes()
            assert git("show",":"+path) == content
            result[path]=sha(content)
        return result
    prompt = compare_source(v204,down)
    parent = compare_source(v306,kernels)
    wave = layers+"humming_prefill_waves.py"
    def jit_source(data):
        nodes=ast.parse(data).body
        return {n.name:ast.dump(n,include_attributes=False) for n in nodes if isinstance(n,ast.FunctionDef)
                and any(ast.unparse(d)=="triton.jit" for d in n.decorator_list)}
    donor_source=git("show",":"+wave)
    src=jit_source(donor_source)
    assert src.keys()=={"_pack_wave","_load_wave"} and src==jit_source((stage/wave).read_bytes())
    all_images=json.loads((v306/"packer_sm90.json").read_bytes())["cases"]
    assert len(all_images)==18 and all(r["source_sha256"]==sha(donor_source) for r in all_images)
    assert parent[kernels[1]]==json.loads((v306/"generated_ffi.json").read_bytes())["launcher_sha256"]
    passing_check=json.loads((here.parents[1]/"runs/7fcdf9abf39b486a.json").read_bytes())
    assert passing_check["result"]["passed"] and passing_check["result"]["patch"]["sha256"]==sha((v204/"patch.diff").read_bytes())
    print(json.dumps(dict(v306_native_sources=parent,v204_W2_sources=prompt,
        same_wave_Triton_definitions=sorted(src),donor_sm90_image_count=len(all_images),
        donor_sm90_images=[r["cubin_sha256"] for r in all_images],passing_parent_check=passing_check["id"],
        scope="Exact donor source and Triton definition identity. End-to-end graph integration and time still require an H200 run."),indent=2))
