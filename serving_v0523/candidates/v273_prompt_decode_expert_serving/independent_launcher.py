"""Exercise the exact wheel's C++ interval selector with the final CPU dispatch table."""
import ast
from hashlib import sha256
from io import BytesIO
import json
import os
from math import isqrt
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from urllib.request import urlopen
from zipfile import ZipFile
import torch

here=Path(__file__).resolve().parent
rel="python/sglang/srt/layers/muse_spark"
files=("prompt_weight_pipeline.py","decode_mma_pair.py","muse_spark_humming_moe.py")
if len(sys.argv)>1:
    model_files={n:(Path(sys.argv[1])/rel/n).read_text() for n in files}
else:
    # Rebuild only this published export in a temporary index. The campaign's
    # working tree and object database are both read-only.
    repo=here.parents[1]/"sglang"
    validation=json.loads((here/"validation.json").read_bytes())
    with tempfile.TemporaryDirectory(prefix="okc-v273-index-") as temp:
        obj=Path(temp)/"objects"; obj.mkdir()
        gitdir=subprocess.check_output(["git","-C",repo,"rev-parse","--absolute-git-dir"],text=True).strip()
        env=os.environ|dict(GIT_INDEX_FILE=str(Path(temp)/"index"),GIT_OBJECT_DIRECTORY=str(obj),
            GIT_ALTERNATE_OBJECT_DIRECTORIES=str(Path(gitdir)/"objects"))
        def git(*args):
            return subprocess.check_output(["git","-C",repo,*args],env=env)
        git("read-tree",validation["base"])
        git("apply","--cached","--whitespace=error-all",here/"patch.diff")
        assert git("write-tree").decode().strip()==validation["tree"]
        model_files={n:git("show",":"+rel+"/"+n).decode() for n in files}
manifest=json.loads((here/"expert_codegen.json").read_bytes())
wheel_sha="cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331"
url="https://files.pythonhosted.org/packages/ec/a2/f96912ee7d68f56e61c15555657ffa057b8afbaa21241c63342b5969bb74/humming_kernels-0.1.12-py3-none-manylinux_2_28_x86_64.whl"
with urlopen(url) as f:
    wheel=f.read()
assert sha256(wheel).hexdigest()==wheel_sha
with ZipFile(BytesIO(wheel)) as z:
    launcher=z.read("humming/csrc/launcher/launcher.cpp").decode()
    wrapper=z.read("humming/ops/__init__.py").decode()
    layer_class=z.read("humming/layer.py").decode()
    select_platform=z.read("humming/tune/__init__.py").decode()
    native_h200=z.read("humming/tune/sm90.py").decode()
    tuning=z.read("humming/config/config.py").decode()
assert sha256(launcher.encode()).hexdigest()==manifest["vendor_closure"]["csrc/launcher/launcher.cpp"]
def native(file, name, scope):
    node=next(n for n in ast.parse(file).body if getattr(n,"name",None)==name)
    exec("from __future__ import annotations\n"+ast.unparse(node),scope)
    return scope[name]

config_file=model_files["prompt_weight_pipeline.py"]
make=native(config_file,"_prompt_grid_ranges",dict(torch=torch))
narrow=[123,0]; wide=[456,120]
table=make(32,narrow,wide)
assert table.is_contiguous() and table.dtype==torch.int64 and table.device.type=="cpu"
ranges=table.view(-1,4).tolist()
assert ranges[0][0]==0 and ranges[-1][1]==1<<30
assert all(row[0]<row[1] for row in ranges)
assert all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
# Execute the real Python binding with isolated torch.ops transport only.
seen=[]
emulated=SimpleNamespace(Tensor=torch.Tensor,int64=torch.int64,
    ops=SimpleNamespace(humming=SimpleNamespace(launch_kernel=lambda *args:seen.append(args))))
call=native(wrapper,"launch_kernel",dict(torch=emulated))
inputs=torch.empty((173,4096),device="meta",dtype=torch.bfloat16)
outputs=torch.empty_like(inputs)
offsets=torch.empty(33,device="meta",dtype=torch.int32)
layer=SimpleNamespace(w2_weight="weights",w2_weight_scale="scales",locks="locks")
ops=SimpleNamespace(launch_kernel=call)
from unittest.mock import patch
with patch.dict(sys.modules,humming=SimpleNamespace(ops=ops)):
    native(model_files["decode_mma_pair.py"],"run_down",{})(table,layer,inputs,outputs,offsets)
assert len(seen)==1 and seen[0][0] is table and seen[0][1] is inputs
assert seen[0][8] is outputs and seen[0][12] is offsets and seen[0][-3:]==("locks",1,0)
assert "return get_heuristics_config(" in layer_class
assert 'if "H20" in name and "H200" not in name:' in select_platform
assert "90: Sm90Heuristics," in select_platform and "num_ctas_per_sm" not in select_platform
assert "if layer_config.a_dtype.num_bits == 16:\n            func = cls.get_config1" in native_h200
assert "num_ctas_per_sm" not in native_h200 and "num_ctas_per_sm: int = 1" in tuning
signature=(here.parent/"v270_wide_prompt_weight_reuse"/"occupancy.json")
assert json.loads(signature.read_bytes())["model_src_sha256"]==sha256(config_file.encode()).hexdigest()
for line in ["find_kernel_launch_data(configs, 1, context)", "int64_t shape_m = a.size(0);",
             "valid_shape_m = shape_m * (base_kernel_data.gemm_type_id == 1 ? top_k : 1);",
             "find_kernel_launch_data(configs, valid_shape_m, context);",
             "kernel_data.num_ctas_per_sm * get_num_sms(num_sms, dev);",
             "config.sharedMemBytes = kernel_data.smem_size;",
             "config.blockDimX = kernel_data.num_threads;",
             "kernel_data.multi_cast_size_a * kernel_data.multi_cast_size_b;"]:
    assert line in launcher
model=model_files["muse_spark_humming_moe.py"]
assert 'grouped[-1][2] | {"use_stream_k": False, "use_tma_c": False}' in model
assert model.index("if num_routes == 0:")<model.index("used_down = run_prompt_down")
body=launcher[launcher.index("inline int64_t find_kernel_configs_target_index"):
              launcher.index("inline KernelLaunchData find_kernel_launch_data")]
# Independent oracle: the actual table's (low,high] convention bins by ceil(M/experts).
# Test every native row in a complete 16K top-eight prompt, not only the boundaries.
points=list(range(1,131073))+[1<<30]
def choose(rows):
    k=(rows+31)//32; d=isqrt(k)
    samples=[max(1,k-d),k+d]
    sweeps=lambda m:sum((a+m-1)//m for a in samples)
    return wide[0] if sweeps(256)<sweeps(176) else narrow[0]
input_data=" ".join(str(v) for v in table.tolist())+"\n"+"\n".join(f"{m} {choose(m)}" for m in points)
cpp="""
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using IntArrayRef = std::vector<int64_t>;
#define ASSERT_CHECK(c,...) do { if (!(c)) std::abort(); } while(0)
"""+body+"""
int main() {
    std::string header; std::getline(std::cin,header);
    std::istringstream fields(header); IntArrayRef cfg;
    int64_t v; while(fields>>v) cfg.push_back(v);
    int64_t n,id; while(std::cin>>n>>id) {
        int64_t i=find_kernel_configs_target_index(cfg,n);
        assert(cfg[i]==id);
        assert((cfg[i]==123 && cfg[i+1]==0) || (cfg[i]==456 && cfg[i+1]==120));
    }
}
"""
with tempfile.TemporaryDirectory(prefix="okc-v273-selector-") as tmp:
    source=Path(tmp)/"selector.cc"; binary=Path(tmp)/"selector"
    source.write_text(cpp)
    subprocess.run(["clang++","-std=c++17","-O2",str(source),"-o",str(binary)],check=True)
    subprocess.run([str(binary)],input=input_data,text=True,check=True)

print(json.dumps(dict(wheel_sha256=wheel_sha, launcher_sha256=sha256(launcher.encode()).hexdigest(),
    source_sha256=sha256(config_file.encode()).hexdigest(), torch=torch.__version__,
    cxx_native_selector_rows=len(points), zero_local_rows="returned by model before calling Humming",
    actual_ranges=ranges, registered_native_python_call_order=True,
    h200_heuristic_class="Sm90Heuristics.get_config1 for BF16; num_ctas_per_sm defaults to one",
    native_h200_heuristics_sha256=sha256(native_h200.encode()).hexdigest(),
    launch_grid_conversion="NUM_CTAS_PER_SM * explicit_num_sms; wide target emits one CTA per SM and two CTAs per cluster.",
    scope="Exact native selector executed on CPU; pinned CUDA cluster launch fields source-checked, no accelerator or timing."),indent=2))
