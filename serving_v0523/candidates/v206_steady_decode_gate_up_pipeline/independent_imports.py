"""Check the production lazy import in the pinned package layout with driver-free test doubles."""
import ast
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from urllib.request import urlopen
from zipfile import ZipFile

here = Path(__file__).resolve().parent
stage = Path(sys.argv[1]) / "python"
candidate_sha = hashlib.sha256((here / "patch.diff").read_bytes()).hexdigest()
assert candidate_sha == "1919f7bd4dec48dd03da35ccd8eea2f4a13096c887b964c904b90cc5be40c6ce"
meta = json.load(urlopen("https://pypi.org/pypi/humming-kernels/0.1.12/json", timeout=10))
release = next(r for r in meta["urls"] if r["filename"].endswith("x86_64.whl"))
blob = urlopen(release["url"], timeout=10).read()
wheel_sha = hashlib.sha256(blob).hexdigest()
assert wheel_sha == release["digests"]["sha256"] == "cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331"
wheel = ZipFile(io.BytesIO(blob))
defs = {"humming/jit/compiler.py": "NVCCCompiler", "humming/jit/runtime.py": "KernelRuntime",
        "humming/kernel/humming.py": "HummingKernel"}
for path, name in defs.items():
    assert any(isinstance(node, ast.ClassDef) and node.name == name for node in ast.parse(wheel.read(path)).body)
header_path = "humming/include/humming/kernel/humming_ws.cuh"
header = wheel.read(header_path)
header_sha = hashlib.sha256(header).hexdigest()
assert header_sha == "6a148cc29b81d4009066667c347d9091a8a138009f7884226e9969a6936e2899"

def package(name):
    mod = ModuleType(name)
    mod.__path__ = [str(stage / name.replace(".", "/"))]
    sys.modules[name] = mod
    return mod

for name in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.muse_spark",
             "sglang.kernels", "sglang.kernels.jit", "sglang.kernels.jit.utils"):
    package(name)
jit = sys.modules["sglang.kernels.jit"]
jit.__file__ = str(stage / "sglang/kernels/jit/__init__.py")
sys.modules["sglang.kernels.jit.utils"].load_jit = lambda *_a, **_k: None
tp = package("torch")
tp.tensor = lambda x, **kw: x
tp.int64 = "int64"
for n in ("humming","humming.jit","humming.jit.compiler","humming.jit.runtime",
          "humming.kernel","humming.kernel.humming"):
    package(n)

class StandInKernel:
    def __init__(self, **cfg):
        self.__dict__.update(cfg)
        self.code = "#include <humming/kernel/humming_ws.cuh>"
        self.func, self.kernel_id, self.estimated_smem_size = 33, 47, 82944

    def check_context(self):
        self.prepare()

renders = []
sys.modules["humming.kernel.humming"].HummingKernel = StandInKernel
sys.modules["humming.jit.runtime"].KernelRuntime = SimpleNamespace(prepare=lambda k: renders.append(k.code))
sys.modules["humming.jit.compiler"].NVCCCompiler = object
pkg = "sglang.srt.layers.muse_spark."
up = importlib.import_module(pkg + "decode_gate_up")
assert pkg + "decode_mma_pair" not in sys.modules
up._launcher = lambda: SimpleNamespace(configure=lambda *_: True)
cfg = dict(gemm_type="grouped_contiguous", mma_type="wgmma", use_f16_accum=False,
           a_dtype="bfloat16", b_dtype="float4e2m1", c_dtype="bfloat16", bs_dtype="float8e4m3",
           weight_scale_group_size=16, weight_scale_type="group", weight_scale_2_type="none",
           sm_version=90, num_stages=4, num_threads=640, num_math_threads=512,
           use_tma_a=True,use_tma_b=True,use_tma_bs=True,use_tma_as=False,use_tma_bzp=False,
           use_tma_bias=False,use_tma_bs2=False,use_warp_spec=True,use_stream_k=False,use_tma_c=False,
           use_mbarrier=True,multi_cast_size_a=1,multi_cast_size_b=1)
orig_read = Path.read_text
def read(path, *args, **kwargs):
    if path == stage / header_path:
        return header.decode()
    return orig_read(path,*args,**kwargs)

with patch.object(Path, "read_text", read):
    for m in (8,16,24):
        args = (1, json.dumps(cfg), "{}", json.dumps(dict(block_shape=[m,256,128],warp_shape=[m,32,64])))
        assert up._compile(*args)[0].both_pairs
        down = importlib.import_module(pkg+"decode_mma_pair")
        assert down._prepare_down(1,json.dumps(cfg),"{}",json.dumps(dict(block_shape=[m,128,256],warp_shape=[m,32,64]))) == [0,1<<30,47,0]
assert len(renders) == 6
for code in renders:
    assert "#define MUSE_PAIR_LAST_K_SLICES" in code and "#include <humming/kernel/humming_ws.cuh>" not in code
assert up.__file__ == str(stage / "sglang/srt/layers/muse_spark/decode_gate_up.py")
assert down.__file__ == str(stage / "sglang/srt/layers/muse_spark/decode_mma_pair.py")
print(json.dumps(dict(patch_sha256=candidate_sha,package=release["filename"],wheel_sha256=wheel_sha,
    released_header_sha256=header_sha,reviewed_wheel_classes=defs,real_lazy_module_import=True,
    actual_production_preparations=6,projection_widths=[8,16,24],
    limitations="Actual Python adapters and published header; package parents, Torch, Humming classes and CUDA handles are test doubles. No driver, physical timing, or full launch."),indent=2))
