"""Render actual Humming 0.1.12 GPU kernel templates with a local CPU host driver.

Arguments are the extracted official vendor humming package, assembled SGLang tree,
and a temporary destination. Only the CUDA context and compiler/loader are stubbed.
"""
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import importlib.util
import json
import sys
import torch

vendor, stage, target = map(Path, sys.argv[1:])
target.mkdir(exist_ok=True)
for name, path in (("humming", vendor), ("humming.kernel", vendor/"kernel"),
                   ("humming.utils", vendor/"utils"), ("humming.jit", vendor/"jit")):
    mod = ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
for name in ("humming.utils.jit", "humming.jit.runtime", "humming.tune"):
    sys.modules[name] = ModuleType(name)
sys.modules["humming.utils.jit"].is_power_of_two = lambda n: n > 0 and (n & (n-1)) == 0
sys.modules["humming.tune"].get_heuristics_config = None
@dataclass(kw_only=True)
class OfflineRuntime:
    @staticmethod
    def current_context(): return 0
    def __post_init__(self):
        self.sm_version = 90
        self.init_kernel()
    def prepare(self):
        path = target / f"{self.shape_n}_{self.block_shape[0]}.cu"
        path.write_text(self.code + "\nauto ptr = reinterpret_cast<void*>(&" + self.kernel_expr + ");\n")
        stats[path.name] = {"shared_bytes": self.estimated_smem_size, "stages": self.num_stages,
                            "threads": self.num_threads,"warp_shape": self.warp_shape,"block_shape": self.block_shape}
sys.modules["humming.jit.runtime"].KernelRuntime = OfflineRuntime
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = object()  # Unused BF8 enum in older CPU builds.
sys.modules["humming.utils.device"] = ModuleType("humming.utils.device")
sys.modules["humming.utils.device"].get_device_num_sms = lambda *_: 132
from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.kernel.humming import HummingKernel
scope = dict(__name__="offline_tuner")
base = ModuleType("humming.tune.base")
sys.modules[base.__name__] = base
exec(compile((vendor/"tune/base.py").read_text(), str(vendor/"tune/base.py"),"exec"),base.__dict__)
exec(compile((vendor/"tune/sm90.py").read_text(),str(vendor/"tune/sm90.py"),"exec"),scope)
sp=importlib.util.spec_from_file_location("tile_policy",stage/"python/sglang/srt/layers/muse_spark/prefill_expert_tiles.py")
policy=importlib.util.module_from_spec(sp);sp.loader.exec_module(policy)
stats={}; configs={};metas={};objects={}
compute=json.dumps({"gemm_type":"grouped_contiguous","use_f16_accum":False})
def emit(layer_cfg,compute_cfg,tuning):
    kw=json.loads(tuning)
    gemm=json.loads(compute_cfg)
    meta=objects[layer_cfg]
    HummingKernel(shape_n=meta.shape_n,shape_k=meta.shape_k,num_experts=32,
          a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,
          c_dtype=dtypes.bfloat16,weight_scale_group_size=16,**(gemm | kw))
    return torch.tensor([0,1<<30,1,0])
with patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(multi_processor_count=132)),patch.object(HummingKernel,"prepare_kernels",side_effect=emit):
    for name,n in (("w13",8192),("w2",4096)):
        obj=LayerConfig(shape_n=n,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,
              bs_dtype=dtypes.float8e4m3,c_dtype=dtypes.bfloat16,weight_scale_group_size=16)
        cfg=scope["Sm90Heuristics"].get_config(obj,65408,gemm_type=GemmType.GROUPED_CONTIGUOUS)
        assert tuple(cfg["block_shape"])==(176,256,64)
        cfg.update(use_stream_k=False,use_tma_c=False)
        configs[name]=json.dumps(cfg);metas[name]=obj.to_str(); objects[metas[name]]=obj
        emit(metas[name],compute,configs[name])
    result=policy.prepare_tile_widths(metas,configs,compute,4096,4096,0)
    assert set(result[0])=={136,152,176} and len(stats)==6
print(json.dumps(stats,indent=2))
