"""Import entire production modules, verify option constructors and run real Humming Python preparation."""
from dataclasses import dataclass
from functools import cache
from importlib import import_module
import inspect
import typing
import json
from pathlib import Path
import sys
import subprocess
from types import ModuleType, SimpleNamespace
from unittest.mock import patch,PropertyMock
import torch

stage,packages=map(Path,sys.argv[1:])
tree="f1ed4dd4f1d567819e193b8c437717424ace0e9d"
def module(name,path=None):
    mod=ModuleType(name)
    mod.__path__=[str(path)] if path else []
    sys.modules[name]=mod
    if "." in name:
        base,member=name.rsplit(".",1)
        setattr(sys.modules[base],member,mod)
    return mod

loaded=[]
def pinned(name):
    p="python/"+name.replace(".","/")+".py"
    src=subprocess.check_output(["git","-C",stage,"show",tree+":"+p])
    m=module(name)
    m.__file__=str(stage/p)
    m.__package__=name.rsplit(".",1)[0]
    exec(compile(src,m.__file__,"exec"),m.__dict__)
    loaded.append(name)
    return m

for n in ("sglang","sglang.srt","sglang.srt.arg_groups","sglang.srt.arg_groups.fields",
          "sglang.srt.layers","sglang.srt.layers.muse_spark","sglang.kernels","sglang.kernels.jit","sglang.kernels.jit.utils"):
    module(n)
pinned("sglang.srt.arg_groups.arg_utils")
pinned("sglang.srt.arg_groups.choices")
pinned("sglang.srt.arg_groups.prefill_buffer_ceiling")
parallel=pinned("sglang.srt.arg_groups.fields.parallel").Parallel
spec=pinned("sglang.srt.arg_groups.fields.spec").Spec
import msgspec
options=["muse_sparse_prompt_exchange","speculative_dflash_sp_prefill_context","speculative_dflash_share_verify_schedule"]
for name in options:
    cls=parallel if name==options[0] else spec
    assert getattr(cls(),name) is False and getattr(cls(**{name:True}),name) is True
    assert any(f.name==name and (f.type is bool or typing.get_args(f.type)[0] is bool) for f in msgspec.structs.fields(cls))
env=pinned("sglang.srt.environ").envs
rt=pinned("sglang.srt.runtime_context")
draft=pinned("sglang.srt.layers.dflash_context_projection")
tiles=pinned("sglang.srt.layers.muse_spark.prefill_expert_tiles")
assert tiles.choose_tiles(None,[1]) is None and draft._prefill_context.get() is None
ex=rt._ConfigBag("exec");det=rt._ConfigBag("exec.deterministic");det._set("enable_deterministic_inference",False)
ex._set_sub("deterministic",det);rt.get_context()._config_bags={"exec":ex}
bag=rt._ConfigBag("parallel");bag._set(options[0],True);rt.get_parallel()._config=bag
jit=sys.modules["sglang.kernels.jit.utils"]
jit.cache_once=pinned("sglang.kernels.jit.utils.common").cache_once
jit.get_activation_cuda_cflags=lambda:[]
compiled=[]
fail=set()
def native(name,**kw):
    compiled.append(name)
    if name in fail: raise OSError("synthetic missing optional module")
    return SimpleNamespace(prepare=lambda *a: a[-1].numel()==520)
jit.load_jit=native
triton=module("triton");module("triton.language").constexpr=int
triton.jit=lambda fn:fn
route=pinned("sglang.srt.layers.muse_spark.muse_spark_humming_moe")
fields=["SGLANG_ENABLE_MUSE_SPARK_ROUTED_PROJECTION_PAIR","SGLANG_ENABLE_MUSE_SPARK_MEDIUM_ROUTE_PAIR",
        "SGLANG_ENABLE_MUSE_SPARK_COMPACT_PREFILL","SGLANG_ENABLE_MUSE_SPARK_FINALIZE_AR","SGLANG_ENABLE_DETERMINISTIC_INFERENCE"]
defaults={field:getattr(env,field).get() for field in fields}
assert list(defaults.values())==[True,True,True,True,False]

# Import real vendor schemas, metadata, transform and heuristic methods while
# replacing Linux-only CUDA registration, driver context, JIT and GPU operations.
vendor=packages/"humming"
module("humming",vendor);module("humming.kernel",vendor/"kernel")
module("humming.utils",vendor/"utils");module("humming.jit",vendor/"jit")
ops=module("humming.ops")
if not hasattr(torch,"float8_e8m0fnu"):
    torch.float8_e8m0fnu=object()
device=module("humming.utils.device");device.get_device_num_sms=lambda *a:132
module("humming.utils.jit").is_power_of_two=lambda v: v>0 and not v&(v-1)
@dataclass(kw_only=True)
class OfflineRuntime:
    @staticmethod
    def current_context():return 0
module("humming.jit.runtime").KernelRuntime=OfflineRuntime
from humming.kernel.humming import HummingKernel
from humming.layer import HummingMethod
from humming.schema.humming import HummingInputSchema,HummingWeightSchema
from humming.transform import transform_humming_tensors
from humming import dtypes
from humming.config import GemmType,MmaType
assert [dtypes.bfloat16.num_bits,dtypes.float4e2m1.num_bits,dtypes.float8e4m3.num_bits]==[16,4,8]
assert GemmType.INDEXED and GemmType.GROUPED_CONTIGUOUS and MmaType.WGMMA
calls={}
import ast
bindings={"HummingKernel":HummingKernel,"HummingMethod":HummingMethod,"HummingInputSchema":HummingInputSchema,
          "HummingWeightSchema":HummingWeightSchema,"transform_humming_tensors":transform_humming_tensors}
for name in (loaded[-1],"sglang.srt.layers.muse_spark.prefill_expert_tiles"):
    src=subprocess.check_output(["git","-C",stage,"show",tree+":python/"+name.replace(".","/")+".py"])
    for call in (q for q in ast.walk(ast.parse(src)) if isinstance(q,ast.Call)):
        expr=ast.unparse(call.func);part=expr.split(".")
        if part[0] not in bindings:continue
        fn=bindings[part[0]]
        for attr in part[1:]:fn=getattr(fn,attr)
        inspect.signature(fn).bind(*[None]*len(call.args),**{kw.arg:None for kw in call.keywords})
        calls[expr]=calls.get(expr,0)+1
assert len(calls)>=7
def repack(**kw):
    weight=kw["inputs"]
    e,n,words=weight.shape
    assert kw["use_wgmma"] and kw["activation_bits"]==16 and kw["weight_bits"]==4
    return torch.zeros(e,words//2,n*2,dtype=torch.int32)
ops.repack_weight=repack
import humming.transform as transforms
scale_transform = transforms.transform_humming_weight_scale
def cpu_scale_permute(scales, **kwargs):
    # Torch 2.5's CPU gather lacks FP8. The pinned operation is entirely a
    # permutation, so run its complete body on byte views, retaining all bits.
    if scales.dtype == torch.float8_e4m3fn:
        return scale_transform(scales.view(torch.uint8), **kwargs).view(scales.dtype)
    return scale_transform(scales, **kwargs)
transforms.transform_humming_weight_scale = cpu_scale_permute
props=SimpleNamespace(major=9,minor=0,multi_processor_count=132,shared_memory_per_multiprocessor=233472)
def prepare(soft,failed=False,decline_narrow=False,missing_tma=False):
    layer=torch.nn.Module();layer.quant_config=SimpleNamespace(group_size=16)
    layer.num_local_experts=2;layer.hidden_size=layer.intermediate_size_per_partition=256
    for name,n in (("gate_up_proj",512),("down_proj",256)):
        setattr(layer,name,torch.zeros(2,n,128,dtype=torch.uint8))
        setattr(layer,name+"_weight_scale",torch.full((2,n,16),8,dtype=torch.uint8).view(torch.float8_e4m3fn))
        setattr(layer,name+"_weight_scale_2",torch.ones((2,2) if name=="gate_up_proj" else (2,)))
    observed=[]
    original=HummingKernel.prepare_kernels
    def compiled_heuristics(meta,compute,tuning):
        inspect.signature(original).bind(meta,compute,tuning)
        decoded=json.loads(tuning)
        width=decoded["block_shape"][0]
        observed.append(width)
        if decline_narrow and width!=176:raise RuntimeError("synthetic unsupported tile")
        return None
    with patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),patch.object(torch.cuda,"get_device_name",return_value="H200"), \
         patch.object(torch.cuda,"get_device_properties",return_value=props),patch.object(HummingKernel,"prepare_kernels",side_effect=compiled_heuristics), \
         patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=soft):
        route.prepare_humming_nvfp4(layer)
    assert layer.w13_weight.shape==(2,16,1024) and layer.w2_weight.shape==(2,16,512)
    assert not hasattr(layer,"gate_up_proj") and set(layer.humming_grouped_tunings)=={"w13","w2"}
    assert layer.humming_route_pair==(soft and not failed) and layer.humming_medium_route_pair==(soft and not failed)
    assert (layer.humming_route_maps is not None)==(soft and not failed and not missing_tma)
    if soft and not failed and not decline_narrow:assert {136,152,176}==set(layer.humming_prefill_tiles[0])
    else:assert layer.humming_prefill_tiles is None
    return dict(optional_selected=soft and not failed,jit_failure=failed,missing_tma=missing_tma,narrow_build_declined=decline_narrow,requested_widths=observed,prepared_banks=sorted(layer.humming_metas))
cases=[prepare(False),prepare(True)]
tiles._prepare_cohort_variant.cache_clear()
cases.append(prepare(True,decline_narrow=True))
tiles._prepare_cohort_variant.cache_clear()
fail.add("muse_tma_route_pair")
route=pinned("sglang.srt.layers.muse_spark.muse_spark_humming_moe")
cases.append(prepare(True,missing_tma=True))
fail.update(compiled)
route=pinned("sglang.srt.layers.muse_spark.muse_spark_humming_moe")
cases.append(prepare(True,failed=True))
print(json.dumps(dict(complete_pinned_modules=loaded,native_option_defaults=options,real_environment_defaults=defaults,
    vendor_calls_bound=calls,full_startup=cases,jit_modules=compiled,torch=str(torch.__version__),msgspec=msgspec.__version__,
    mocked_boundaries=["parent package conveniences","Triton decorator","JIT wrappers and device query",
    "CUDA weight-repack operator","FP8 gather uses uint8 views with the real vendor permutation","vendor compilation/driver context","explicit RuntimeContext config bags"],
    hardware=False),indent=2))
