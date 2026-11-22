"""Model-load dispatch on the production prepare function with synthetic BF16/packed banks."""
import ast
from itertools import product
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as S
from unittest.mock import patch
import torch

stage = Path(sys.argv[1])
source = stage / "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
tree = ast.parse(source.read_text())
prepare = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="prepare_humming_nvfp4")
prepare.returns = None
prepare.args.args[0].annotation = None
torch.set_num_threads(1)
modules={}
def module(name,**fields):
    m=ModuleType(name)
    m.__dict__.update(fields)
    modules[name]=m

class Method:
    @staticmethod
    def prepare_layer_meta(layer, shape_n, shape_k, sublayer_name, **kwargs):
        return S(n=shape_n,k=shape_k,weight_name=sublayer_name+"_weight",weight_scale_name=sublayer_name+"_weight_scale",
                 to_str=lambda:sublayer_name)
    @staticmethod
    def get_default_tuning_configs(layer, *, gemm_type, sublayer_name):
        return [(0,64,{"block_shape":[8,256,128]}),(64,128,{"block_shape":[16,256,128]}),
                (128,256,{"block_shape":[24,256,128]}),(256,None,{"block_shape":[176,256,64]})]

def transform(meta, weights, already_padded):
    assert already_padded and weights["weight"].dtype==torch.int32
    e=weights["weight"].shape[0]
    return {"weight":torch.zeros((e,meta.k//16,2*meta.n),dtype=torch.int32),
            "weight_scale":torch.zeros((e,meta.k//16,meta.n),dtype=torch.uint8).view(torch.float8_e4m3fn)}

def bind(layer, meta, compute, configs, *,prompt=False):
    assert not hasattr(layer,"gate_up_proj") and "w13_weight" in dict(layer.named_parameters()) and meta=="w13"
    assert torch.equal(layer.w13_global_scale,torch.full((2,),2,dtype=torch.bfloat16))
    blocks=[json.loads(c)["block_shape"] for _,c in configs]
    assert blocks==([[176,256,64]] if prompt else [[8,256,128],[16,256,128],[24,256,128]])
    assert all(json.loads(c)["use_tma_c"] is False and json.loads(c)["use_stream_k"] is False for _,c in configs)
    return (layer.w13_weight.data_ptr(),layer.w13_weight_scale.data_ptr(),prompt)

module("humming",dtypes=S(bfloat16=1,float4e2m1=2,float8e4m3=3))
module("humming.config",GemmType=S(INDEXED=1,GROUPED_CONTIGUOUS=2))
module("humming.kernel.humming",HummingKernel=S(prepare_kernels=lambda *args:None))
module("humming.layer",HummingMethod=Method)
module("humming.schema.humming",HummingInputSchema=S,HummingWeightSchema=S)
module("humming.transform",transform_humming_tensors=transform)
module("sglang.srt.layers.muse_spark.decode_gate_up",prepare_gate_up=bind)
module("sglang.srt.layers.muse_spark.decode_mma_pair",prepare_down=lambda *args:{"down":True})
cfg=json.dumps({"gemm_type":"grouped_contiguous","use_f16_accum":False})
accepted={"prompt":0,"decode":0}
for arch,det,decode in product(((8,0),(9,0)),(False,True),(False,True)):
    layer=torch.nn.Module()
    layer.quant_config=S(group_size=16)
    layer.global_num_experts=16
    layer.num_local_experts=2
    for name,n in (("gate_up_proj",512),("down_proj",256)):
        setattr(layer,name,torch.zeros((2,n,128),dtype=torch.uint8))
        setattr(layer,name+"_weight_scale",torch.zeros((2,n,16),dtype=torch.uint8).view(torch.float8_e4m3fn))
        setattr(layer,name+"_weight_scale_2",torch.tensor([[2,3],[2,4]] if name=="gate_up_proj" else [2,3],dtype=torch.bfloat16))
    flags=S(SGLANG_ENABLE_MUSE_SPARK_HUMMING_GROUPED_DECODE=S(get=lambda:decode),
            SGLANG_ENABLE_DETERMINISTIC_INFERENCE=S(get=lambda:det))
    env=dict(torch=torch,json=json,envs=flags,_GROUPED_COMPUTE=cfg,_MAX_DECODE_EXPERTS=64,_MAX_DECODE_SLOTS=2048)
    exec(compile(ast.Module(body=[prepare],type_ignores=[]),str(source),"exec"),env)
    with patch.dict(sys.modules,modules),patch.object(torch.cuda,"get_device_capability",return_value=arch):
        env["prepare_humming_nvfp4"](layer)
    expected=(arch==(9,0) and not det)
    assert (layer.humming_prompt_gate_up is not None)==expected
    assert (layer.humming_decode_gate_up is not None)==(expected and decode)
    if expected and decode:assert layer.humming_prompt_gate_up[:2]==layer.humming_decode_gate_up[:2]
    accepted["prompt"]+=int(expected)
    accepted["decode"]+=int(expected and decode)
print(json.dumps(dict(cases=8,roles_bound=accepted,production="prepare_humming_nvfp4",mocked="vendor tensor transform, tuner, kernel compilation and CUDA capability",
    grouped_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),evidence="Calls actual preparation and registration with PyTorch BF16/uint8/FP8 tensors"),indent=2))
