"""Exercise the selected W2 startup chain on the released generator, keeping its actual CUDA inputs."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parent
script=HERE.parent/'v169_grouped_expert_tile_cohorts/vendor_codegen.py'
exec(compile(script.read_text().split('sp=importlib.util.spec_from_file_location',1)[0],str(script),'exec'))
layers=stage/'python/sglang/srt/layers/muse_spark'
for name in ('decode_mma_pair','residual_prompt_mma','live_decode_reduce'):
    mod=ModuleType('sglang.srt.layers.muse_spark.'+name)
    mod.__file__=str(layers/(name+'.py'))
    exec(compile(Path(mod.__file__).read_bytes(),mod.__file__,'exec'),vars(mod))
    sys.modules[mod.__name__]=mod
jit=ModuleType('sglang.kernels.jit');jit.__file__=str(stage/'python/sglang/kernels/jit/__init__.py')
kernels=ModuleType('sglang.kernels');kernels.jit=jit;sys.modules[kernels.__name__]=kernels
compiler=ModuleType('humming.jit.compiler');compiler.NVCCCompiler=type('NVCCCompiler',(),{});sys.modules[compiler.__name__]=compiler
pair=sys.modules['sglang.srt.layers.muse_spark.decode_mma_pair']._prepare_down
OfflineRuntime.check_context=lambda self:None
sources={};calls=[]
def emit(k):
    m,n,_=k.block_shape
    assert k.both_pairs and k.live_reduction
    suffix = '' if k.shape_n == k.shape_k == 4096 else f'_d{k.shape_n}_k{k.shape_k}_e{k.num_experts}'
    name=f'w2_m{m}_n{n}'+suffix
    calls.append(n)
    if fail and n==256: raise ValueError('simulated wide kernel refusal')
    k.kernel_id=m*1000+n
    source=k.code+'\nauto ptr = reinterpret_cast<void*>(&'+k.kernel_expr+');\n'
    (target/(name+'.cu')).write_text(source)
    sources[name]=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       block=k.block_shape,warp=k.warp_shape,stages=k.num_stages,smem=k.estimated_smem_size)
OfflineRuntime.prepare=emit
with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)), patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(multi_processor_count=132)):
    obj=LayerConfig(shape_n=4096,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,
                bs_dtype=dtypes.float8e4m3,c_dtype=dtypes.bfloat16,weight_scale_group_size=16)
    configs=scope['Sm90Heuristics'].get_configs(layer_config=obj,gemm_type=GemmType.GROUPED_CONTIGUOUS)
    for lo,hi,config in configs:
        if lo>=256:continue
        m=config['block_shape'][0]
        tuning=json.dumps(config|dict(use_stream_k=False,use_tma_c=False))
        for fail in (False,True):
            calls.clear()
            ans=pair(int(fail),obj.to_str(),json.dumps({'gemm_type':'grouped_contiguous','use_f16_accum':False}),tuning)
            assert calls == ([128] if m==8 else [256,128] if fail else [256]),(m,calls)
            assert ans.tolist()==[0,1<<30,m*1000+calls[-1],0]
    # Additional grouped metadata with both eligible and declined output alignments.
    for n,k,experts,expected in ((5120,3072,19,256),(5120-128,3072,19,128)):
        alt=LayerConfig(shape_n=n,shape_k=k,num_experts=experts,a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,
                        bs_dtype=dtypes.float8e4m3,c_dtype=dtypes.bfloat16,weight_scale_group_size=16)
        config=next(t for lo,hi,t in configs if t["block_shape"][0]==16 and lo<256)
        tuning=json.dumps(config|dict(use_stream_k=False,use_tma_c=False))
        fail=False;calls.clear()
        ans=pair(73,alt.to_str(),json.dumps({"gemm_type":"grouped_contiguous","use_f16_accum":False}),tuning)
        assert calls==[expected] and ans.tolist()==[0,1<<30,16*1000+expected,0]
print(json.dumps(sources,indent=2))
