"""Execute the actual compact two-phase prompt entry with CPU tensor shapes and observed launch arguments.

Usage: python independent_entry.py extracted/humming assembled/sglang
No native kernels launch. The Hopper tuner, range table, Python launch routing and finalizer entry are real.
"""
import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

here=Path(__file__).resolve().parent
vendor,stage=map(Path,sys.argv[1:])
model=stage/'python/sglang/srt/layers/muse_spark'
prefix="sglang.srt.layers.muse_spark."
compute=json.dumps(dict(gemm_type="grouped_contiguous",use_f16_accum=False))


def functions(path,names,scope):
    tree=ast.parse(path.read_text())
    defs=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in defs}==set(names)
    tree=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),*defs],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(path),"exec"),scope)


def module(name,**values):
    obj=ModuleType(name)
    obj.__dict__.update(values)
    sys.modules[name]=obj
    return obj


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


with tempfile.TemporaryDirectory() as scratch:
    args=sys.argv
    sys.argv=[args[0],str(vendor),str(stage),scratch]
    base=next(here.parent.glob('v169*/vendor_codegen.py'))
    boot=dict(__name__="v279_boot",__file__=str(base))
    exec(compile(base.read_text().split('sp=importlib.util.spec_from_file_location',1)[0],str(base),'exec'),boot)
    sys.argv=args
    tunings={}; metas={}; bands={}
    with patch.object(torch.cuda,"get_device_capability",return_value=(9,0)),patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(multi_processor_count=132)):
        for name,n in (('w13',8192),('w2',4096)):
            dtype=boot['dtypes']
            obj=boot['LayerConfig'](shape_n=n,shape_k=4096,num_experts=32,a_dtype=dtype.bfloat16,b_dtype=dtype.float4e2m1,
                c_dtype=dtype.bfloat16,bs_dtype=dtype.float8e4m3,weight_scale_group_size=16)
            grouped=boot['scope']['Sm90Heuristics'].get_configs(obj,gemm_type=boot['GemmType'].GROUPED_CONTIGUOUS)
            assert list(grouped[-1][2]["block_shape"])==[176,256,64]
            assert list(grouped[0][2]["block_shape"])!=[176,256,64]
            tunings[name]=json.dumps(grouped[-1][2]|{"use_stream_k":False,"use_tma_c":False})
            bands[name]=dict(entries=len(grouped),last_interval=grouped[-1][:2],block=grouped[-1][2]["block_shape"])
            metas[name]=obj
top=model/'muse_spark_humming_moe.py'
assert "grouped[-1][2] |" in top.read_text() and "((None, grouped_tunings[\"w13\"]),), prompt=True" in top.read_text()
humming=module("humming.layer")
gate=module(prefix+"decode_gate_up",torch=torch)
down=module(prefix+"prompt_weight_pipeline",torch=torch)
pair=module(prefix+"decode_mma_pair")
layer_mod=module(prefix+"muse_spark_humming_moe",torch=torch,_GROUPED_COMPUTE=compute)
functions(model/'decode_gate_up.py',{'run_gate_up'},gate.__dict__)
functions(model/'prompt_weight_pipeline.py',{'_prompt_grid_ranges','run_prompt_down','_binding'},down.__dict__)
functions(model/'decode_mma_pair.py',{'run_down'},pair.__dict__)
functions(top,{'humming_grouped_expert_outputs'},layer_mod.__dict__)
fin={}
functions(model/'muse_spark_fp4_moe.py',{'_grouped_humming_and_combine_impl'},fin)
native=(vendor/'csrc/launcher/launcher.cpp').read_text()
for needle in ('int64_t shape_m = a.size(0);','shape_m > min_val && shape_m <= max_val','find_kernel_launch_data(configs, valid_shape_m, context)'):
    assert needle in native
layer=SimpleNamespace(hidden_size=4096,intermediate_size_per_partition=4096,num_local_experts=32,humming_metas=metas,
                      humming_grouped_tunings=tunings,_post_expert_norm_weight=None,_post_expert_norm_eps=0.0,_post_expert_norm_weight_first=False)
# Strided CPU views stand in for bound packed banks. The corresponding constructors' dtype/TMA
# requirements are tested separately; here we only use their live handles and exact run-time checks.
for name in ('w13','w2'):
    setattr(layer,name+'_weight',torch.empty((1,),dtype=torch.int32))
    setattr(layer,name+'_weight_scale',torch.empty((1,),dtype=torch.float8_e4m3fn))
    setattr(layer,name+'_global_scale',torch.ones((32,),dtype=torch.bfloat16))
events=[]; current={}; refuse=False


def gateop(kind,inputs,active,keys,offsets,scale,maps,handle,shared,width,sms):
    assert inputs.shape==active.shape==(keys.numel(),4096) and keys.dtype==torch.int32
    assert current['packed'] is inputs and current['offsets'] is offsets
    events.append(kind)
    if refuse:
        return False
    current['active']=active
    current['fused']=True
    return True


launcher=SimpleNamespace(run_prompt=lambda *a:gateop('gate1',*a),run_prompt_multicast=lambda *a:gateop('gate4',*a))


def downop(configs,inputs,outputs,expert_layout,**kw):
    assert inputs is current['active'] and expert_layout is current['offsets']
    assert inputs.data_ptr()!=outputs.data_ptr() and inputs.is_contiguous() and outputs.is_contiguous()
    if current.get('fused'):
        assert outputs is current['packed']
    n=inputs.size(0)
    entries=[r for r in configs.reshape(-1,4).tolist() if r[0]<n<=r[1]]
    assert len(entries)==1
    events.append('down'+str(entries[0][2])); current['raw']=outputs


sys.modules['humming'].ops=SimpleNamespace(launch_kernel=downop)


def fallback(layer,inputs,outputs,expert_layout,compute_config,tuning_config,sublayer_name):
    assert tuning_config==tunings[sublayer_name] and expert_layout is current['offsets']
    events.append('stock'+sublayer_name)
    if sublayer_name=='w2':
        assert inputs is current['active']; current['raw']=outputs


humming.HummingMethod=SimpleNamespace(forward_layer=fallback)
def silu(gate_up,activation):
    events.append('silu');current['active']=activation
module("sglang.kernels.ops.activation.activation",silu_and_mul=silu)
def finalize(raw,scores,**kw):
    assert raw is current['raw'] and kw['slot_row_indices'] is current['map'] and kw['expert_output_scales'] is layer.w2_global_scale
    events.append('final')
fin['fused_post_expert_norm_weighted_sum']=finalize
def pack(hidden,mapped,inputs):
    current['packed']=inputs
    events.append('pack' if inputs is not None else 'empty-pack')
workspace=SimpleNamespace(pack=pack)
out=[]
for n,wide,allow_cluster,failure,expected in (
    (0,4,True,'','empty-pack'),(704,4,True,'','pack gate1 down1 final'),(705,4,True,'','pack gate4 down1 final'),
    (8192,4,True,'','pack gate4 down4 final'),(8192,2,True,'','pack gate4 down2 final'),
    (10240,4,True,'','pack gate4 down1 final'),(16384,4,True,'','pack gate4 down4 final'),
    (8192,4,False,'','pack gate1 down4 final'),(8192,4,True,'gate','pack gate4 stockw13 silu down4 final'),
    (8192,4,True,'down','pack gate4 stockw2 final'),(8192,4,True,'both','pack stockw13 silu stockw2 final')):
    events.clear();current.clear();refuse=failure=='gate'
    layer.humming_prompt_gate_up=(launcher,None,tuple((x.data_ptr(),x.shape)for x in (layer.w13_weight,layer.w13_weight_scale)),
        {tunings['w13']:(None,111,128000,176),**({(tunings['w13'],'multicast'):(None,444,128000,176,128)} if allow_cluster else {})},132)
    bank=(down._binding(layer.w2_weight),down._binding(layer.w2_weight_scale))
    layer.humming_prompt_down=(down._prompt_grid_ranges(32,[1,0],[wide,128]),metas['w2'].to_str(),compute,
                               'mismatched' if failure in ('down','both') else tunings['w2'],bank)
    if failure=='both':
        layer.humming_prompt_gate_up=None
    row_count=max(n,1)
    hidden=torch.empty((row_count,4096),dtype=torch.bfloat16)
    local_ids=torch.zeros((row_count,8),dtype=torch.int32)
    mapped=torch.zeros_like(local_ids)
    offsets=torch.arange(33,dtype=torch.int32)*n//32
    keys=torch.repeat_interleave(torch.arange(32,dtype=torch.int32),offsets[1:]-offsets[:-1])
    current.update(offsets=offsets,map=mapped)
    result=torch.empty(1)
    fin['_grouped_humming_and_combine_impl'](layer,hidden,local_ids,None,result,peer_plan=(workspace,(keys,mapped,offsets)))
    assert events==expected.split(),(n,events,expected)
    if not n: assert result.item()==0
    out.append(dict(rows=n,wide=wide,allow_gate_cluster=allow_cluster,failure=failure,events=events.copy()))
print(json.dumps(dict(last_native_intervals=bands,cases=out,
    source_sha256={p.name:digest(p) for p in (top,model/'muse_spark_fp4_moe.py',model/'decode_gate_up.py',
        model/'prompt_weight_pipeline.py',model/'decode_mma_pair.py',vendor/'tune/sm90.py',vendor/'tune/base.py',vendor/'csrc/launcher/launcher.cpp')},
    scope='PyTorch CPU tensors, native tuner and production Python model-to-launch code. GPU kernels, layer weights, driver occupancy and finalizer arithmetic are represented by call recorders.'),indent=2))
