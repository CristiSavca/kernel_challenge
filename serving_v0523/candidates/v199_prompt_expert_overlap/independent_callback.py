"""Run V199 normal grouped prompt entry and inherited W13 plus W2 adapter before actual scaled finalizer wrapper."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
SRC = STAGE/'python/sglang/srt/layers/muse_spark'
base = next(HERE.parent.glob('v197_*/prompt_check.py'))
# Reuse its seeded full numeric per-expert projections, signed BF16 scales and pinned W13 launcher adapter.
setup = base.read_text().split('\ncounts={}')[0]
ns={'__name__':'v199_callback_fixture','__file__':str(base)}
exec(compile(setup,str(base),'exec'),ns)
layer, grouped, module = ns['layer'],ns['run'],ns['module']
events=ns['events']
ns['mode']='paired'
layer.use_humming=True;layer.global_num_experts=256;layer.hidden_size=128
layer.humming_metas={"w2":NS(to_str=lambda:"group-meta")}
layer._post_expert_norm=object()
layer._post_expert_norm_weight=ns['rand'](128)
layer._post_expert_norm_eps=1e-6
layer._post_expert_norm_weight_first=False
layer.w2_weight=torch.empty((8,8,256),dtype=torch.int32)
layer.w2_weight_scale=torch.empty((8,8,128),dtype=torch.float8_e4m3fn)
layer.locks=torch.zeros(2,dtype=torch.int32)
ordinary=ns['Humming'].forward_layer
down_mod=module('sglang.srt.layers.muse_spark.decode_mma_pair')
exec(compile((SRC/'decode_mma_pair.py').read_bytes(),'decode_mma_pair.py','exec'),vars(down_mod))
config=torch.tensor([0,1<<30,123,0],dtype=torch.int64)
prompt_module=module('sglang.srt.layers.muse_spark.prompt_weight_pipeline')
exec(compile((SRC/'prompt_weight_pipeline.py').read_bytes(),'prompt_weight_pipeline.py','exec'),vars(prompt_module))

def launch(*,configs,inputs,outputs,expert_layout,weight,weight_scale,locks):
    assert configs is config and weight is layer.w2_weight and weight_scale is layer.w2_weight_scale and locks is layer.locks
    ordinary(layer,inputs=inputs,outputs=outputs,expert_layout=expert_layout,compute_config=None,
             tuning_config=None,sublayer_name='w2')
    _,src,dst=events[-1];events[-1]=('W2 custom',src,dst)

module('humming',ops=NS(launch_kernel=launch))
module('humming.kernel.humming',HummingKernel=NS(current_context=lambda:17))
class DenseFallback(Exception):pass

def indexed(*args):raise DenseFallback()
model_hum=module('sglang.srt.layers.muse_spark.muse_spark_humming_moe',
          humming_decode_eligible=lambda *a:False,humming_expert_outputs=indexed,
          humming_grouped_decode_outputs=object(),local_humming_expert_ids=lambda _,ids:ids.int(),
          humming_grouped_expert_outputs=grouped)
final_file=STAGE/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'
fn,add=ns['fn'],ns['add']
scope={'torch':torch,'triton':NS(next_power_of_2=lambda x:1<<(x-1).bit_length()),'_MIN_TOKENS_FOR_FUSED':32,
       '_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT':256,'is_in_breakable_cuda_graph':lambda:True}
counts=Counter()
class Final:
    def __getitem__(self,grid):
        def perform(values,weight,scores,out,ids,scale_ids,scales,**k):
            assert k['HAS_ROW_MAP'] and k['HAS_WEIGHT'] and k['APPLY_NORM']
            assert not (k['HAS_IDS'] or k['HAS_EXPERT_SCALE'] or k['ROUND_PRODUCT'])
            assert grid==(len(out),)
            acc=torch.zeros_like(out,dtype=torch.float32)
            for col in range(scores.shape[1]):
                rows=torch.where(ids[:,col]>=0)[0]
                if len(rows):
                    v=values[ids[rows,col].long()].float()
                    if k['WEIGHT_FIRST']:v=v*weight.float()
                    v=v*torch.rsqrt((v*v).mean(1,keepdim=True)+k['eps'])
                    if not k['WEIGHT_FIRST']:v=v*weight.float()
                    acc[rows]+=v*scores[rows,col,None].float()
            out.copy_(acc)
            counts['scaled_weighted_finalizer_calls']+=1
        return perform
scope['_post_expert_norm_weighted_sum_kernel']=Final()
add(fn(final_file,'fused_post_expert_norm_weighted_sum'),scope)
for name in ['_grouped_humming_and_combine_impl','_forward_w4a16']:
    add(fn(ns['w4'],name,'MuseSpark1xFp4Experts'),scope)
layer._grouped_humming_and_combine=types.MethodType(scope['_grouped_humming_and_combine_impl'],layer)
def forward(x,scores,ids):
    return scope['_forward_w4a16'](layer,x,scores,ids,layer._post_expert_norm)

def bind():
    with patch.object(prompt_module,'version',return_value='0.1.12'),patch.object(prompt_module,'_prepare',return_value=config):
        result=prompt_module.prepare_prompt_down(layer,layer.humming_metas['w2'].to_str(),ns['env']['_GROUPED_COMPUTE'],layer.humming_grouped_tunings['w2'])
    assert result[0] is config
    return result

def check():
    for rows,kind in [(8192,'sparse'),(8201,'skew'),(8192,'empty')]:
        x=ns['rand'](rows,128)
        ids=torch.full((rows,8),-1,dtype=torch.int64)
        if kind=='sparse':
            ids[::97,0]=2;ids[::103,2]=7;ids[::97,5]=2
        elif kind=='skew':ids[::29,2]=5
        scores=ns['rand'](rows,8).float()
        for weight_first in (False,True):
            layer._post_expert_norm_weight_first=weight_first
            layer.humming_prompt_gate_up=None;layer.humming_prompt_down=None
            expected=forward(x,scores,ids)
            if kind=='empty':assert torch.count_nonzero(expected)==0
            for mode in ('bound','down_missing','gate_moved','stale_tuning','new_bank','new_scales','new_meta','new_compute'):
                layer.humming_prompt_gate_up=ns['pstate']('moved' if mode=='gate_moved' else 'paired')
                layer.humming_prompt_down=None if mode=='down_missing' else bind()
                oldt=layer.humming_grouped_tunings['w2'];oldw=layer.w2_weight;olds=layer.w2_weight_scale
                oldm=layer.humming_metas['w2'];oldc=ns['env']['_GROUPED_COMPUTE']
                if mode=='new_meta':layer.humming_metas['w2']=NS(to_str=lambda:'another-meta')
                if mode=='new_compute':ns['env']['_GROUPED_COMPUTE']='another-compute'
                if mode=='stale_tuning':layer.humming_grouped_tunings['w2']='another grouped tuning'
                if mode=='new_bank':layer.w2_weight=oldw.clone()
                if mode=='new_scales':layer.w2_weight_scale=olds.clone()
                try:
                    events.clear()
                    got=forward(x,scores,ids)
                    assert torch.equal(got,expected),(rows,mode,weight_first)
                    counts['compared_bf16_outputs']+=got.numel()
                    names=[e[0] for e in events]
                    custom=mode in ('bound','gate_moved')
                    assert names==([] if kind=='empty' else
                                  ['w13' if mode=='gate_moved' else 'paired','W2 custom' if custom else 'w2']), (mode,names)
                    counts[mode]+=1
                finally:
                    layer.w2_weight,layer.w2_weight_scale=oldw,olds
                    layer.humming_grouped_tunings['w2']=oldt
                    layer.humming_metas['w2']=oldm;ns['env']['_GROUPED_COMPUTE']=oldc
        try:forward(x[:8191],scores[:8191],ids[:8191])
        except DenseFallback:counts['below_admission_indexed']+=1
        else:raise AssertionError('below original grouped admission')
    layer.humming_prompt_down=bind()
    x=torch.empty((17,128),dtype=torch.bfloat16);offset=torch.tensor([0,17],dtype=torch.int32)
    v=torch.empty((128,17),dtype=torch.bfloat16).T
    for wrong_input,wrong_out in [(v,x),(x,v),(x.float(),x),(x,x.float()),(x[:,1:].contiguous(),x),(x,x[:,:-1].contiguous())]:
        n=len(events)
        assert not prompt_module.run_prompt_down(layer,wrong_input,wrong_out,offset,ns['env']['_GROUPED_COMPUTE'],layer.humming_grouped_tunings['w2'])
        assert len(events)==n
        counts['ineligible_tensor_layouts']+=1
    return counts

if __name__=='__main__':
    with torch.inference_mode(),patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
        result=check()
    print(json.dumps(dict(torch=torch.__version__,h200=False,counts=dict(result),
        official_patch_sha256=sha256((HERE/'patch.diff').read_bytes()).hexdigest(),
        parent_role='V198 grouped prompt expert branch, original W13 fusion and separate decode paths are retained',
        scope='Actual grouped prompt routing, W13 branch, W2 launch adapter, BF16 global scaling and mapped finalizer wrapper; host-modeled math and device launch',
        sha256={f.name:sha256(f.read_bytes()).hexdigest() for f in
            (SRC/'prompt_weight_pipeline.py',SRC/'decode_mma_pair.py',ns['hum'],ns['w4'],final_file,
             STAGE/'python/sglang/kernels/jit/csrc/muse_spark/prompt_weight_pipeline.cuh')}),
        indent=2))
