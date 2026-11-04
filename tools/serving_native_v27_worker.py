"""Bounded V27 serving-component CUDA diagnostic; host import is stdlib only."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap
import time
import traceback
import types
from typing import Callable, Optional

IMAGE = 'lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af'
BASE = 'a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f'
PINS = {
    'NATIVE_NOTICE.txt':'43ac9707b8c367e0c92c2d2e9b0506170df89690e5af3d68df82af544fe37e08',
    'NATIVE_LICENSE.txt':'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
    'SGLANG_LICENSE.txt':'1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747',
    'native_v27.py':'7310dd760cee4ae4802b1290d46d3dbb2e871a1cde131b78bb6e69743c23d0c1',
    'candidate_experts.py':'c507f29d313bcee8aec96314e1ebcfcb38d8235f29487490565655302918ccf3',
    'stock_experts.py':'1d377424e4c1f486f72a0e8ea67ea41dca4bd3f6929aa3729f5f7480861ed352',
    'stock_marlin_utils.py':'a9011b5af7f8c9d25755d2ea4125e9fbdad06b862138be3ca27c85fc2f8a8e08',
    'stock_finalizer.py':'9b816e971817f96a22d2d09424dc8dfa6f6020e18de73fb895674c76494fe591',
    'checks.py':'9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',
}
DEPENDENCIES = {
    'srt/layers/muse_spark_layers.py':'930e66da715bae576ae681f2dfd8353a36a9ca469a229d88fa3f77bc1e74cef0',
    'srt/layers/quantization/marlin_utils.py':'468b7d842c77c8cd29ca8fb4211fe60d33b74838423462152025347800622cf4',
    'kernels/ops/quantization/gptq_marlin_repack.py':'f6b7ef8dc42a44eea1f594cca33e2848bed12ec7f28b1860de5f86ffb034071b',
    'kernels/ops/activation/activation.py':'416b4d2eb24d2abfbd215277f508b9786ade0c2fc736e470d37621cc3cd605bc',
    'kernels/ops/moe/moe_wna16_marlin.py':'a34fa1daf24a6286d5d027c6ae45dac5fea144abd606a65b722fecd091513398',
}
FILES = frozenset(PINS) | {'native_v27_worker.py'}
BUDGET = 240
SEED = 20261104
RESULT = Path('/tmp/okc-result.json')
FOLDER = Path('/tmp/okc-native-v27')
MAX_BYTES = 1024 * 1024
TIMED = (1,8,16,32,128,256)
POLICY = {'whole_relative_l2':.015,'row_factor':5.,'row_norm_floor':.1,
          'allzero_exact':True,'timing_geomean_speed_min':1.03,
          'each_timing_latency_max':1.02,'timed_tokens':list(TIMED),
          'paired_rounds':5,'repeats':4,'cache_protocol':'warm_no_flush',
          'no_threshold_retune':True,'diagnostics_only':True,
          'performance_promotion':False,'timing_scope':'eager_whole_chain_cuda_events',
          'scale_coverage':'PRMT_only_LUT_not_attempted',
                          'global_scale_fixture':'varying_per_expert_distinct_gate_up_down_v1'}
FINALIZER_IMPORT = 'from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum'
FINALIZER_REPLACEMENT = 'fused_post_expert_norm_weighted_sum = _owned_finalizer'
POLICY_IMPORT = '    from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel'
POLICY_REPLACEMENT = '    get_disagg, get_exec, get_parallel = _qualified_fixture_getters'


def sha(b):
    return hashlib.sha256(b.encode() if isinstance(b,str) else b).hexdigest()


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def cases():
    result=[]
    for rank in range(8):
        for pattern in ('remote_before','live','remote_after'):
            result.append(dict(name=f'rank{rank}_{pattern}',t=1,rank=rank,pattern=pattern,norm='first',eps=1e-8))
    for t in (0,8,16,31,32,128,256,383,384,385,769,16384):
        result.append(dict(name=f'boundary_{t}',t=t,rank=3,pattern='mixed',norm='first',eps=1e-8))
    result.extend([
        dict(name='standard_norm',t=31,rank=3,pattern='mixed',norm='standard',eps=1e-5),
        dict(name='expert_skew',t=384,rank=3,pattern='skew',norm='first',eps=3e-6),
    ])
    for i,c in enumerate(result):c['seed']=SEED+101*i
    return result


def manifest_sha():
    return sha(canonical({'cases':cases(),'policy':POLICY,'graph_shape':32,'graph_replays':3,
                          'scale_coverage':'PRMT_only_LUT_not_attempted',
                          'global_scale_fixture':'varying_per_expert_distinct_gate_up_down_v1'}))


def adapt(source):
    changed=source
    for old,new in ((FINALIZER_IMPORT,FINALIZER_REPLACEMENT),(POLICY_IMPORT,POLICY_REPLACEMENT)):
        if changed.count(old)!=1 or new in changed:raise ValueError('Unexpected V27 private import boundary')
        changed=changed.replace(old,new,1)
    restored=changed.replace(FINALIZER_REPLACEMENT,FINALIZER_IMPORT).replace(POLICY_REPLACEMENT,POLICY_IMPORT)
    if restored!=source:raise ValueError('Private import adapter did not restore')
    return changed


def extract(source,names,cls=None):
    tree=ast.parse(source)
    body=tree.body if cls is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    selected=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in selected}!=set(names):raise ValueError('Missing original method')
    # Preserve each original AST, including decorators; no rewritten arithmetic.
    return ast.Module(body=selected,type_ignores=[])


def validate_payload(p):
    expected=dict(serving_native_v27=True,gpu='H200',image=IMAGE,timeout_seconds=300,
                  validation_budget_seconds=BUDGET,seed=SEED,pinned_commit=BASE,
                  case_manifest_sha256=manifest_sha(),policy=POLICY,case_descriptors=cases())
    for k,v in expected.items():
        if type(p.get(k)) is not type(v) or p[k]!=v:raise ValueError('Fixed V27 policy mismatch: '+k)
    if set(p.get('files',{}))!=FILES or set(p.get('file_sha256',{}))!=FILES:raise ValueError('Unexpected flat payload')
    for name,pin in {**PINS,'native_v27_worker.py':sha(Path(__file__).read_bytes())}.items():
        raw=p['files'][name]
        if type(raw) is not str or len(raw.encode())>MAX_BYTES or sha(raw)!=pin or p['file_sha256'][name]!=pin:
            raise ValueError('Source mismatch: '+name)
    adapt(p['files']['native_v27.py'])


def build_payload(root,image=IMAGE):
    root=Path(root); checkout=root/'serving/sglang'
    def stock(path):
        return subprocess.check_output(['git','show',BASE+':python/sglang/'+path],cwd=checkout,text=True,timeout=15)
    overlay=root/'serving/candidates/native_serving_v27/candidate/python/sglang'
    files={'SGLANG_LICENSE.txt':subprocess.check_output(['git','show',BASE+':LICENSE'],cwd=checkout,text=True,timeout=15),
           'NATIVE_LICENSE.txt':(root/'third_party/public_scout/keell0renz-890809c/LICENSE').read_text(),
           'NATIVE_NOTICE.txt':(root/'third_party/public_scout/keell0renz-890809c/NOTICE').read_text(),
           'native_v27.py':(overlay/'kernels/ops/muse_spark_v12/native_serving_v27.py').read_text(),
           'candidate_experts.py':(overlay/'srt/layers/muse_spark_fp4_moe.py').read_text(),
           'stock_experts.py':stock('srt/layers/muse_spark_fp4_moe.py'),
           'stock_marlin_utils.py':stock('srt/layers/quantization/marlin_utils_fp4.py'),
           'stock_finalizer.py':stock('kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py'),
           'checks.py':(root/'muse_moe_w4a16/checks.py').read_text(),
           'native_v27_worker.py':Path(__file__).read_text()}
    payload=dict(serving_native_v27=True,gpu='H200',image=image,timeout_seconds=300,
                 validation_budget_seconds=BUDGET,seed=SEED,pinned_commit=BASE,
                 case_manifest_sha256=manifest_sha(),policy=POLICY,case_descriptors=cases(),
                 files=files,file_sha256={n:sha(s) for n,s in files.items()})
    validate_payload(payload);return payload


def _save(r):
    raw=json.dumps(r,indent=2,sort_keys=True,allow_nan=False).encode()
    if len(raw)>MAX_BYTES:raise RuntimeError('Checkpoint exceeds1MiB')
    tmp=RESULT.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(RESULT)


def _load(path,name,injected=None):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    if injected:module.__dict__.update(injected)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def qualified_getters():
    ns=types.SimpleNamespace
    ex=ns(overlap=ns(enable_two_batch_overlap=False,enable_single_batch_overlap=False),
          offload=ns(cpu_offload_gb=0,offload_group_size=-1),features=ns(enable_memory_saver=False),
          moe=ns(enable_eplb=False),graph=ns(enable_torch_compile=False,cuda_graph_config=ns(
              decode=ns(backend='full'),prefill=ns(backend='breakable'))))
    return (lambda:ns(enable_pdmux=False),lambda:ex,lambda:ns(enable_layernorm_sp=False))


def make_stock_api(torch,p,finalizer):
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace,marlin_permute_scales,marlin_permute_bias
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
    scope=dict(torch=torch,Callable=Callable,Optional=Optional,nn=torch.nn,logger=logging.getLogger('v27_stock'),
               marlin_make_workspace=marlin_make_workspace,marlin_permute_scales=marlin_permute_scales,
               marlin_permute_bias=marlin_permute_bias,gptq_marlin_repack=gptq_marlin_repack,
               fused_post_expert_norm_weighted_sum=finalizer)
    names=('nvfp4_marlin_process_scales','nvfp4_marlin_process_global_scale',
           '_repack_moe_fp4_weight_for_marlin','_permute_moe_fp4_scales_for_marlin','prepare_moe_nvfp4_layer_for_marlin')
    prep=extract(p['files']['stock_marlin_utils.py'],names)
    exec(compile(prep,str(FOLDER/'stock_marlin_utils.py'),'exec'),scope)
    tree=ast.parse(p['files']['stock_experts.py'])
    constant=next(n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='_MARLIN_BLOCK_SIZE_M_CANDIDATES' for t in n.targets))
    exec(compile(ast.Module([constant],type_ignores=[]),'stock_constant','exec'),scope)
    methods=extract(p['files']['stock_experts.py'],('_marlin_expert_outputs','_forward_marlin','set_post_expert_norm'),'MuseSpark1xFp4Experts')
    exec(compile(methods,str(FOLDER/'stock_experts.py'),'exec'),scope)
    candidate_scope=dict(scope)
    setter=extract(p['files']['candidate_experts.py'],('set_post_expert_norm',),'MuseSpark1xFp4Experts')
    exec(compile(setter,str(FOLDER/'candidate_experts.py'),'exec'),candidate_scope)
    class Layer(torch.nn.Module):
        def __init__(self,rank=3,count=32):
            super().__init__()
            self.local_expert_offset=rank*32;self.num_local_experts=count;self.global_num_experts=256
            self.hidden_size=4096;self.intermediate_size_per_partition=4096;self.top_k=8
            self.params_dtype=torch.bfloat16;self.use_marlin=True;self.expert_tp=False;self.use_megamoe=False
            self.quant_config=types.SimpleNamespace(group_size=16)
            self.moe_runner_config=types.SimpleNamespace(is_gated=True)
            self._post_expert_norm=None;self._fused_finalize_op=None
            self.register_buffer('_post_expert_norm_weight',None,persistent=False)
            self.register_buffer('_expert_map',mapping(torch,rank,count),persistent=False)
    Layer._forward_marlin=scope['_forward_marlin']
    Layer._marlin_expert_outputs=scope['_marlin_expert_outputs']
    Layer.set_post_expert_norm=scope['set_post_expert_norm']
    return Layer,scope['prepare_moe_nvfp4_layer_for_marlin'],candidate_scope['set_post_expert_norm'],extraction_evidence(p)


def extraction_evidence(p):
    prep=extract(p['files']['stock_marlin_utils.py'],(
        'nvfp4_marlin_process_scales','nvfp4_marlin_process_global_scale',
        '_repack_moe_fp4_weight_for_marlin','_permute_moe_fp4_scales_for_marlin','prepare_moe_nvfp4_layer_for_marlin'))
    methods=extract(p['files']['stock_experts.py'],('_marlin_expert_outputs','_forward_marlin','set_post_expert_norm'),'MuseSpark1xFp4Experts')
    setter=extract(p['files']['candidate_experts.py'],('set_post_expert_norm',),'MuseSpark1xFp4Experts')
    return {'preparation_ast_sha256':sha(ast.dump(prep,include_attributes=False)),
            'stock_methods_ast_sha256':sha(ast.dump(methods,include_attributes=False)),
            'candidate_setter_ast_sha256':sha(ast.dump(setter,include_attributes=False)),
            'scope':'Verbatim original functions/methods on real nn.Module fixtures; not model construction or scheduler.'}


def mapping(torch,rank,count=32):
    result=torch.full((256,),-1,dtype=torch.int32,device='cuda')
    result[rank*32:rank*32+count]=torch.arange(count,dtype=torch.int32,device='cuda')
    return result


def weights(torch,count=32,lut=False):
    gen=torch.Generator(device='cuda').manual_seed(SEED+int(lut))
    mean_sq=4.**7*math.exp(2*(math.log(2)*.4)**2)
    gs=1./math.sqrt(8.6*mean_sq*4096)
    result={}
    for code,scale,rows in (('w13','w13_sf',8192),('w2','w2_sf',4096)):
        result[code]=torch.randint(0,256,(count,rows,2048),device='cuda',dtype=torch.uint8,generator=gen)
        z=torch.randn((count,rows,256),device='cuda',generator=gen)
        result[scale]=(2.**(7.+.4*z)).clamp(8.,448.).to(torch.float8_e4m3fn)
        if lut:result[scale].view(torch.uint8).view(-1)[:8]=torch.arange(8,device='cuda',dtype=torch.uint8)
    factor=torch.linspace(.8,1.2,count,device='cuda',dtype=torch.float32)
    result['w13_gs']=torch.stack((gs*factor,gs*(1.6-.3*factor)),dim=1)
    result['w2_gs']=gs*factor.flip(0)
    return result


def bind_raw(torch,layer,raw):
    for key,name in (('w13','gate_up_proj'),('w2','down_proj'),('w13_sf','gate_up_proj_weight_scale'),
                     ('w2_sf','down_proj_weight_scale'),('w13_gs','gate_up_proj_weight_scale_2'),('w2_gs','down_proj_weight_scale_2')):
        layer.register_parameter(name,torch.nn.Parameter(raw[key].clone(),requires_grad=False))


def prepare_stock(torch,layer,raw,prepare):
    for key,name in (('w13','w13_weight'),('w2','w2_weight'),('w13_sf','w13_weight_scale'),
                     ('w2_sf','w2_weight_scale'),('w13_gs','w13_weight_scale_2'),('w2_gs','w2_weight_scale_2')):
        value=raw[key][:,0] if key=='w13_gs' else raw[key]
        layer.register_parameter(name,torch.nn.Parameter(value.clone().contiguous(),requires_grad=False))
    prepare(layer)


def norm_objects(torch):
    from sglang.srt.layers.muse_spark_layers import MuseSpark1xRMSNorm,MuseSpark1xRMSNormWithInputScale
    gen=torch.Generator(device='cuda').manual_seed(SEED+999)
    def make(cls,eps):
        n=cls(4096,eps=eps,has_weight=True,zero_centered_gamma=True).to(device='cuda',dtype=torch.bfloat16)
        with torch.no_grad():n.weight.copy_(.1*torch.randn((4096,),generator=gen,device='cuda'))
        n.precompute_effective_weight();return n
    return make(MuseSpark1xRMSNorm,1e-5),{
        ('first',1e-8):make(MuseSpark1xRMSNormWithInputScale,1e-8),
        ('first',3e-6):make(MuseSpark1xRMSNormWithInputScale,3e-6),
        ('standard',1e-5):make(MuseSpark1xRMSNorm,1e-5)}


def inputs(torch,c):
    g=torch.Generator(device='cuda').manual_seed(c['seed']);t=c['t'];lo=c['rank']*32
    x=torch.randn((t,4096),generator=g,device='cuda').to(torch.bfloat16)
    if c['pattern']=='skew':
        local=torch.arange(8,device='cuda')[None,:].expand(t,8)
    else:local=torch.rand((t,32),generator=g,device='cuda').argsort(dim=1)[:,:8]
    remote=(local+((c['rank']+1)%8)*32).to(torch.int32)
    ids=(local+lo).to(torch.int32)
    if c['pattern'].startswith('remote'):ids=remote
    elif c['pattern']=='mixed':ids[:,2:]=remote[:,2:]
    scores=torch.rand((t,8),generator=g,device='cuda')+.1
    scores=scores/scores.sum(-1,keepdim=True)
    return x,scores,ids


def bits(torch,a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def compare(torch,out,ref,checks):
    if out.shape!=ref.shape or out.dtype!=torch.bfloat16 or out.device!=ref.device:
        return {'pass':False,'kind':'invalid','reason':'shape/dtype/device mismatch'}
    if not out.numel():return {'pass':True,'kind':'empty','reason':'empty exact shape/dtype/device'}
    if not bool(torch.isfinite(ref).all()):return {'pass':False,'kind':'invalid','reason':'nonfinite stock output'}
    if not bool((ref!=0).any()):
        return {'pass':bool(torch.equal(out,torch.zeros_like(ref))),'kind':'allzero','reason':'allzero requires exactzeros'}
    passed,message=checks.compare(out,ref,.015)
    r=ref.float();d=out.float()-r
    rows=r.reshape(-1,4096).norm(dim=-1);floor=(.1*rows.mean()).clamp(min=1e-12)
    err=float(d.norm()/r.norm().clamp(min=1e-12))
    worst=float((d.reshape(-1,4096).norm(dim=-1)/torch.maximum(rows,floor)).max())
    return {'pass':bool(passed),'kind':'relative','reason':message,'relative_l2':err if math.isfinite(err) else None,
            'worst_row':worst if math.isfinite(worst) else None}


def clone_binding(torch,Layer,source,rank,owner,setter):
    layer=Layer(rank,source.num_local_experts)
    layer._native_v27_owner=owner
    layer._native_v27_bank=dict(source._native_v27_bank,lo=rank*32)
    for name in source._buffers:
        if name.startswith('_native_v27_weight_'):
            layer.register_buffer(name,getattr(source,name),persistent=False)
    layer.set_post_expert_norm=types.MethodType(setter,layer)
    return layer


def timing_gate(rows):
    if [r.get('t') for r in rows]!=list(TIMED):return {'pass':False,'reason':'incomplete timings'}
    ratios=[r['candidate_us']/r['baseline_us'] for r in rows]
    if any(not math.isfinite(x) or x<=0 for x in ratios):raise ValueError('Invalid timings')
    gm=math.exp(sum(math.log(x) for x in ratios)/len(ratios))
    return {'pass':1/gm>=1.03 and max(ratios)<=1.02,'speedup_geomean':1/gm,
            'maximum_latency_ratio':max(ratios),'significance_claim':False,'diagnostics_only':True,'performance_promotion':False}


def _positive(x):
    return type(x) in (int,float) and math.isfinite(x) and x>0


def _comparison(row, complete):
    if type(row) is not dict or type(row.get('pass')) is not bool:
        raise ValueError('Malformed comparison')
    if row.get('kind') not in ('invalid','empty','allzero','relative') or (row['pass'] and row['kind']=='invalid'):
        raise ValueError('Unspecified numerical comparison')
    if row['kind']=='relative' and not {'relative_l2','worst_row'}<=row.keys():
        raise ValueError('Missing numerical metrics')
    for key in ('relative_l2','worst_row'):
        if key in row and row[key] is not None and (type(row[key]) not in (int,float)
                or not math.isfinite(row[key]) or row[key]<0):
            raise ValueError('Invalid numerical metric')
    if complete and row['pass'] is not True:raise ValueError('Failed comparison in completed report')
    if row['pass'] and 'relative_l2' in row:
        if row['relative_l2'] is None or row.get('worst_row') is None:
            raise ValueError('Passing comparison lacks finite metrics')
        if row['relative_l2']>POLICY['whole_relative_l2'] or row['worst_row']>POLICY['row_factor']*POLICY['whole_relative_l2']:
            raise ValueError('Passing comparison exceeds fixed tolerance')


def validate_result(r,p):
    """Accept trustworthy partial evidence, never promote it to completion.

    True means the completed eager timing *diagnostic* passes its fixed screen;
    False also covers a valid partial/failed report or a completed timing miss.
    This API never grants deployment or performance promotion.
    """
    validate_payload(p)
    for k,v in {'image':IMAGE,'file_sha256':p['file_sha256'],'case_manifest_sha256':manifest_sha(),
                'policy':POLICY,'official':False,'full_engine_validated':False,
                'performance_promotion':False}.items():
        if type(r.get(k)) is not type(v) or r[k]!=v:raise ValueError('Result provenance mismatch: '+k)
    status=r.get('status')
    if status not in ('running','validation_complete','validation_failed','validation_incomplete'):
        raise ValueError('Unknown validation state')
    complete=status=='validation_complete'
    rows=r.get('cases')
    if type(rows) is not list or len(rows)>len(cases()) or [x.get('case') for x in rows]!=cases()[:len(rows)]:
        raise ValueError('Missing/reordered case prefix')
    if complete and len(rows)!=len(cases()):raise ValueError('Incomplete correctness coverage')
    for row in rows:
        _comparison(row.get('comparison'),complete)
        for key in ('baseline_after_exact','inputs_unchanged'):
            if type(row.get(key)) is not bool or (complete and not row[key]):raise ValueError('Invalid case gate: '+key)
    if 'environment' in r:
        env=r['environment']
        if env.get('device_count')!=1 or env.get('capability')!=[9,0] or 'H200' not in str(env.get('device')):
            raise ValueError('Unexpected device evidence')
    if 'dependencies' in r and r['dependencies']!=DEPENDENCIES:raise ValueError('Installed source mismatch')
    if 'installation' in r:
        i=r['installation']
        if any(type(i.get(k)) is not bool for k in ('registered_bank_only','prmt','independent_raw_storages')) or any(
                type(i.get(k)) is not int or i[k]<0 for k in ('actual_nonzero_rank','workspace_instances','weight_bytes')):
            raise ValueError('Malformed installation evidence')
        if complete and (i.get('actual_nonzero_rank')!=3 or i.get('registered_bank_only') is not True
                or i.get('prmt') is not True or i.get('workspace_instances')!=1
                or i.get('independent_raw_storages') is not True
                or i.get('weight_bytes')!=905970816):raise ValueError('Invalid actual installation evidence')
    if 'baseline_stability' in r:
        st=r['baseline_stability']
        if (st.get('t')!=32 or st.get('rank')!=3 or st.get('pattern')!='mixed' or type(st.get('exact')) is not bool
                or st.get('before_any_candidate_forward') is not True):
            raise ValueError('Malformed baseline stability evidence')
        if complete and st['exact'] is not True:raise ValueError('Stock baseline was not stable')
    if 'serial_layers' in r:
        _comparison(r['serial_layers'],complete)
        if type(r['serial_layers'].get('one_workspace')) is not bool or (complete and not r['serial_layers']['one_workspace']):
            raise ValueError('Serial ownership failure')
    if 'streams' in r:
        stream=r['streams'];checks=stream.get('comparisons',[])
        if len(checks)!=2:raise ValueError('Expected two independent-owner streams')
        pointers=stream.get('workspace_ptrs',[])
        if len(pointers)!=2 or any(len(x)!=5 or any(type(y) is not int or y<=0 for y in x) for x in pointers) or set(pointers[0])&set(pointers[1]):
            raise ValueError('Workspace storage ownership evidence invalid')
        for check in checks:_comparison(check,complete)
        if stream.get('pass') != all(x['pass'] for x in checks):raise ValueError('Stream gate mismatch')
        for key in ('inputs_unchanged','distinct_workspaces'):
            if type(stream.get(key)) is not bool or (complete and not stream[key]):raise ValueError('Stream ownership failure')
    if 'graphs' in r:
        g=r['graphs'];checks=g.get('checks',[])
        expected=[(i,j) for j in range(3) for i in range(2)]
        if (g.get('expected_checks')!=6 or len(checks)>6
                or [(x.get('graph'),x.get('replay')) for x in checks]!=expected[:len(checks)]):
            raise ValueError('Invalid graph replay prefix')
        for check in checks:
            _comparison(check,complete)
            if any(type(check.get(k)) is not bool for k in ('inputs_unchanged','fresh_inputs_changed')):
                raise ValueError('Graph input gate missing')
        if g.get('pass')!=all(x['pass'] and x['inputs_unchanged'] and x['fresh_inputs_changed'] for x in checks):raise ValueError('Graph gate mismatch')
        outptr=g.get('output_ptrs',[])
        if len(outptr)!=2 or any(type(x) is not int or x<=0 for x in outptr) or len(set(outptr))!=2:
            raise ValueError('Graph output ownership evidence invalid')
        if complete and (len(checks)!=6 or not g['pass'] or g.get('distinct_workspaces') is not True):
            raise ValueError('Graph ownership coverage incomplete')
    timings=r.get('timings')
    if type(timings) is not list or (complete and len(timings)!=len(TIMED)) or [x.get('t') for x in timings]!=list(TIMED)[:len(timings)]:
        raise ValueError('Invalid timing prefix')
    for row in timings:
        _comparison(row.get('comparison'),complete)
        samples=row.get('samples_us')
        if (type(samples) is not list or len(samples)!=2 or any(type(x) is not list or len(x)!=POLICY['paired_rounds']
                or any(not _positive(y) for y in x) for x in samples)):
            raise ValueError('Invalid raw timing samples')
        if (row.get('baseline_us')!=statistics.median(samples[0]) or row.get('candidate_us')!=statistics.median(samples[1])):
            raise ValueError('Timing medians do not reproduce')
    if 'timing_gate' in r and r['timing_gate']!=timing_gate(timings):raise ValueError('Timing gate mismatch')
    if r.get('optional_lut')!={'status':'not_attempted_scope','reason':'Fixed PRMT-only run; no LUT GPU coverage claimed.'}:
        raise ValueError('Invalid declared scale coverage')
    if not complete:return False
    required=('baseline_stability','environment','dependencies','installation','serial_layers','streams','graphs','timing_gate','private_imports','extraction','immutable_bank_views')
    if any(key not in r for key in required):raise ValueError('Missing completion evidence')
    if r['private_imports']!={'reverse_restoration':True,'transformed_sha256':sha(adapt(p['files']['native_v27.py'])),
                              'installed_modules_mutated':False}:raise ValueError('Import adaptation evidence mismatch')
    if r['extraction']!=extraction_evidence(p):raise ValueError('Original function extraction mismatch')
    if r['immutable_bank_views']!={'rank_metadata_rebindings':list(range(8)),'actual_installations':1,
                                  'unique_native_banks':1,'all_buffers_share_storage':True}:
        raise ValueError('Unexpected native bank ownership')
    return r['timing_gate']['pass']


def check_budget(start,reserve=0):
    if time.monotonic()-start>=BUDGET-reserve:
        raise TimeoutError('Fixed240s coverage budget exhausted')


def run(payload):
    start=time.monotonic()
    report=dict(status='running',official=False,full_engine_validated=False,image=IMAGE,
                file_sha256=payload.get('file_sha256'),case_manifest_sha256=manifest_sha(),policy=POLICY,
                cases=[],timings=[],performance_promotion=False,
                optional_lut={'status':'not_attempted_scope','reason':'Fixed PRMT-only run; no LUT GPU coverage claimed.'},
                scope=['Generated local weights, actual component methods on nn.Module fixtures.',
                       'Qualified private runtime getters; not real scheduler/capture-lane policy validation.',
                       'Same bank reused with explicit rank metadata rebind; only initial rank3 is installed.',
                       'No full model,8GPU,checkpoint peak memory or TTFT claim.'])
    def save(stage):
        report['stage']=stage;report['elapsed_seconds']=time.monotonic()-start;_save(report)
    def budget(reserve=0):
        check_budget(start,reserve)
    try:
        validate_payload(payload);save('imports')
        import torch
        import sglang
        if torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0):raise RuntimeError('Expected one H200')
        report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                                   device_count=1,capability=list(torch.cuda.get_device_capability(0)))
        package=Path(sglang.__file__).resolve().parent
        observed={n:sha((package/n).read_bytes()) for n in DEPENDENCIES}
        if observed!=DEPENDENCIES:raise RuntimeError('Installed serving dependencies differ from pin')
        report['dependencies']=observed
        FOLDER.mkdir(exist_ok=True)
        for name,source in payload['files'].items():
            (FOLDER/name).write_text(source)
        finalizer=_load(FOLDER/'stock_finalizer.py','okc_v27_stock_finalizer').fused_post_expert_norm_weighted_sum
        checks=_load(FOLDER/'checks.py','okc_v27_checks')
        transformed=adapt(payload['files']['native_v27.py']);path=FOLDER/'adapted_native_v27.py';path.write_text(transformed)
        native=_load(path,'okc_private_native_v27',{'_owned_finalizer':finalizer,'_qualified_fixture_getters':qualified_getters()})
        if native._serial_policy() is not True:raise RuntimeError('Qualified policy fixture not selected')
        report['private_imports']={'reverse_restoration':True,'transformed_sha256':sha(transformed),
                                   'installed_modules_mutated':False}
        Layer,stock_prepare,candidate_setter,evidence=make_stock_api(torch,payload,finalizer)
        report['extraction']=evidence
        with torch.inference_mode():
            budget();save('generate_weights')
            raw=weights(torch);pre,norms=norm_objects(torch);post=norms['first',1e-8]
            owner=native.NativeModelOwner();candidate=Layer(3)
            candidate._native_v27_owner=owner;candidate.post_expert_norm=post
            candidate.set_post_expert_norm=types.MethodType(candidate_setter,candidate)
            bind_raw(torch,candidate,raw)
            raw_independent=all(getattr(candidate,name).data_ptr()!=raw[key].data_ptr() for key,name in native._V27_RAW)
            if not raw_independent:raise RuntimeError('Raw/candidate parameter alias')
            budget();save('native_install_and_compile')
            if native.try_install_native(candidate) is not True:raise RuntimeError('Actual installation declined')
            if not candidate._native_v27_bank['prmt']:raise RuntimeError('Expected primary PRMT fixture')
            report['installation']={'actual_nonzero_rank':3,'registered_bank_only':
                all(name in candidate._buffers for name in native._V27_BUFFERS)
                and not any(name in candidate._parameters for _,name in native._V27_RAW),
                'weight_bytes':sum(t.numel()*t.element_size() for t in candidate._native_v27_bank['w']),
                'workspace_instances':len(owner._workspaces),'prmt':True,
                'independent_raw_storages':raw_independent}
            budget();save('stock_prepare')
            baseline=Layer(3);prepare_stock(torch,baseline,raw,stock_prepare)
            torch.cuda.synchronize();budget()
            stability_args=inputs(torch,dict(t=32,rank=3,pattern='mixed',seed=SEED+8999))
            first=baseline._forward_marlin(*stability_args,post,pre).clone()
            second=baseline._forward_marlin(*stability_args,post,pre)
            report['baseline_stability']={'t':32,'rank':3,'pattern':'mixed','exact':bits(torch,first,second),
                'before_any_candidate_forward':True}
            save('baseline_stability_checked')
            if not report['baseline_stability']['exact']:
                raise RuntimeError('Stock baseline is not bit-stable before candidate execution; contamination comparison is inconclusive')
            report['memory_after_prepare']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                                            'peak_allocated':torch.cuda.max_memory_allocated(),
                                            'limit':'Fixture keeps raw copies+stock+one native bank, not deployed model memory.'}
            candidates={rank:(candidate if rank==3 else clone_binding(torch,Layer,candidate,rank,owner,candidate_setter))
                        for rank in range(8)}
            shared=all(all(a.data_ptr()==b.data_ptr() for a,b in zip(view._native_v27_bank['w'],candidate._native_v27_bank['w'])) for view in candidates.values())
            report['immutable_bank_views']={'rank_metadata_rebindings':list(range(8)),'actual_installations':1,
                                            'unique_native_banks':1,'all_buffers_share_storage':shared}
            if not shared:raise RuntimeError('Rank views duplicated native bank storage')
            def pair(c):
                budget();save('case:'+c['name'])
                args=inputs(torch,c);snap=[t.clone() for t in args]
                baseline._expert_map.copy_(mapping(torch,c['rank']))
                norm=norms[c['norm'],c['eps']];current=candidates[c['rank']]
                ws=owner.workspace(current._native_v27_bank['device']);ws.expert_out.fill_(float('nan'))
                if c['t']==0:
                    ref=args[0].new_empty((0,4096))  # stock outer model's zero-token guard
                else:ref=baseline._forward_marlin(*args,norm,pre).clone()
                out=native.native_forward(current,*args,norm,pre)
                if c['t']==0:again=ref
                else:again=baseline._forward_marlin(*args,norm,pre)
                row={'case':c,'comparison':compare(torch,out,ref,checks),
                     'baseline_after_exact':bits(torch,ref,again),
                     'inputs_unchanged':all(bits(torch,a,b) for a,b in zip(args,snap))}
                report['cases'].append(row);save('checked:'+c['name'])
                if not row['comparison']['pass'] or not row['baseline_after_exact'] or not row['inputs_unchanged']:
                    raise RuntimeError('Fixed numerical/state gate failed: '+c['name'])
            for c in cases():pair(c)
            # Two serial layers use one workspace and a single immutable bank.
            budget();save('serial_layers')
            c=dict(cases()[28],t=32,rank=3,pattern='mixed',seed=SEED+9001)
            args=inputs(torch,c);baseline._expert_map.copy_(mapping(torch,3))
            a=candidates[3];b=clone_binding(torch,Layer,candidate,3,owner,candidate_setter)
            y1=native.native_forward(a,*args,post,pre).clone()
            y2=native.native_forward(b,y1,args[1],args[2],post,pre)
            r1=baseline._forward_marlin(*args,post,pre)
            r2=baseline._forward_marlin(r1,args[1],args[2],post,pre)
            serial=compare(torch,y2,r2,checks)
            report['serial_layers']={**serial,'one_workspace':len(owner._workspaces)==1}
            save('serial_layers_checked')
            if not serial['pass'] or not report['serial_layers']['one_workspace']:raise RuntimeError('Serial two-layer comparison failed')
            # Separate owners share immutable bank tensors but never scratch.
            budget();save('streams_graphs')
            owners=[native.NativeModelOwner(),native.NativeModelOwner()]
            views=[clone_binding(torch,Layer,candidate,3,o,candidate_setter) for o in owners]
            workspaces=[o.workspace(candidate._native_v27_bank['device']) for o in owners]
            workspace_ptrs=[[t.data_ptr() for t in [*ws.scratch,ws.expert_out,ws.local_ids]] for ws in workspaces]
            if set(workspace_ptrs[0])&set(workspace_ptrs[1]):raise RuntimeError('Workspace alias')
            streams=[torch.cuda.Stream(),torch.cuda.Stream()]
            streams_results=[];stream_args=[]
            torch.cuda.synchronize()
            for i in range(2):
                stream_args.append(inputs(torch,dict(c,seed=SEED+9100+i)))
            snapshots=[[t.clone() for t in a] for a in stream_args]
            torch.cuda.synchronize()
            for i in range(2):
                with torch.cuda.stream(streams[i]):
                    streams_results.append(native.native_forward(views[i],*stream_args[i],post,pre))
            torch.cuda.synchronize()
            stream_checks=[compare(torch,streams_results[i],baseline._forward_marlin(*stream_args[i],post,pre),checks) for i in range(2)]
            report['streams']={'pass':all(x['pass'] for x in stream_checks),'comparisons':stream_checks,
                               'inputs_unchanged':all(bits(torch,a,b) for aa,bb in zip(stream_args,snapshots) for a,b in zip(aa,bb)),
                               'distinct_workspaces':True,'workspace_ptrs':workspace_ptrs,'overlap_measured':False}
            save('streams_checked')
            if not report['streams']['pass'] or not report['streams']['inputs_unchanged']:raise RuntimeError('Separate-stream check failed')
            graphs=[];goutputs=[]
            for i in range(2):
                budget();torch.cuda.synchronize()
                with torch.cuda.stream(streams[i]):
                    native.native_forward(views[i],*stream_args[i],post,pre)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=streams[i]):
                    output=native.native_forward(views[i],*stream_args[i],post,pre)
                graphs.append(graph);goutputs.append(output)
            graph_checks=[]
            for replay in range(3):
                budget()
                fresh=[inputs(torch,dict(c,seed=SEED+9200+replay*2+i)) for i in range(2)]
                changed=[not bits(torch,stream_args[i][0],fresh[i][0]) for i in range(2)]
                for aa,bb in zip(stream_args,fresh):
                    for a,b in zip(aa,bb):a.copy_(b)
                torch.cuda.synchronize()
                for i in range(2):
                    with torch.cuda.stream(streams[i]):graphs[i].replay()
                torch.cuda.synchronize()
                for i in range(2):
                    check=compare(torch,goutputs[i],baseline._forward_marlin(*fresh[i],post,pre),checks)
                    check.update(graph=i,replay=replay,fresh_inputs_changed=changed[i],inputs_unchanged=all(bits(torch,a,b) for a,b in zip(stream_args[i],fresh[i])))
                    graph_checks.append(check)
                report['graphs']={'pass':all(x['pass'] and x['inputs_unchanged'] and x['fresh_inputs_changed'] for x in graph_checks),
                                  'checks':graph_checks,'expected_checks':6,'distinct_workspaces':True,
                                  'output_ptrs':[o.data_ptr() for o in goutputs],
                                  'scope':'Explicit component graphs, not SGLang graph backend or scheduler.'}
                save('graphs_replay:'+str(replay))
                if not report['graphs']['pass']:raise RuntimeError('Changed-input graph replay failed')
            # Whole chain, including prenorm/finalizer/copies; warm cache paired events.
            budget();save('timing')
            for t in TIMED:
                budget(5)
                descriptor=dict(c,t=t,seed=SEED+10000+t)
                args=inputs(torch,descriptor)
                functions=[lambda:baseline._forward_marlin(*args,post,pre),
                           lambda:native.native_forward(candidates[3],*args,post,pre)]
                refs=[f() for f in functions]
                check=compare(torch,refs[1],refs[0],checks)
                if not check['pass']:raise RuntimeError('Timed-case correctness failed')
                samples=[[],[]]
                for round_index in range(POLICY['paired_rounds']):
                    budget()
                    for role in ((0,1) if round_index%2==0 else (1,0)):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(POLICY['repeats']):value=functions[role]()
                        end.record();end.synchronize()
                        samples[role].append(begin.elapsed_time(end)*1000/POLICY['repeats'])
                report['timings'].append({'t':t,'baseline_us':statistics.median(samples[0]),
                    'candidate_us':statistics.median(samples[1]),'samples_us':samples,'comparison':check})
                save('timed:'+str(t))
            report['timing_gate']=timing_gate(report['timings'])
            budget();report['status']='validation_complete';save('complete')
            validate_result(report,payload)
    except TimeoutError as exc:
        report['status']='validation_incomplete';report['error']=str(exc);save('budget_exhausted')
    except Exception as exc:
        report['status']='validation_failed';report['error']=str(exc)
        report['traceback']=traceback.format_exc()[-12000:];save('failed')
    return report


if __name__=='__main__':
    payload=json.loads(Path(sys.argv[1]).read_text())
    result=run(payload)
    print(canonical({'status':result['status'],'result_path':str(RESULT)}))
    raise SystemExit(0 if result['status']=='validation_complete' else 1)
