"""Fixed actual-JIT dual-ID router component screen; stdlib-only host import."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback
import types

IMAGE = 'lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af'
BASE = 'a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f'
PINS = {'baseline_routing.py':'9d88c80d5351c0aba45fa671f49d83026b0891ef3ffa9d4271d062f0992d0e99',
        'baseline_routing.cuh':'0a965acc078e4d8fd97349f98a1f386be75baed7745a49d031cd5c1111e04ef0',
        'candidate_routing.py':'7c6531af814c46f3d432271178f0d418826f48826e5bc6fec7dfa8414f20e5cf', 'candidate_routing.cuh':'52d2d6d85a6fc1ed1e4558131ad3414a3c7b7f208c60fcf64182590630ffb268',
        'jit_dependencies.json':'285d4156a5cdaee8586fcdf8423539be477721bc674817bb4fbdbb91cbb72c30'}
FILES = frozenset(PINS) | {'router_local_ids.py'}
PY_PATH = 'python/sglang/kernels/ops/muse_spark_v12/routing.py'
CU_PATH = 'python/sglang/kernels/jit/csrc/muse_spark_v12/routing.cuh'
TOKENS = (0, 1, 3, 8, 16, 31, 32, 128, 256)
PATTERNS = ('random', 'ties', 'extreme', 'local', 'remote', 'mixed')
TIMED = (8, 16, 32, 128, 256)
SEED = 20261104
REPEATS = 64
ROUNDS = 12
ROUTE_ARGS = {'router_output_multiplier':0.17677669529663687,
              'router_score_division_eps':1e-15}
RESULT = Path('/tmp/okc-result.json')
FOLDER = Path('/tmp/okc-router-local-ids')
MAX_BYTES = 1024 * 1024
POLICY = {'latency_geomean_max':.95, 'each_latency_max':1.02,
          'exact_score_bits':True, 'exact_global_and_local_ids':True,
          'removed_index_kernels':1, 'repeats_per_graph':REPEATS, 'paired_rounds':ROUNDS,
          'cache_protocol':'warm_no_flush', 'timed_rank':0, 'timed_pattern':'mixed',
          'route_scale':1.0, 'route_arguments':ROUTE_ARGS,
          'route_argument_origin':'pinned_source_config_defaults_not_checkpoint'}
IMPORT = '''from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)'''
ADAPTED_IMPORT = '# Owned import adapter: utilities injected into this private module.'


def sha(x): return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()
def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
def descriptors():
    return [{'t':t, 'rank':r, 'pattern':p, 'seed':SEED + i}
            for i, (t, r, p) in enumerate((t,r,p) for t in TOKENS for r in range(8) for p in PATTERNS)]
def manifest(): return sha(canonical({'cases':descriptors(), 'policy':POLICY, 'padding_ranks':list(range(8))}))
def adapter_sha():
    source=Path(__file__).read_text(); tree=ast.parse(source)
    return sha('\n'.join(ast.get_source_segment(source,n) for n in tree.body
                         if isinstance(n,ast.FunctionDef) and n.name in ('adapt','private_module')))


def adapt(source):
    if source.count(IMPORT) != 1 or ADAPTED_IMPORT in source: raise ValueError('Unexpected utility import')
    tree = ast.parse(source)
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'sglang.kernels.jit.utils']
    if len(imports) != 1 or [(n.name,n.asname) for n in imports[0].names] != [(n,None) for n in ('cache_once','is_arch_support_pdl','load_jit','make_cpp_args')]:
        raise ValueError('Unexpected utility symbols')
    changed = source.replace(IMPORT, ADAPTED_IMPORT)
    if changed.replace(ADAPTED_IMPORT, IMPORT) != source: raise ValueError('Adapter restoration failed')
    return changed


def private_module(source, role, header, utilities, evidence, collect=None):
    """Only this module's load_jit binding changes; no installed globals are patched."""
    if role not in ('baseline','candidate'): raise ValueError('Unknown role')
    changed = adapt(source); header = Path(header).resolve()
    prefix = f'okc_router_v24_{role}_{sha(source + header.read_text())[:16]}'
    calls = []
    def local_loader(*args, **kwargs):
        expected_tail = tuple(utilities.make_cpp_args(8, 8, 0, 1, utilities.is_arch_support_pdl()))
        expected_kw = {'cuda_files':['muse_spark_v12/routing.cuh'],
                       'cuda_wrappers':[('fused_muse_spark_routing', f'fused_muse_spark_routing<{", ".join(expected_tail)}>')]}
        if args != ('muse_spark_v12_routing', *expected_tail) or kwargs != expected_kw:
            raise ValueError('Unexpected actual router JIT invocation')
        owned = {**kwargs, 'cuda_files':[str(header)]}
        module_args = (prefix, *args[1:])
        calls.append({'module_args':list(module_args), 'header':str(header), 'header_sha256':sha(header.read_bytes())})
        if len(calls) != 1: raise RuntimeError('Unexpected extra router specialization/build')
        module = utilities.load_jit(*module_args, **owned)
        if collect is not None: evidence['build'] = collect(module_args, header)
        return module
    module = types.ModuleType(f'okc_private_router_{role}')
    module.__file__ = str(header.with_suffix('.py'))
    module.__dict__.update({n:getattr(utilities,n) for n in ('cache_once','is_arch_support_pdl','make_cpp_args')})
    module.load_jit = local_loader
    exec(compile(changed, module.__file__, 'exec'), module.__dict__)
    evidence.update(source_sha256=sha(source), transformed_sha256=sha(changed),
                    adapter_sha256=adapter_sha(), reverse_restoration=True,
                    private_module_name=module.__name__, calls=calls)
    return module


def dependency_manifest(root):
    """Hashes only: bounded source dependency closure, no package vendoring."""
    package = (Path(root) / 'serving/sglang/python/sglang').resolve()
    selected = set((package/'kernels/jit/utils').rglob('*.py'))
    selected.add(package/'srt/environ.py')
    include = package/'kernels/jit/include'
    pending = [include/'sgl_kernel'/p for p in ('tensor.h','utils.h','type.cuh','utils.cuh','warp.cuh')]
    while pending:
        path = pending.pop()
        if path in selected: continue
        selected.add(path)
        for kind, name in re.findall(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', path.read_text(), re.M):
            nested = include/name if name.startswith('sgl_kernel/') else path.parent/name if kind == '"' else None
            if nested is not None and nested.is_file(): pending.append(nested.resolve())
    if len(selected) > 64: raise ValueError('Dependency closure exceeded bounded scope')
    return {str(p.relative_to(package)):sha(p.read_bytes()) for p in sorted(selected)}


def validate_payload(p):
    fixed = dict(serving_router_local_ids=True, gpu='H200', image=IMAGE, timeout_seconds=600,
                 validation_budget_seconds=540, seed=SEED, case_manifest_sha256=manifest(), policy=POLICY,
                 case_descriptors=descriptors(), pinned_commit=BASE)
    if any(p.get(k) != v or type(p.get(k)) is not type(v) for k,v in fixed.items()): raise ValueError('Fixed router policy changed')
    if set(p.get('files',{})) != FILES or set(p.get('file_sha256',{})) != FILES: raise ValueError('Unexpected router payload files')
    for name, pin in {**PINS, 'router_local_ids.py':sha(Path(__file__).read_bytes())}.items():
        data = p['files'][name]
        if not isinstance(data,str) or len(data.encode()) > MAX_BYTES or sha(data) != pin or p['file_sha256'][name] != pin:
            raise ValueError(f'Frozen router source mismatch: {name}')
    for role in ('baseline','candidate'): adapt(p['files'][role+'_routing.py'])
    deps = json.loads(p['files']['jit_dependencies.json'])
    if not 1 <= len(deps) <= 64 or any('..' in Path(k).parts or Path(k).is_absolute() or not re.fullmatch('[0-9a-f]{64}',v) for k,v in deps.items()):
        raise ValueError('Invalid dependency manifest')


def build_payload(root, image=IMAGE):
    root = Path(root); candidate = root/'serving/candidates/router_local_ids_v24/candidate'
    files = {'baseline_routing.py':subprocess.check_output(['git','show',BASE+':'+PY_PATH], cwd=root/'serving/sglang', text=True, timeout=15),
             'baseline_routing.cuh':subprocess.check_output(['git','show',BASE+':'+CU_PATH], cwd=root/'serving/sglang', text=True, timeout=15),
             'candidate_routing.py':(candidate/PY_PATH).read_text(), 'candidate_routing.cuh':(candidate/CU_PATH).read_text(),
             'jit_dependencies.json':canonical(dependency_manifest(root)), 'router_local_ids.py':Path(__file__).read_text()}
    p = dict(serving_router_local_ids=True,gpu='H200',image=image,timeout_seconds=600,
             validation_budget_seconds=540,seed=SEED,case_manifest_sha256=manifest(),policy=POLICY,
             case_descriptors=descriptors(),pinned_commit=BASE,files=files,file_sha256={k:sha(v) for k,v in files.items()})
    validate_payload(p); return p


def _save(r):
    raw = json.dumps(r,sort_keys=True,indent=2,allow_nan=False).encode()
    if len(raw)>MAX_BYTES: raise RuntimeError('Router checkpoint exceeds1MiB')
    tmp=RESULT.with_suffix('.tmp'); tmp.write_bytes(raw); tmp.replace(RESULT)


def _bits(torch, a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _range(t): return (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
def _disjoint(tensors):
    spans=[_range(t) for t in tensors if t.numel()]
    return all(a+n<=b or b+m<=a for i,(a,n) in enumerate(spans) for b,m in spans[i+1:])


def make_inputs(torch, d):
    gen=torch.Generator(device='cuda'); gen.manual_seed(d['seed']); t=d['t']; rank=d['rank']; pattern=d['pattern']
    x=torch.randn((t,256),device='cuda',dtype=torch.float32,generator=gen)
    bias=torch.randn((256,),device='cuda',dtype=torch.float32,generator=gen)*.05
    mapping=torch.full((256,),-1,device='cuda',dtype=torch.int32)
    mapping[rank*32:(rank+1)*32]=torch.arange(32,device='cuda',dtype=torch.int32)
    if pattern=='ties': x.zero_(); bias.zero_()
    elif pattern=='extreme':
        x.copy_(torch.where(torch.arange(256,device='cuda')[None,:]%2==0,80.,-80.).expand(t,256)); bias.zero_()
    elif pattern in ('local','remote','mixed'):
        bias.fill_(-10.)
        selected=([rank*32+i for i in range(8)] if pattern=='local' else
                  [((rank+1)%8)*32+i for i in range(8)] if pattern=='remote' else
                  [rank*32+i for i in range(4)]+[((rank+1)%8)*32+i for i in range(4)])
        bias[torch.tensor(selected,device='cuda')]=torch.arange(8,0,-1,device='cuda',dtype=torch.float32)+10.
    return x,bias,mapping


def chain(torch, module, role, args):
    x,bias,mapping=args; t=x.shape[0]
    extra={}; packed=None; local=None
    if role=='baseline': packed=torch.empty((t,8),device=x.device,dtype=torch.int32)
    else:
        local=torch.empty((t,8),device=x.device,dtype=torch.int32)
        extra=dict(marlin_expert_map=mapping,marlin_topk_ids=local)
    scores,ids,gotpacked=module.fused_muse_spark_routing(x,bias,8,1.0,packed_topk_ids=packed,**ROUTE_ARGS,**extra)
    if role=='baseline': local=mapping[ids].contiguous()
    elif gotpacked is not None: raise RuntimeError('Candidate unexpectedly generated packed output')
    return scores,ids,local,gotpacked


def compare(torch, old, new, args):
    checks={'score_bits':_bits(torch,new[0],old[0]), 'global_ids':_bits(torch,new[1],old[1]),
            'local_ids':_bits(torch,new[2],old[2]), 'inputs_outputs_disjoint':_disjoint([*args,*old[:3],old[3],*new[:3]])}
    expected=(old[1]<<16) | (old[0].to(torch.bfloat16).view(torch.int16).to(torch.int32)&65535)
    checks['parent_packed_bits']=_bits(torch,old[3],expected)
    checks['pass']=all(checks.values()); return checks


def correctness(torch, modules, desc):
    args=make_inputs(torch,desc); saved=[t.clone() for t in args]
    old=chain(torch,modules['baseline'],'baseline',args); new=chain(torch,modules['candidate'],'candidate',args)
    torch.cuda.synchronize(); checks=compare(torch,old,new,args)
    checks['inputs_unchanged']=all(_bits(torch,a,b) for a,b in zip(args,saved)); checks['pass'] &= checks['inputs_unchanged']
    if not checks['pass']: raise RuntimeError('Router exactness/ownership failure: '+canonical({'case':desc,'checks':checks}))
    return {'descriptor':desc,**checks}


def padding(torch, modules, rank):
    args=make_inputs(torch,dict(t=8,rank=rank,pattern='mixed',seed=SEED+10000+rank)); x,bias,mapping=args
    valid=torch.tensor([3],device='cuda',dtype=torch.int32); outputs={}
    for role in ('baseline','candidate'):
        score=torch.full((8,8),17.,device='cuda'); ids=torch.full((8,8),-777,device='cuda',dtype=torch.int32)
        local=torch.full_like(ids,-777); extra={} if role=='baseline' else dict(marlin_expert_map=mapping,marlin_topk_ids=local)
        modules[role].fused_muse_spark_routing(x,bias,8,1.,topk_weights=score,topk_ids=ids,num_valid_tokens=valid,**ROUTE_ARGS,**extra)
        outputs[role]=(score,ids,local)
    torch.cuda.synchronize(); old=outputs['baseline']; new=outputs['candidate']
    checks={'active_scores_exact':_bits(torch,old[0][:3],new[0][:3]),'global_ids_exact':_bits(torch,old[1],new[1]),
            'active_local_exact':_bits(torch,mapping[old[1][:3]],new[2][:3]),
            'padded_scores_untouched':bool(torch.all(old[0][3:]==17.) and torch.all(new[0][3:]==17.)),
            'padded_global_minus1':bool(torch.all(new[1][3:]==-1)), 'padded_local_minus1':bool(torch.all(new[2][3:]==-1))}
    if not all(checks.values()): raise RuntimeError('Padded routing contract failure')
    return {'rank':rank,**checks,'pass':True,'old_map_minus1_is_not_sentinel_oracle':True,'production_fast_path_excludes_padding':True}


def legacy_optional(torch, modules, rank):
    """No dual-ID arguments: actual original map/packed contract stays available."""
    x,bias,mapping=make_inputs(torch,dict(t=3,rank=rank,pattern='mixed',seed=SEED+11000+rank))
    outputs=[]
    for role in ('baseline','candidate'):
        packed=torch.empty((3,8),device='cuda',dtype=torch.int32)
        outputs.append(modules[role].fused_muse_spark_routing(x,bias,8,1.,local_expert_mapping=mapping,packed_topk_ids=packed,**ROUTE_ARGS))
    torch.cuda.synchronize()
    checks={name:_bits(torch,a,b) for name,a,b in zip(('score_bits','mapped_ids','packed_bits'),*outputs)}
    if not all(checks.values()):raise RuntimeError('Optional dual-ID absence changed legacy routing')
    return {'rank':rank,**checks,'pass':True,'dual_output_requested':False}


def _profile(torch, fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        out=fn(); torch.cuda.synchronize()
    events=[]
    for e in p.events():
        if str(e.device_type).endswith('CUDA'):
            duration=float(e.time_range.elapsed_us())
            if duration>0: events.append({'name':e.name,'duration_us':duration,'device_type':'CUDA'})
    return out,events


def activation(torch, modules,t):
    args=make_inputs(torch,dict(t=t,rank=0,pattern='mixed',seed=SEED+20000+t)); rows={}
    for role in ('baseline','candidate'):
        _,events=_profile(torch,lambda:chain(torch,modules[role],role,args)); rows[role]=events
    old=rows['baseline']; new=rows['candidate']
    passed=(len(old)==2 and len(new)==1 and sum('routing_kernel' in e['name'] for e in old)==1 and
            'routing_kernel' in new[0]['name'] and sum('index' in e['name'].lower() for e in old)==1)
    if not passed: raise RuntimeError('Actual CUDA launch-removal evidence missing: '+canonical(rows))
    return {'t':t,'pass':True,'events':rows,'removed_index_kernels':1,'added_kernels':0}


def stream_check(torch, modules):
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]; pending=[]
    for i in range(8):
        args=make_inputs(torch,dict(t=(3,31)[i%2],rank=i,pattern='mixed',seed=SEED+30000+i))
        old=chain(torch,modules['baseline'],'baseline',args); saved=[x.clone() for x in args]
        stream=streams[i%2]; stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): new=chain(torch,modules['candidate'],'candidate',args)
        pending.append((args,saved,old,new))
    for s in streams:s.synchronize()
    checks=[]; alloutputs=[]
    for i,(args,saved,old,new) in enumerate(pending):
        check=compare(torch,old,new,args); check['inputs_unchanged']=all(_bits(torch,a,b) for a,b in zip(args,saved)); check['pass'] &= check['inputs_unchanged']
        checks.append({'call':i,**check}); alloutputs += list(new[:3])
    if not all(c['pass'] for c in checks) or not _disjoint(alloutputs): raise RuntimeError('Concurrent stream ownership/exactness failed')
    return {'pass':True,'streams':2,'calls':8,'checks':checks,'distinct_live_outputs':True,'overlap_measured':False}


def graph_check(torch, modules):
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]; graphs=[]; args=[]; outputs=[]
    for i in range(2):
        a=make_inputs(torch,dict(t=16,rank=i,pattern='mixed',seed=SEED+40000+i)); args.append(a)
        streams[i].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[i]):
            for _ in range(3): chain(torch,modules['candidate'],'candidate',a)
        streams[i].synchronize(); g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=streams[i]): out=chain(torch,modules['candidate'],'candidate',a)
        graphs.append(g); outputs.append(out)
    if not _disjoint([v for a in args for v in a]+[v for out in outputs for v in out[:3]]): raise RuntimeError('Cross-graph alias')
    checks=[]; changed=[]; previous=[None,None]
    for replay in range(4):
        pending=[]
        for i in range(2):
            fresh=make_inputs(torch,dict(t=16,rank=(i+replay)%8,pattern='random',seed=SEED+41000+2*replay+i))
            expected=chain(torch,modules['baseline'],'baseline',fresh)
            streams[i].wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(streams[i]):
                for dst,src in zip(args[i],fresh):dst.copy_(src)
                graphs[i].replay()
                snapshot=tuple(v.clone() if v is not None else None for v in outputs[i])
            pending.append((fresh,expected,snapshot))
        for s in streams:s.synchronize()
        for i,(fresh,expected,snapshot) in enumerate(pending):
            check=compare(torch,expected,snapshot,fresh)
            check['inputs_unchanged']=all(_bits(torch,a,b) for a,b in zip(args[i],fresh))
            check['snapshot_matches_live']=all(_bits(torch,a,b) for a,b in zip(outputs[i][:3],snapshot[:3]))
            check['pass'] &= check['inputs_unchanged'] and check['snapshot_matches_live']
            if not check['pass']:raise RuntimeError('Fresh graph output mismatch')
            if previous[i] is not None:changed.append(not _bits(torch,previous[i],snapshot[0]))
            previous[i]=snapshot[0]; checks.append({'graph':i,'replay':replay,**check})
    if not all(changed):raise RuntimeError('Fresh graph inputs did not change outputs')
    return {'pass':True,'graphs':2,'fresh_replays':8,'checks':checks,'cross_graph_ranges_disjoint':True,
            'fresh_outputs_changed':True,'overlap_measured':False,
            'output_ranges':[[list(_range(v)) for v in out[:3]] for out in outputs]}


def timing(torch, modules,t):
    args=make_inputs(torch,dict(t=t,rank=0,pattern='mixed',seed=SEED+50000+t)); graphs={}; outputs={}
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    for role in ('baseline','candidate'):
        with torch.cuda.stream(stream):
            for _ in range(5):chain(torch,modules[role],role,args)
        stream.synchronize(); graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            for _ in range(REPEATS):out=chain(torch,modules[role],role,args)
        graphs[role]=graph; outputs[role]=out
    for _ in range(5):
        for role in ('baseline','candidate'):graphs[role].replay()
    torch.cuda.synchronize(); pairs=[]
    for i in range(ROUNDS):
        fresh=make_inputs(torch,dict(t=t,rank=0,pattern='mixed',seed=SEED+51000+t*100+i))
        for dst,src in zip(args,fresh):dst.copy_(src)
        pair={'round':i,'order':['baseline','candidate'] if i%2==0 else ['candidate','baseline']}
        for role in pair['order']:
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();graphs[role].replay();end.record();end.synchronize()
            pair[role+'_us']=begin.elapsed_time(end)*1000/REPEATS
        if not compare(torch,outputs['baseline'],outputs['candidate'],args)['pass']:raise RuntimeError('Timed graph output mismatch')
        pair['latency_ratio']=pair['candidate_us']/pair['baseline_us'];pairs.append(pair)
    med={role:statistics.median(p[role+'_us'] for p in pairs) for role in ('baseline','candidate')}
    return {'t':t,'pairs':pairs,'baseline_us':med['baseline'],'candidate_us':med['candidate'],
            'latency_ratio':med['candidate']/med['baseline'],'exact_outputs':True}


def _verify_dependencies(p):
    spec=importlib.util.find_spec('sglang'); package=Path(next(iter(spec.submodule_search_locations))).resolve()
    pins=json.loads(p['files']['jit_dependencies.json'])
    observed={rel:sha((package/rel).read_bytes()) for rel in pins}
    if observed!=pins:raise RuntimeError('Installed JIT dependency mismatch: '+','.join(k for k in pins if observed[k]!=pins[k]))
    return {'count':len(pins),'manifest_sha256':sha(canonical(observed)),'package':str(package),'verified':True}


def _collector(module_args, header):
    from sglang.kernels.jit.utils.compile import cache
    name='sgl_kernel_jit_'+'_'.join(module_args)
    matches=list(cache.cache_root().glob('*/'+name+'/build-*/deps-*/sgl_deps.json'))
    if len(matches)!=1:raise RuntimeError('Expected exactly one owned JIT build leaf')
    leaf=matches[0].parent; raw=matches[0].read_bytes()
    if len(raw)>MAX_BYTES:raise RuntimeError('Dependency manifest unexpectedly large')
    deps=json.loads(raw); owned=[d for d in deps if d[:2]==['abs',str(header)]]
    if owned!=[['abs',str(header),sha(header.read_bytes())]]:raise RuntimeError('Compiled dependency lacks owned header')
    common=[d for d in deps if d not in owned]
    artifacts={}
    for filename in (name+'.so','cuda.cu','build.ninja','sgl_deps.json'):
        path=leaf/filename
        if not path.is_file() or path.stat().st_size>128*1024*1024:raise RuntimeError('Missing/oversized owned artifact')
        artifacts[filename]={'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())}
    generated=(leaf/'cuda.cu').read_text()
    if str(header) not in generated or 'fused_muse_spark_routing' not in generated:raise RuntimeError('Generated source lacks owned kernel')
    return {'leaf':str(leaf),'artifacts':artifacts,'dependency_count':len(deps),'common_dependencies_sha256':sha(canonical(sorted(common))),
            'owned_header_sha256':sha(header.read_bytes()),'common_dependency_entries':common}


def validate_result(r,p):
    validate_payload(p)
    for key,value in {'file_sha256':p['file_sha256'],'image':IMAGE,'case_manifest_sha256':manifest(),'policy':POLICY,
                      'component_only':True,'full_serving_validated':False,'official':False}.items():
        if r.get(key)!=value:raise ValueError('Router result provenance/scope mismatch: '+key)
    if r.get('status') not in ('running','failed','incomplete','screen_complete'):raise ValueError('Invalid component status')
    if r['status']!='screen_complete':return
    if r.get('device_count')!=1 or r.get('capability')!=[9,0] or 'H200' not in r.get('device',''):
        raise ValueError('Expected exactly one H200')
    rows=r.get('cases',[])
    if len(rows)!=len(descriptors()) or [x.get('descriptor') for x in rows]!=descriptors():raise ValueError('Missing exact router coverage')
    for row in rows:
        if any(row.get(k) is not True for k in ('pass','score_bits','global_ids','local_ids','inputs_outputs_disjoint','parent_packed_bits','inputs_unchanged')):raise ValueError('Failed router case')
    def true_fields(row,names):
        if any(row.get(k) is not True for k in names):raise ValueError('Missing/failed evidence: '+','.join(names))
    pad=r.get('padding',[])
    if [x.get('rank') for x in pad]!=list(range(8)):raise ValueError('Missing padding evidence')
    for row in pad:true_fields(row,('pass','active_scores_exact','global_ids_exact','active_local_exact','padded_scores_untouched',
                                    'padded_global_minus1','padded_local_minus1','old_map_minus1_is_not_sentinel_oracle','production_fast_path_excludes_padding'))
    fallback=r.get('legacy_optional',[])
    if [x.get('rank') for x in fallback]!=list(range(8)):raise ValueError('Missing optional-output fallback evidence')
    for row in fallback:
        true_fields(row,('pass','score_bits','mapped_ids','packed_bits'))
        if row.get('dual_output_requested') is not False:raise ValueError('Legacy path requested dual outputs')
    for name,count in (('two_streams',8),('two_graphs',8)):
        row=r.get(name,{})
        if row.get('pass') is not True or len(row.get('checks',[]))!=count or any(x.get('pass') is not True for x in row['checks']):raise ValueError('Missing graph/stream evidence')
        if row.get('overlap_measured') is not False:raise ValueError('Unmeasured overlap claimed')
        for check in row['checks']:true_fields(check,('pass','score_bits','global_ids','local_ids','inputs_outputs_disjoint','parent_packed_bits'))
    streams=r['two_streams'];graphs=r['two_graphs']
    if streams.get('streams')!=2 or streams.get('calls')!=8 or [x.get('call') for x in streams['checks']]!=list(range(8)):
        raise ValueError('Stream call identities missing')
    true_fields(streams,('distinct_live_outputs',))
    for check in streams['checks']:true_fields(check,('inputs_unchanged',))
    if graphs.get('graphs')!=2 or graphs.get('fresh_replays')!=8 or [(x.get('graph'),x.get('replay')) for x in graphs['checks']]!=[(i,j) for j in range(4) for i in range(2)]:
        raise ValueError('Graph replay identities missing')
    true_fields(graphs,('cross_graph_ranges_disjoint','fresh_outputs_changed'))
    for check in graphs['checks']:true_fields(check,('inputs_unchanged','snapshot_matches_live'))
    ranges=graphs.get('output_ranges',[])
    if len(ranges)!=2 or any(len(group)!=3 for group in ranges):raise ValueError('Missing output ownership ranges')
    flat=[span for group in ranges for span in group]
    if any(len(v)!=2 or any(type(x) is not int or x<=0 for x in v) for v in flat):raise ValueError('Malformed output ownership range')
    if any(not(a+n<=b or b+m<=a) for i,(a,n) in enumerate(flat) for b,m in flat[i+1:]):raise ValueError('Graph outputs alias')
    activ=r.get('activation',[])
    if [x.get('t') for x in activ]!=[16,256] or any(x.get('pass') is not True or x.get('removed_index_kernels')!=1 or x.get('added_kernels')!=0 for x in activ):raise ValueError('Missing launch-removal evidence')
    for row in activ:
        old=row['events']['baseline'];new=row['events']['candidate']
        if (len(old)!=2 or len(new)!=1 or 'routing_kernel' not in new[0]['name'] or
            sum('routing_kernel' in e['name'] for e in old)!=1 or sum('index' in e['name'].lower() for e in old)!=1 or
            any(e.get('device_type')!='CUDA' or type(e.get('duration_us')) not in (int,float) or not math.isfinite(e['duration_us']) or e['duration_us']<=0 for e in old+new)):
            raise ValueError('Invalid actual CUDA events')
    deps=r.get('dependencies',{});expected_deps=json.loads(p['files']['jit_dependencies.json'])
    if (deps.get('verified') is not True or deps.get('count')!=len(expected_deps) or
            deps.get('manifest_sha256')!=sha(canonical(expected_deps)) or r.get('common_dependencies_equal') is not True):raise ValueError('Unverified compiled dependencies')
    digest=lambda s:isinstance(s,str) and re.fullmatch('[0-9a-f]{64}',s) is not None
    builds=r.get('builds',{})
    if set(builds)!= {'baseline','candidate'}:raise ValueError('Missing both owned builds')
    for role in ('baseline','candidate'):
        evidence=builds[role];source=p['files'][role+'_routing.py'];header=p['files'][role+'_routing.cuh']
        prefix=f'okc_router_v24_{role}_{sha(source+header)[:16]}'
        args=[prefix,'8','8','0','1','true'];name='sgl_kernel_jit_'+'_'.join(args)
        expected_call={'module_args':args,'header':str(FOLDER/(role+'_routing.cuh')),'header_sha256':sha(header)}
        expected={'source_sha256':sha(source),'transformed_sha256':sha(adapt(source)), 'adapter_sha256':adapter_sha(),
                  'reverse_restoration':True,'private_module_name':'okc_private_router_'+role,'calls':[expected_call]}
        if any(evidence.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong private wrapper/build invocation')
        build=evidence.get('build',{});artifacts=build.get('artifacts',{})
        if (build.get('owned_header_sha256')!=sha(header) or type(build.get('dependency_count')) is not int or
                build['dependency_count']<2 or not digest(build.get('common_dependencies_sha256')) or
                set(artifacts)!= {name+'.so','cuda.cu','build.ninja','sgl_deps.json'}):raise ValueError('Missing owned build provenance')
        if any(not digest(row.get('sha256')) or type(row.get('bytes')) is not int or not 0<row['bytes']<=128*1024*1024 for row in artifacts.values()):
            raise ValueError('Malformed owned artifact')
    if (builds['baseline']['build']['common_dependencies_sha256']!=builds['candidate']['build']['common_dependencies_sha256'] or
            builds['baseline']['build']['dependency_count']!=builds['candidate']['build']['dependency_count']):raise ValueError('Common dependency fingerprints differ')
    timings=r.get('timings',[])
    if [x.get('t') for x in timings]!=list(TIMED):raise ValueError('Missing timed production case')
    ratios=[]
    for row in timings:
        pairs=row.get('pairs',[])
        if len(pairs)!=ROUNDS or row.get('exact_outputs') is not True:raise ValueError('Missing timed checks')
        for i,pair in enumerate(pairs):
            if pair.get('round')!=i or pair.get('order')!=(['baseline','candidate'] if i%2==0 else ['candidate','baseline']):raise ValueError('Pair alternation changed')
            for role in ('baseline','candidate'):
                value=pair.get(role+'_us')
                if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Nonfinite timing')
            if pair.get('latency_ratio')!=pair['candidate_us']/pair['baseline_us']:raise ValueError('Wrong paired ratio')
        old=statistics.median(x['baseline_us'] for x in pairs);new=statistics.median(x['candidate_us'] for x in pairs)
        if (row.get('baseline_us'),row.get('candidate_us'),row.get('latency_ratio'))!=(old,new,new/old):raise ValueError('Wrong case medians')
        ratios.append(new/old)
    gm=math.exp(sum(map(math.log,ratios))/len(ratios)); passed=gm<=.95 and max(ratios)<=1.02
    if r.get('latency_geomean')!=gm or r.get('performance_gate_passed') is not passed:raise ValueError('Wrong prespecified gate')


def run(payload):
    validate_payload(payload); start=time.monotonic()
    r=dict(status='running',stage='staging',file_sha256=payload['file_sha256'],image=IMAGE,case_manifest_sha256=manifest(),
           policy=POLICY,component_only=True,full_serving_validated=False,official=False,cases=[],padding=[],legacy_optional=[],activation=[],timings=[],builds={})
    def checkpoint(stage):
        r['stage']=stage;r['elapsed_seconds']=time.monotonic()-start;_save(r)
        if r['elapsed_seconds']>540:raise TimeoutError('Worker cooperative budget exhausted')
    try:
        FOLDER.mkdir(parents=True,exist_ok=True)
        for name,source in payload['files'].items():
            path=FOLDER/name;path.write_text(source)
            if sha(path.read_bytes())!=payload['file_sha256'][name]:raise RuntimeError('Staged source mismatch')
        checkpoint('dependency_preflight');r['dependencies']=_verify_dependencies(payload)
        import torch
        from sglang.kernels.jit import utils
        if (not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(9,0) or
                torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name()):raise RuntimeError('Expected exactly one H200')
        r['device']=torch.cuda.get_device_name();r['device_count']=torch.cuda.device_count();r['capability']=list(torch.cuda.get_device_capability());r['torch_version']=torch.__version__
        modules={}
        for role in ('baseline','candidate'):
            r['builds'][role]={};checkpoint('compile_'+role)
            modules[role]=private_module(payload['files'][role+'_routing.py'],role,FOLDER/(role+'_routing.cuh'),utils,r['builds'][role],_collector)
            modules[role]._jit_module(8,8,0,1)
        old=r['builds']['baseline']['build'];new=r['builds']['candidate']['build']
        r['common_dependencies_equal']=old['common_dependency_entries']==new['common_dependency_entries']
        # Stable order is normalized explicitly; manifests can enumerate dependencies in different order.
        r['common_dependencies_equal']=sorted(old['common_dependency_entries'])==sorted(new['common_dependency_entries'])
        old.pop('common_dependency_entries');new.pop('common_dependency_entries')
        if not r['common_dependencies_equal']:raise RuntimeError('Common compiled dependencies differ')
        checkpoint('correctness')
        for i,d in enumerate(descriptors()):
            r['cases'].append(correctness(torch,modules,d))
            if i%48==47:checkpoint('correctness')
        for rank in range(8):r['padding'].append(padding(torch,modules,rank))
        for rank in range(8):r['legacy_optional'].append(legacy_optional(torch,modules,rank))
        checkpoint('activation')
        for t in (16,256):r['activation'].append(activation(torch,modules,t))
        checkpoint('streams');r['two_streams']=stream_check(torch,modules)
        checkpoint('graphs');r['two_graphs']=graph_check(torch,modules)
        checkpoint('timing')
        for t in TIMED:r['timings'].append(timing(torch,modules,t));checkpoint('timing')
        ratios=[x['latency_ratio'] for x in r['timings']]
        r['latency_geomean']=math.exp(sum(map(math.log,ratios))/len(ratios))
        r['performance_gate_passed']=r['latency_geomean']<=.95 and max(ratios)<=1.02
        r['status']='screen_complete';validate_result(r,payload);checkpoint('complete')
        print(canonical({'status':r['status'],'latency_geomean':r['latency_geomean'],'performance_gate_passed':r['performance_gate_passed']}),flush=True)
        return r
    except BaseException as exc:
        r['status']='incomplete' if isinstance(exc,TimeoutError) else 'failed';r['error']=str(exc)[:6000];r['traceback']=traceback.format_exc()[-12000:]
        r['elapsed_seconds']=time.monotonic()-start;_save(r)
        print(canonical({'status':r['status'],'stage':r['stage'],'error':r['error']}),flush=True)
        raise
