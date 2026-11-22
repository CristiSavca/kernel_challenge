"""Execute the V232 native scheduler/allocator and prepared graph admission on CPU."""
import ast
import dataclasses
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

HERE=Path(__file__).resolve().parent
S=Path(sys.argv[1]).resolve()/'python/sglang/srt'

def load(name, path):
    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def compile_function(function,env):
    exec('from __future__ import annotations\n'+ast.unparse(function),env)

def first(file, name, owner=None):
    top=ast.parse((S/file).read_text()).body
    body=next(n.body for n in top if getattr(n,"name",None)==owner) if owner else top
    return next(n for n in body if getattr(n,'name',None)==name)

def config():
    helpers=load('v232_cap',S/'arg_groups/prefill_buffer_ceiling.py')
    sched=load('v232_prior_sched',HERE.parent/'v160_mixed_dflash_serving/scheduler_admission_cpu.py')
    original=ast.parse(inspect.getsource(sched.initialize)).body[0]
    # The remaining old test checks V160's positional CLI signature and is specific to that release.
    original.body=original.body[:next(i for i,n in enumerate(original.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='arg')]
    compile_function(original,vars(sched));sched.initialize()
    sched.mixed_prefill_graph_fits=helpers.mixed_prefill_graph_fits
    p=load('v232_prior_plan',HERE.parent/'v157_mixed_graph_admission/cpu_check.py');p.load()
    p.mixed_prefill_graph_fits=helpers.mixed_prefill_graph_fits
    compile_function(first('speculative/dflash_mixed_graph_admission.py','plan_mixed_dflash_target'),vars(p))
    return helpers,sched,p

def native_scheduler(f,rows,requests,all_keys,limit=16640,budget=16384,free=None,physical=16640):
    s,p,d=f.create(p=1,t=requests,q=rows)
    s.chunked_prefill_size=16384; s.max_prefill_tokens=limit
    pre=s.tp_worker.model_runner.prefill_cuda_graph_runner
    pre.capture_num_tokens=all_keys;pre.max_num_tokens=all_keys[-1];pre.mixed_dflash_token_capacity=physical
    alloc=p.token_to_kv_pool_allocator
    if free is not None:alloc.free=free
    cost=d.new_tokens_required_next_decode()
    old=alloc.free
    result,running=f.schedule(s,p,d,budget)
    admitted=result is not p
    if admitted:
        assert running.is_empty() and alloc.reservations==1 and alloc.free==old-cost
        assert len(result.reqs)==len(p.reqs)+len(d.reqs) and result.forward_mode==f.ForwardMode.MIXED
    else:assert alloc.reservations==0 and alloc.free==old
    return admitted

def native_plan(f,rows,requests,grid,verify,limit=16640,drop=False,physical=16640):
    mr=f.model((grid,verify));f.SETTINGS.max_prefill_tokens=limit;mr.prefill_cuda_graph_runner.mixed_dflash_token_capacity=physical
    prompt,tail,mixed,reqs=f.views(mr,1,requests,rows)
    if drop:
        delkey=mr.prefill_cuda_graph_runner._shape_key(
            mr.prefill_cuda_graph_runner._pad_to_bucket(rows+requests*8,grid),mixed)
        mr.prefill_cuda_graph_runner.backend._graphs.remove(delkey)
    return f.plan_mixed_dflash_target(mr,prompt,tail,mixed,reqs)

def check():
    h,f,p=config()
    grid=f.GRID['target_prefill_graph_rows'];verify=f.GRID['target_verify_graph_requests']
    keys=sorted(h.mixed_prefill_graph_sizes(16384,16640,8,verify))
    assert keys==[16392,16400,16416,16448,16512,16640]
    all_keys=grid+keys
    cases=[]
    for requests in range(1,32):
        def test(name,q,expected,gr=all_keys,lim=16640,budget=16384,free=None,physical=16640):
            observed=native_scheduler(f,q,requests,gr,lim,budget,free,physical)
            assert observed==expected,(name,requests,q,observed)
            cases.append((name,observed))
        test('full_chunk',16384,True)
        test('at_original_graph',16384,False,grid)
        test('ordinary_token_cap',16384,False,lim=16384)
        test('dynamic_chunk_cap',16384,False,budget=16383)
        test('physical_shortage',16384,False,free=0)
        test('prompt_over_cap',16385,False)
        test('physical_ceiling',16384,False,physical=16384+8*requests-1)
        if 16384+8*requests not in keys:
            test('padded_physical_ceiling',16384,False,physical=16384+8*requests)
        plan=native_plan(p,16384,requests,all_keys,verify); assert plan
        assert plan.prompt_rows==16384 and plan.mixed_rows in keys
        assert 0<=plan.mixed_rows-plan.prompt_rows-plan.verify_rows<=plan.verify_rows
        assert not native_plan(p,16384,requests,all_keys,verify,drop=True)
        assert not native_plan(p,16384,requests,all_keys,verify,limit=16384)
        assert not native_plan(p,16385,requests,all_keys,verify)
        assert not native_plan(p,16384,requests,all_keys,verify,physical=16384+8*requests-1)
    return dict(physical_shape_scheduler_forwards=len(cases),admitted=sum(x[1] for x in cases),
        physical_allocator_declines=len([x for x in cases if x==('physical_shortage',False)]),
        real_prepared_graph_plan_cases=31*5,added_graph_keys=keys,
        note='CPU execution of native scheduler branch, greedy gates, page-one reservation, prepared graph planner, and graph-eligibility/backend-key checks. CUDA body and FA3 timings require hardware.')

if __name__=='__main__':print(json.dumps(check(),indent=2))
