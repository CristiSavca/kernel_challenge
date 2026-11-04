"""Focused no-provider tests of actual wrapper adaptation and fixed screen evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('router_worker',ROOT/'tools/serving_router_local_ids_worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def flags():
    return dict(score_bits=True,global_ids=True,local_ids=True,inputs_outputs_disjoint=True,parent_packed_bits=True,inputs_unchanged=True,pass_=True)
def checked(**extra):
    row=flags();row['pass']=row.pop('pass_');row.update(extra);return row


def complete(payload,ratio=.90):
    """Synthetic complete reader fixture; ratio>1.02 is completed but rejected speed."""
    r=dict(status='screen_complete',file_sha256=copy.deepcopy(payload['file_sha256']),image=w.IMAGE,
           case_manifest_sha256=w.manifest(),policy=copy.deepcopy(w.POLICY),component_only=True,
           full_serving_validated=False,official=False,device='NVIDIA H200',device_count=1,capability=[9,0],
           cases=[checked(descriptor=d) for d in w.descriptors()],padding=[],legacy_optional=[],activation=[],timings=[],builds={},
           common_dependencies_equal=True,dependencies=dict(verified=True,count=len(json.loads(payload['files']['jit_dependencies.json'])),
               manifest_sha256=w.PINS['jit_dependencies.json'],package='/verified/sglang'))
    for rank in range(8):
        r['padding'].append(dict(rank=rank,**{name:True for name in ('pass','active_scores_exact','global_ids_exact','active_local_exact',
                  'padded_scores_untouched','padded_global_minus1','padded_local_minus1','old_map_minus1_is_not_sentinel_oracle','production_fast_path_excludes_padding')}))
        r['legacy_optional'].append(dict(rank=rank,score_bits=True,mapped_ids=True,packed_bits=True,dual_output_requested=False,**{'pass':True}))
    for role in ('baseline','candidate'):
        source=payload['files'][role+'_routing.py'];header=payload['files'][role+'_routing.cuh']
        prefix=f'okc_router_v24_{role}_{w.sha(source+header)[:16]}';args=[prefix,'8','8','0','1','true']
        name='sgl_kernel_jit_'+'_'.join(args)
        r['builds'][role]=dict(source_sha256=w.sha(source),transformed_sha256=w.sha(w.adapt(source)),adapter_sha256=w.adapter_sha(),
            reverse_restoration=True,private_module_name='okc_private_router_'+role,
            calls=[dict(module_args=args,header=str(w.FOLDER/(role+'_routing.cuh')),header_sha256=w.sha(header))],
            build=dict(leaf='/owned/'+name,owned_header_sha256=w.sha(header),dependency_count=27,common_dependencies_sha256='a'*64,
                       artifacts={file:dict(bytes=2048,sha256='b'*64) for file in (name+'.so','cuda.cu','build.ninja','sgl_deps.json')}))
    for t in (16,256):
        event=lambda name:dict(name=name,duration_us=1.,device_type='CUDA')
        r['activation'].append(dict(t=t,removed_index_kernels=1,added_kernels=0,**{'pass':True},
            events={'baseline':[event('routing_kernel<T>'),event('index_elementwise_kernel')],'candidate':[event('routing_kernel<T>')]}))
    r['two_streams']=dict(streams=2,calls=8,checks=[checked(call=i) for i in range(8)],distinct_live_outputs=True,overlap_measured=False,**{'pass':True})
    r['two_graphs']=dict(graphs=2,fresh_replays=8,checks=[checked(graph=i,replay=j,snapshot_matches_live=True) for j in range(4) for i in range(2)],
        cross_graph_ranges_disjoint=True,fresh_outputs_changed=True,overlap_measured=False,**{'pass':True},
        output_ranges=[[[100000+g*10000+i*1000,512] for i in range(3)] for g in range(2)])
    ratios=[]
    for t in w.TIMED:
        pairs=[dict(round=i,order=['baseline','candidate'] if i%2==0 else ['candidate','baseline'],
                    baseline_us=4.,candidate_us=4.*ratio,latency_ratio=ratio) for i in range(w.ROUNDS)]
        r['timings'].append(dict(t=t,pairs=pairs,baseline_us=4.,candidate_us=4.*ratio,latency_ratio=ratio,exact_outputs=True));ratios.append(ratio)
    r['latency_geomean']=w.math.exp(sum(map(w.math.log,ratios))/len(ratios));r['performance_gate_passed']=r['latency_geomean']<=.95 and max(ratios)<=1.02
    return r


class Args(list):
    def __str__(self):return ', '.join(self)


def spy_utilities():
    def memo(fn):
        cache={}
        def wrapper(*args):
            if args not in cache:cache[args]=fn(*args)
            return cache[args]
        return wrapper
    calls=[]
    def load(*args,**kwargs):
        calls.append((args,kwargs));return types.SimpleNamespace(fused_muse_spark_routing=lambda *args:None)
    return types.SimpleNamespace(cache_once=memo,is_arch_support_pdl=lambda:True,load_jit=load,
          make_cpp_args=lambda *a:Args('true' if x is True else 'false' if x is False else str(x) for x in a)),calls


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.payload=w.build_payload(ROOT)

    def test_fixed_source_manifest_and_domains(self):
        w.validate_payload(self.payload)
        self.assertEqual(len(w.descriptors()),432)
        self.assertEqual({d['rank'] for d in w.descriptors()},set(range(8)))
        self.assertEqual({d['t'] for d in w.descriptors()},set(w.TOKENS))
        self.assertEqual(self.payload['policy']['route_arguments'],dict(router_output_multiplier=.17677669529663687,router_score_division_eps=1e-15))
        self.assertEqual(len(json.loads(self.payload['files']['jit_dependencies.json'])),24)
        self.assertNotIn('torch',sys.modules)

    def test_actual_private_wrapper_import_and_exact_loader_contract(self):
        utils,calls=spy_utilities();fake_torch=types.ModuleType('torch');fake_torch.Tensor=object
        with tempfile.TemporaryDirectory(prefix='okc-router-adapter-') as folder,patch.dict(sys.modules,{'torch':fake_torch}):
            modules=[];records=[]
            for role in ('baseline','candidate'):
                header=Path(folder)/(role+'.cuh');header.write_text(self.payload['files'][role+'_routing.cuh']);record={}
                module=w.private_module(self.payload['files'][role+'_routing.py'],role,header,utils,record)
                a=module._jit_module(8,8,0,1);self.assertIs(a,module._jit_module(8,8,0,1));modules.append(module);records.append(record)
            self.assertEqual(len(calls),2);self.assertNotEqual(calls[0][0][0],calls[1][0][0]);self.assertNotEqual(modules[0]._jit_module,modules[1]._jit_module)
            for i,role in enumerate(('baseline','candidate')):
                self.assertEqual(calls[i][0][1:],('8','8','0','1','true'))
                self.assertEqual(calls[i][1],dict(cuda_files=[str((Path(folder)/(role+'.cuh')).resolve())],
                    cuda_wrappers=[('fused_muse_spark_routing','fused_muse_spark_routing<8, 8, 0, 1, true>')]))
                self.assertEqual(w.adapt(self.payload['files'][role+'_routing.py']).replace(w.ADAPTED_IMPORT,w.IMPORT),self.payload['files'][role+'_routing.py'])
                self.assertTrue(records[i]['reverse_restoration'])
            with self.assertRaisesRegex(ValueError,'Unexpected actual'):modules[0]._jit_module(8,8,1,1)
            with self.assertRaisesRegex(ValueError,'Unexpected actual'):modules[1].load_jit('wrong')
            self.assertEqual(len(calls),2)
        self.assertNotIn('sglang.kernels.ops.muse_spark_v12.routing',sys.modules)

    def test_adapter_source_mutations_rejected(self):
        source=self.payload['files']['baseline_routing.py']
        for changed in (source.replace('    load_jit,','    load_jit as bad,'), source+'\n'+w.IMPORT, source.replace(w.IMPORT,'')):
            with self.assertRaises(ValueError):w.adapt(changed)

    def test_real_isolated_six_file_package(self):
        with tempfile.TemporaryDirectory(prefix='okc-router-package-') as folder:
            root=Path(folder)
            for name,source in self.payload['files'].items():(root/name).write_text(source)
            (root/'payload.json').write_text(json.dumps(self.payload))
            code="""import importlib.util,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('worker',p/'router_local_ids.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
x=json.loads((p/'payload.json').read_text());w.validate_payload(x)
for name,digest in x['file_sha256'].items():assert w.sha((p/name).read_bytes())==digest
assert 'torch' not in sys.modules and 'sglang' not in sys.modules
assert not any('HASIDIC_PROJECTS' in x for x in sys.path)
print('isolated-six-file-PASS')
"""
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            cmd=[sys.executable,'-I','-S','-c',code,folder]
            run=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=15,check=True)
            self.assertEqual(run.stdout.strip(),'isolated-six-file-PASS')
            (root/'candidate_routing.cuh').unlink()
            failed=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(failed.returncode,0);self.assertIn('FileNotFoundError',failed.stderr)

    def test_payload_source_and_policy_mutations_rejected(self):
        for name in w.FILES:
            p=copy.deepcopy(self.payload);p['files'][name]+='\n';p['file_sha256'][name]=w.sha(p['files'][name])
            with self.subTest(name=name),self.assertRaises(ValueError):w.validate_payload(p)
        for name,value in [('timeout_seconds',900),('validation_budget_seconds',541),('gpu',None),('image','other'),('seed',0),('case_manifest_sha256','0'*64)]:
            p=copy.deepcopy(self.payload);p[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):w.validate_payload(p)
        p=copy.deepcopy(self.payload);p['policy']['route_arguments']['router_score_division_eps']=0.
        with self.assertRaises(ValueError):w.validate_payload(p)

    def test_installed_dependency_mismatch_fails_before_import_compile(self):
        with tempfile.TemporaryDirectory(prefix='okc-router-deps-') as folder:
            package=Path(folder);(package/'foo.py').write_text('known')
            p={'files':{'jit_dependencies.json':w.canonical({'foo.py':w.sha('known')})}}
            with patch.object(w.importlib.util,'find_spec',return_value=types.SimpleNamespace(submodule_search_locations=[folder])):
                self.assertTrue(w._verify_dependencies(p)['verified'])
                (package/'foo.py').write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'dependency mismatch'):w._verify_dependencies(p)
        self.assertNotIn('sglang.kernels.jit.utils',sys.modules)

    def test_complete_speed_accept_and_reject_are_both_complete(self):
        for ratio,passed in ((.90,True),(1.03,False),(.97,False)):
            result=complete(self.payload,ratio);w.validate_result(result,self.payload)
            self.assertEqual(result['performance_gate_passed'],passed)
        result=complete(self.payload,1.03);result['performance_gate_passed']=True
        with self.assertRaises(ValueError):w.validate_result(result,self.payload)
        result['status']='pass'
        with self.assertRaises(ValueError):w.validate_result(result,self.payload)

    def test_missing_case_padding_fallback_stream_graph_evidence(self):
        mutations=[lambda r:r['cases'].pop(),lambda r:r['cases'][0].update(score_bits=False),
          lambda r:r['padding'][7].update(padded_local_minus1=False),lambda r:r['padding'][0].pop('active_scores_exact'),
          lambda r:r['legacy_optional'][0].update(dual_output_requested=True),lambda r:r['legacy_optional'].pop(),
          lambda r:r['two_streams'].update(calls=7),lambda r:r['two_streams']['checks'][0].update(inputs_unchanged=False),
          lambda r:r['two_graphs'].update(fresh_replays=7),lambda r:r['two_graphs']['checks'][0].update(replay=9),
          lambda r:r['two_graphs']['output_ranges'][1].__setitem__(0,r['two_graphs']['output_ranges'][0][0]),
          lambda r:r['two_graphs'].update(fresh_outputs_changed=False)]
        for mutate in mutations:
            r=complete(self.payload);mutate(r)
            with self.assertRaises(ValueError):w.validate_result(r,self.payload)

    def test_missing_build_or_fake_cuda_activation_rejected(self):
        mutations=[lambda r:r['builds'].clear(),lambda r:r['builds']['candidate'].update(source_sha256='0'*64),
            lambda r:r['builds']['baseline'].update(calls=[]),lambda r:r['builds']['candidate']['build'].update(owned_header_sha256='0'*64),
            lambda r:r['builds']['candidate']['build'].update(common_dependencies_sha256='c'*64),
            lambda r:r['builds']['baseline']['build']['artifacts'].pop('cuda.cu'),lambda r:r['dependencies'].update(count=1),
            lambda r:r['activation'][0]['events']['candidate'][0].update(device_type='CPU'),
            lambda r:r['activation'][0]['events']['candidate'][0].update(duration_us=0.),lambda r:r.update(device='A100'),
            lambda r:r.update(device_count=2)]
        for mutate in mutations:
            r=complete(self.payload);mutate(r)
            with self.assertRaises(ValueError):w.validate_result(r,self.payload)

    def test_timing_recomputed_and_alternation_enforced(self):
        for mutate in (lambda r:r['timings'][0]['pairs'].pop(),lambda r:r['timings'][0]['pairs'][1].update(order=['baseline','candidate']),
                       lambda r:r['timings'][0].update(latency_ratio=.1),lambda r:r['timings'][0]['pairs'][0].update(candidate_us=float('nan')),
                       lambda r:r.update(latency_geomean=.1)):
            r=complete(self.payload);mutate(r)
            with self.assertRaises(ValueError):w.validate_result(r,self.payload)

    def test_profiler_requires_actual_positive_cuda_events(self):
        class Context:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def events(self):
                return [types.SimpleNamespace(name=name,device_type=kind,time_range=types.SimpleNamespace(elapsed_us=lambda n=duration:n))
                        for name,kind,duration in [('routing_kernel','CPU',2.),('routing_kernel','CUDA',1.),('index','CUDA',0.)]]
        torch=types.SimpleNamespace(profiler=types.SimpleNamespace(profile=lambda **kw:Context(),ProfilerActivity=types.SimpleNamespace(CPU=0,CUDA=1)),
                                    cuda=types.SimpleNamespace(synchronize=lambda:None))
        output,events=w._profile(torch,lambda:'actual output')
        self.assertEqual(output,'actual output')
        self.assertEqual(events,[dict(name='routing_kernel',duration_us=1.,device_type='CUDA')])

    def test_owned_compiler_leaf_requires_actual_header_and_complete_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='okc-router-leaf-') as folder:
            root=Path(folder);header=root/'candidate.cuh';header.write_text('owned source')
            args=('owned','8','8','0','1','true');name='sgl_kernel_jit_'+'_'.join(args)
            leaf=root/'sm90a'/name/'build-fixed'/'deps-fixed';leaf.mkdir(parents=True)
            for file in (name+'.so','build.ninja'):(leaf/file).write_text('fixture')
            (leaf/'cuda.cu').write_text('#include "'+str(header)+'"\n// fused_muse_spark_routing')
            entries=[['abs',str(header),w.sha(header.read_bytes())],['kernels','include/sgl_kernel/tensor.h','a'*64]]
            (leaf/'sgl_deps.json').write_text(json.dumps(entries))
            compile_module=types.ModuleType('sglang.kernels.jit.utils.compile');compile_module.cache=types.SimpleNamespace(cache_root=lambda:root)
            with patch.dict(sys.modules,{'sglang.kernels.jit.utils.compile':compile_module}):
                result=w._collector(args,header)
                self.assertEqual(result['owned_header_sha256'],w.sha('owned source'));self.assertEqual(result['dependency_count'],2)
                self.assertEqual(result['common_dependency_entries'],entries[1:])
                (leaf/'sgl_deps.json').write_text(json.dumps(entries[1:]))
                with self.assertRaisesRegex(RuntimeError,'owned header'):w._collector(args,header)
                (leaf/'sgl_deps.json').write_text(json.dumps(entries));(leaf/'cuda.cu').unlink()
                with self.assertRaisesRegex(RuntimeError,'artifact'):w._collector(args,header)

    def test_partial_build_failure_checkpoint_retains_original_error(self):
        with tempfile.TemporaryDirectory(prefix='okc-router-failure-') as folder:
            result=Path(folder)/'result.json'
            with patch.object(w,'RESULT',result),patch.object(w,'FOLDER',Path(folder)/'files'),\
                 patch.object(w,'_verify_dependencies',side_effect=RuntimeError('known dependency fault')):
                with self.assertRaisesRegex(RuntimeError,'known dependency fault'):w.run(self.payload)
            saved=json.loads(result.read_text());w.validate_result(saved,self.payload)
            self.assertEqual(saved['status'],'failed');self.assertEqual(saved['stage'],'dependency_preflight')
            self.assertEqual(saved['error'],'known dependency fault');self.assertIn('RuntimeError',saved['traceback'])
            self.assertEqual(saved['cases'],[])
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
