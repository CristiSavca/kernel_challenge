"""CPU-only contract tests; synthetic result fixtures are not GPU evidence."""
import ast
import builtins
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_v27_worker',ROOT/'tools/serving_native_v27_worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def checked(**extra):
    return dict({'pass':True,'kind':'relative','reason':'SYNTHETIC CPU READER FIXTURE',
                 'relative_l2':.001,'worst_row':.002},**extra)


def complete(payload,ratio=.9):
    """Synthetic complete schema for reader/controller tests, never device evidence."""
    r=dict(status='validation_complete',official=False,full_engine_validated=False,performance_promotion=False,
           image=w.IMAGE,file_sha256=copy.deepcopy(payload['file_sha256']),case_manifest_sha256=w.manifest_sha(),
           policy=copy.deepcopy(w.POLICY),dependencies=dict(w.DEPENDENCIES),stage='synthetic_complete',elapsed_seconds=1.,
           environment={'device':'NVIDIA H200','device_count':1,'capability':[9,0]},
           optional_lut={'status':'not_attempted_scope','reason':'Fixed PRMT-only run; no LUT GPU coverage claimed.'},
           baseline_stability={'t':32,'rank':3,'pattern':'mixed','exact':True,'before_any_candidate_forward':True},
           installation={'actual_nonzero_rank':3,'registered_bank_only':True,'weight_bytes':905970816,
                         'workspace_instances':1,'prmt':True,'independent_raw_storages':True},
           private_imports={'reverse_restoration':True,'transformed_sha256':w.sha(w.adapt(payload['files']['native_v27.py'])),
                            'installed_modules_mutated':False},extraction=w.extraction_evidence(payload),
           immutable_bank_views={'rank_metadata_rebindings':list(range(8)),'actual_installations':1,
                                 'unique_native_banks':1,'all_buffers_share_storage':True},
           cases=[{'case':c,'comparison':checked(),'baseline_after_exact':True,'inputs_unchanged':True} for c in w.cases()],
           serial_layers=checked(one_workspace=True),
           streams={'pass':True,'comparisons':[checked(),checked()],'inputs_unchanged':True,
                    'distinct_workspaces':True,'workspace_ptrs':[[100,200,300,400,500],[110,210,310,410,510]],'overlap_measured':False},
           graphs={'pass':True,'checks':[checked(graph=i,replay=j,inputs_unchanged=True,fresh_inputs_changed=True)
                                        for j in range(3) for i in range(2)],
                   'expected_checks':6,'distinct_workspaces':True,'output_ptrs':[1000,2000]},timings=[])
    for t in w.TIMED:
        r['timings'].append({'t':t,'baseline_us':10.,'candidate_us':10.*ratio,
                            'samples_us':[[10.]*5,[10.*ratio]*5],'comparison':checked()})
    r['timing_gate']=w.timing_gate(r['timings'])
    return r


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.payload=w.build_payload(ROOT)

    def test_payload_sources_and_fixed_domain(self):
        w.validate_payload(self.payload)
        self.assertEqual(len(w.cases()),38)
        self.assertEqual({x['rank'] for x in w.cases()},set(range(8)))
        self.assertTrue({0,1,31,32,383,384,385,769,16384}<={x['t'] for x in w.cases()})
        self.assertEqual(self.payload['validation_budget_seconds'],240)
        self.assertEqual(self.payload['timeout_seconds'],300)
        self.assertTrue(all('/' not in name for name in self.payload['files']))
        self.assertFalse(w.POLICY['performance_promotion'])
        self.assertIn('Apache License',self.payload['files']['SGLANG_LICENSE.txt'])

    def test_payload_rejects_pin_policy_and_flat_name_mutants(self):
        for edit in (
            lambda p:p.update(validation_budget_seconds=241),
            lambda p:p.update(timeout_seconds=900),
            lambda p:p.update(gpu=None),
            lambda p:p['policy'].update(whole_relative_l2=.02),
            lambda p:p['files'].__setitem__('stock_experts.py',p['files']['stock_experts.py']+'\n'),
            lambda p:p['files'].__setitem__('../unexpected','bad'),
        ):
            mutant=copy.deepcopy(self.payload);edit(mutant)
            with self.assertRaises(ValueError):w.validate_payload(mutant)

    def test_exact_private_substitutions_and_real_policy_logic(self):
        original=self.payload['files']['native_v27.py'];adapted=w.adapt(original)
        restored=adapted.replace(w.FINALIZER_REPLACEMENT,w.FINALIZER_IMPORT).replace(w.POLICY_REPLACEMENT,w.POLICY_IMPORT)
        self.assertEqual(restored,original)
        scope={'_qualified_fixture_getters':w.qualified_getters()}
        exec(compile(w.extract(adapted,('_serial_policy',)),'actual_policy','exec'),scope)
        self.assertIs(scope['_serial_policy'](),True)
        scope['_qualified_fixture_getters'][1]().overlap.enable_two_batch_overlap=True
        self.assertIs(scope['_serial_policy'](),False)
        scope['_qualified_fixture_getters'][1]().overlap.enable_two_batch_overlap=False
        scope['_qualified_fixture_getters'][1]().graph.cuda_graph_config.decode.backend='tc_piecewise'
        self.assertIs(scope['_serial_policy'](),False)
        with self.assertRaises(ValueError):w.adapt(original+'\n'+w.POLICY_IMPORT)

    def test_actual_stock_ast_execution_without_module_registration(self):
        # Only five functions and three methods enter the private namespace.
        # No top-level stock custom-op registration or installed-source mutation.
        modules={}
        def module(name,**attrs):
            obj=types.ModuleType(name);obj.__dict__.update(attrs);modules[name]=obj;return obj
        sentinel=lambda *a,**kw:None
        module('sglang.srt.layers.quantization.marlin_utils',marlin_make_workspace=sentinel,
               marlin_permute_scales=sentinel,marlin_permute_bias=sentinel)
        module('sglang.kernels.ops.quantization.gptq_marlin_repack',gptq_marlin_repack=sentinel)
        fake=types.SimpleNamespace(nn=types.SimpleNamespace(Module=object),Tensor=object,no_grad=lambda:lambda f:f)
        with patch.dict(sys.modules,modules):
            Layer,prepare,setter,evidence=w.make_stock_api(fake,self.payload,sentinel)
        self.assertEqual(evidence,w.extraction_evidence(self.payload))
        self.assertEqual(prepare.__name__,'prepare_moe_nvfp4_layer_for_marlin')
        self.assertEqual(Layer._forward_marlin.__name__,'_forward_marlin')
        self.assertEqual(setter.__name__,'set_post_expert_norm')
        self.assertNotIn('register_custom_op',prepare.__globals__)
        for name in ('_marlin_expert_outputs','_forward_marlin','set_post_expert_norm'):
            fn=getattr(Layer,name)
            self.assertIn('stock_experts.py',fn.__code__.co_filename)
        self.assertIn('candidate_experts.py',setter.__code__.co_filename)

    def test_isolated_actual_flat_payload_import_no_repository_path(self):
        with tempfile.TemporaryDirectory(prefix='okc-v27-import-') as folder:
            dest=Path(folder)
            for name,raw in self.payload['files'].items():(dest/name).write_text(raw)
            code="import importlib.util,sys; s=importlib.util.spec_from_file_location('w','native_v27_worker.py'); w=importlib.util.module_from_spec(s); s.loader.exec_module(w); assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules; assert callable(w.run)"
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            proc=subprocess.run([sys.executable,'-I','-c',code],cwd=dest,env=env,capture_output=True,text=True,timeout=20)
            self.assertEqual(proc.returncode,0,proc.stderr)

    def test_complete_and_descriptive_timing_miss(self):
        r=complete(self.payload)
        self.assertTrue(w.validate_result(r,self.payload))
        self.assertFalse(w.validate_result(complete(self.payload,1.04),self.payload))
        self.assertFalse(r['timing_gate']['performance_promotion'])

    def test_trusted_partial_prefixes_remain_incomplete(self):
        for status,count in [('running',0),('validation_incomplete',3),('validation_failed',8)]:
            r=complete(self.payload);r['status']=status;r['cases']=r['cases'][:count];r['timings']=[]
            for key in ('timing_gate','graphs','streams','serial_layers','installation','immutable_bank_views'):
                r.pop(key,None)
            if status=='validation_failed':r['cases'][-1]['comparison'].update({'pass':False,'relative_l2':.1,'worst_row':.2})
            self.assertFalse(w.validate_result(r,self.payload))
        r=complete(self.payload);r['status']='running';r['graphs']['checks']=r['graphs']['checks'][:2]
        self.assertFalse(w.validate_result(r,self.payload))

    def test_partial_failed_install_and_stock_instability_are_retained(self):
        r=complete(self.payload);r['status']='validation_failed'
        r['installation']['registered_bank_only']=False
        r['baseline_stability']['exact']=False
        self.assertFalse(w.validate_result(r,self.payload))

    def test_reject_incomplete_or_conflicting_complete_evidence(self):
        edits=[lambda r:r['cases'].pop(),lambda r:r['cases'].reverse(),
               lambda r:r['graphs']['checks'].pop(),lambda r:r['graphs']['checks'][0].update(replay=2),
               lambda r:r['graphs']['checks'][0].update(fresh_inputs_changed=False),
               lambda r:r['graphs'].update(output_ptrs=[1,1]),lambda r:r['streams'].update(inputs_unchanged=False),
               lambda r:r['streams'].update(workspace_ptrs=[[1,2,3,4,5],[1,6,7,8,9]]),
               lambda r:r['serial_layers'].update(one_workspace=False),
               lambda r:r['immutable_bank_views'].update(unique_native_banks=2),
               lambda r:r['installation'].update(actual_nonzero_rank=0),lambda r:r.update(official=True),
               lambda r:r['dependencies'].update({'bad':'source'}),lambda r:r['extraction'].update(scope='rewritten'),
               lambda r:r['cases'][0]['comparison'].pop('worst_row'),
               lambda r:r['cases'][0]['comparison'].update(relative_l2=float('nan')),
               lambda r:r['cases'][0]['comparison'].update(worst_row=.076),
               lambda r:r['timings'].pop(),lambda r:r['timings'][0]['samples_us'][0].pop(),
               lambda r:r['timings'][0]['samples_us'][0].__setitem__(0,float('inf')),
               lambda r:r['timings'][0].update(candidate_us=1.),
               lambda r:r['timings'][0]['comparison'].update({'pass':False}),
               lambda r:r['timing_gate'].update({'pass':False}),
               lambda r:r['optional_lut'].update(status='passed')]
        for edit in edits:
            r=complete(self.payload);edit(r)
            with self.assertRaises(ValueError):w.validate_result(r,self.payload)

    def test_checkpoint_before_import_failure_preserves_original(self):
        original=builtins.__import__
        def fail(name,*args,**kw):
            if name=='torch':raise RuntimeError('directed missing device dependency')
            return original(name,*args,**kw)
        with tempfile.TemporaryDirectory(prefix='okc-v27-failure-') as folder,patch.object(w,'RESULT',Path(folder)/'result.json'),patch('builtins.__import__',side_effect=fail):
            result=w.run(self.payload)
            self.assertEqual(result['status'],'validation_failed')
            self.assertIn('directed missing device dependency',result['error'])
            self.assertEqual(json.loads(w.RESULT.read_text()),result)
            self.assertFalse(w.validate_result(result,self.payload))

    def test_fixed_deadline_and_checkpoint_bound(self):
        with patch.object(w.time,'monotonic',return_value=239.9):w.check_budget(0)
        with patch.object(w.time,'monotonic',return_value=240.):
            with self.assertRaises(TimeoutError):w.check_budget(0)
        with patch.object(w.time,'monotonic',return_value=235.):
            with self.assertRaises(TimeoutError):w.check_budget(0,5)
        with tempfile.TemporaryDirectory(prefix='okc-v27-bound-') as folder,patch.object(w,'RESULT',Path(folder)/'result.json'):
            w._save({'status':'running'})
            before=w.RESULT.read_bytes()
            with self.assertRaises(RuntimeError):w._save({'oversized':'x'*w.MAX_BYTES})
            self.assertEqual(w.RESULT.read_bytes(),before)


if __name__=='__main__':unittest.main()
