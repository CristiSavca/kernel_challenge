"""Source-dispatch, full-precision corner cases, and logical traffic of greedy target selection."""
import ast
import copy
import itertools
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PIN = '297e54fd105e974cdf401e5b4d20253a22c16f90'


def method(file, cls, name):
    path = 'python/sglang/srt/' + file + '.py'
    source = subprocess.check_output(['git', '-C', ROOT/'sglang', 'show', PIN + ':' + path], text=True)
    owner = next(n for n in ast.parse(source).body if getattr(n, 'name', '') == cls)
    return next(n for n in owner.body if getattr(n, 'name', '') == name)


def execute(node, env):
    exec('from __future__ import annotations\n' + ast.unparse(ast.Module(body=[node], type_ignores=[])), env)
    return env[node.name]


def dispatch():
    worker = 'speculative/dflash_worker_v2'
    processor = 'layers/logits_processor'
    ctor = ast.unparse(method('managers/scheduler', 'Scheduler', 'init_model_worker'))
    assert ctor.index('self.maybe_init_draft_worker()') < ctor.index('self.init_all_cuda_graphs()')
    cpu_cap = execute(method(processor, 'LogitsProcessor', '_can_gather_softcap_fused'),
                      {'torch':SimpleNamespace(float32='fp32'), '_is_cpu':False, '_is_npu':False})
    proc = SimpleNamespace(do_tensor_parallel_all_gather=True, use_attn_tp_group=False,
                           do_tensor_parallel_all_gather_dp_attn=False, final_logit_softcapping=20.,
                           _can_use_tp_lm_head_all_to_all=lambda *args:False)
    shard = SimpleNamespace(is_cuda=True, dtype='fp32', dim=lambda:2, is_contiguous=lambda:True)
    assert cpu_cap(proc, shard, None, None, None)
    greedy = execute(method(worker, 'DFlashWorkerV2', '_can_skip_verify_logits'), {})
    req = SimpleNamespace(has_grammar=False,return_logprob=False)
    info = SimpleNamespace(is_all_greedy=True,has_custom_logit_processor=False,penalizer_orchestrator=None,
                           sampling_mask_batch_indices=None)
    owner = SimpleNamespace(_selector_sample=None)
    assert greedy(owner,req,info) and greedy(owner,req,None)
    trials = 0
    for obj, field, value in [(owner,'_selector_sample',(1,2)),(req,'has_grammar',True),(req,'return_logprob',True),
                              (info,'is_all_greedy',False),(info,'has_custom_logit_processor',True),
                              (info,'acc_linear_penalties',1),(info,'penalizer_orchestrator',SimpleNamespace(is_required=True)),
                              (info,'grammar_mask',1),(info,'logit_bias',1),(info,'sampling_mask_batch_indices',[0])]:
        ctx = [copy.copy(x) if x is obj else x for x in (owner,req,info)]
        setattr(ctx[[x is obj for x in (owner,req,info)].index(True)],field,value)
        assert not greedy(*ctx)
        trials += 1
    logits = ast.unparse(method(processor, 'LogitsProcessor', '_get_logits'))
    assert logits.index('self._compute_lm_head') < logits.index('logits.mul_') < logits.index('self.defer_verify_gather') < logits.index('self._gather_softcap_fused')
    muse = ast.unparse(method('models/muse_spark','MuseSpark1xLogitsProcessor','_compute_lm_head'))
    assert 'torch.mm(' in muse and 'out_dtype=output_dtype' in muse and 'torch.float32 if self.use_fp32_output' in muse
    return {'before_capture':True,'greedy_deferred':True,'mode_fallback_cases':trials}


def numerical_examples():
    f=np.float32
    tiny=f(2.**-24)
    # All terms and scalars come from exactly representable BF16 factors, with H=8192.
    z=np.zeros(8192,dtype=f)
    z[:4]=(1.,tiny,-1.,tiny)
    summation=np.add.accumulate(z,dtype=f)[-1]
    paired=f(f(z[0]+z[2])+f(z[1]+z[3]))
    other=f(1.5)*tiny
    assert summation < other < paired
    # Explicit scale and cap maintain this reversal across the fully projected row.
    softcap=lambda x:f(20.)*np.tanh(f(x)*f(0.1767766953)/f(20.))
    assert np.argmax(softcap([summation,other]))==1 and np.argmax(softcap([paired,other]))==0
    sat=softcap([2000,3000]);assert sat[0]==sat[1] and np.argmax(sat)==0
    assert np.argmax([2000,3000])==1
    vals=list(zip((np.nan,0.,np.nan,-1.),range(4)))
    def pair(a,b):
        return a if a[0]>b[0] or (a[0]==b[0] and a[1]<b[1]) else b
    a,b,c,d=vals
    left=pair(pair(pair(a,b),c),d); balanced=pair(pair(a,b),pair(c,d));assert left!=balanced
    # Valid integer global IDs and sign-normalized score bits give stable finite tie ordering.
    finite=[f(x) for x in (-np.inf,-1.,-0.,0.,1.,np.inf)]
    def key(score,idx):
        bits=int(np.array([score if score!=0 else 0.],dtype=f).view('i4')[0]);return (bits^((bits>>31)&0x7fffffff), -idx)
    for v in itertools.product(finite,repeat=4):
        winner=max(range(4),key=lambda i:key(v[i],i));assert winner==int(np.argmax(v))
    return {'full_K':len(z),'alternative_fp32_sums':[float(summation),float(paired)],
            'competing_score':float(other),'finite_softcapped_winners':[1,0],
            'raw_vs_saturated_winner':[1,0],'nan_pair_reduction_winners':[left[1],balanced[1]],
            'packed_key_finite_tests':len(finite)**4}


def traffic():
    n,h,tp=25256,8192,8
    weights=n*h*2
    data={}
    for users in (1,4,16,32):
        rows=8*users
        shard=rows*n*4
        # GEMM output, in-place FP32 scaling read+write, then greedy softcap/selection read.
        excluded=4*shard
        draft_rows=7*users
        data[str(users)]={'target_rows':rows,'target_weight_mib':weights/2**20,'target_work_gflop':2*rows*n*h/1e9,
                          'dense_shard_mib':shard/2**20,'four_local_matrix_accesses_mib':excluded/2**20,
                          'matrix_accesses_per_target_weight_pass':excluded/weights,
                          'v64_winner_pair_local_bytes':8*rows,'v64_winner_pair_gather_output_bytes':8*rows*tp,
                          'v91_winner_key_local_bytes':8*rows,'draft_work_gflop':2*draft_rows*n*h/1e9,
                          'draft_dense_bf16_output_mib':draft_rows*n*2/2**20,
                          'candidate_128_column_tile_keys_write_plus_read_mib':rows*math.ceil(n/128)*8*2/2**20}
    return data


if __name__=='__main__':
    print(json.dumps({'pinned':PIN,'dispatch':dispatch(),'cpu_numerical_examples':numerical_examples(),
                      'logical_per_rank':traffic(),'h200_measured':False},indent=2))
