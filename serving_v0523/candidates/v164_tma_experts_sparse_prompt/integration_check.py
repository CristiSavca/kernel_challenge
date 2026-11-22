"""Re-run V163's graph-boundary and hook fixtures with the V161 scaled prompt finalizer."""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
fixture=HERE.parent/'v163_sparse_prompt_exchange/boundary_check.py'
source=fixture.read_text()
# The parent fixture defines the staging readers, production loaders, graph
# fixture and independent per-slot reference before its top-level test loop.
prefix,run=source.split('\ntorch.set_num_threads(1)\n')
scope={'__file__':str(fixture),'__name__':'v164_boundary'}
exec(compile(prefix,str(fixture),'exec'),scope)
env=scope['env'];phase=scope['phase'];counts=Counter();results=[]
stage=scope['STAGE']
humming=scope['source'](scope['HUM'])
scope['install'](humming,['_compact_prefill_windows','compact_prefill_and_combine'],env)
orig=env['_compact_prefill_windows']
def windows(*args,**kwargs):
    planned=list(orig(*args,**kwargs));counts['multiple_window_calls']+=len(planned)>1
    counts['total_windows']+=len(planned)
    if args[-1] is not None:
        assert sum(v[2] for v in planned)==args[-1]
        if len(planned)==1: counts['single_window_known_count']+=1
    yield from planned
env['_compact_prefill_windows']=windows

class ScaledSilu:
    def run(self,gate,out,expert_ids,scales):
        counts['scaled_silu_calls']+=1
        g,u=(gate*scales[expert_ids.long(),None]).bfloat16().float().chunk(2,dim=-1)
        out.copy_(torch.nn.functional.silu(g)*u)

class Kernel:
    def __getitem__(self,grid):
        def launch(values,weight,scores,out,slots,expert_ids,scales,**args):
            assert grid==(len(out),) and args['H']==out.shape[1] and args['HAS_ROW_MAP']
            assert args['HAS_IDS']==args['HAS_EXPERT_SCALE']
            if args['HAS_EXPERT_SCALE']:
                counts['scaled_finalizer_calls']+=1
                live=slots>=0; by_row=values.new_empty((len(values),1))
                by_row[slots[live].long(),0]=scales[expert_ids[live].long()]
                values=(values*by_row).bfloat16()
            scope['finalize'](values,scores,norm_weight=weight,eps=args['eps'],weight_first=args['WEIGHT_FIRST'],
                              slot_expert_ids=None,slot_row_indices=slots,out=out)
            # The V163 oracle checks a single whole-bucket compact map. Assemble
            # the real per-window maps solely for its independent comparisons.
            matches=[i for i,v in enumerate(phase.records) if v[0]=='finalize']
            if matches:
                index=matches[0];_,prior,mapping,_=phase.records[index]
            else:
                index=len(phase.records);phase.records.append(None);prior=values[:0]
                mapping=slots.new_full((phase.bucket_rows,slots.shape[1]),-1)
            begin=out.storage_offset()//out.shape[1]
            mapping[begin:begin+len(out)]=torch.where(slots>=0,slots+len(prior),-1)
            phase.records[index]=('finalize',torch.cat((prior,values)),mapping,None)
        return launch

env['_post_expert_norm_weighted_sum_kernel']=Kernel()
exercise=scope['exercise']
def with_regime(rank,rows):
    phase.bucket_rows=rows
    for scaled in (False,True):
        scope['Expert'].humming_prefill_silu=ScaledSilu() if scaled else None
        exercise(rank,rows)
scope['exercise']=with_regime
# Reuse parent's seeded routes, live changes, hook tests, original-slot oracle
# and real eager wrappers, widen token rows to force a skewed multi-window case.
main=run.split('print(json.dumps')[0].replace('(256,257,512,513)','(256,257,512,513,1025,2049)')
exec(compile('torch.set_num_threads(1)\n'+main,str(fixture),'exec'),scope)
assert counts['multiple_window_calls']>0 and counts['scaled_finalizer_calls']==counts['scaled_silu_calls']
assert counts['single_window_known_count']>0
print(json.dumps(dict(torch=torch.__version__,counts=scope['observations'],scaled=counts,
                      admission=scope['admission'](),parent_fixture='v163_sparse_prompt_exchange/boundary_check.py',
                      scope='Actual grouped source, native host finalizer API and BCG replay with CPU substitutes for CUDA'),indent=2))
