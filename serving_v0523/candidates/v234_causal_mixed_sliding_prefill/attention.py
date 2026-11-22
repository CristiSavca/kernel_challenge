"""Compare V232 and V234 through the native model, Radix callback and FA3 entry."""
import inspect
import json
import subprocess
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v223_batched_direct_prompt'))
import independent_replay as f

candidate=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
f.source,f.harness,_=f.fixtures(candidate)
for name in ('is_decode','is_idle','is_mixed','is_decode_or_idle','is_draft_extend_v2'):
    setattr(f.harness.Mode,name,lambda self:False)
f.harness.Mode.is_extend=lambda self:True
f.harness.Mode.is_extend_or_draft_extend_or_mixed=lambda self,**kwargs:True
reader=sys.modules['v223_source_reader']
reader.PARENT=(HERE.parent/'v232_mixed_prompt_overflow/patch.diff').read_text()
# The inherited physical-cache and FP64 oracle already implements the complete
# verification. Both of these parents have V223; supply both helper functions.
schema=inspect.getsource(f.exercise)
schema=schema.replace('runtime = dict(torch=torch,contextmanager=contextmanager',
                      'runtime = dict(torch=torch,ForwardMode=harness.Mode,contextmanager=contextmanager')
schema=schema.replace('if child and path==harness.HELP','if path==harness.HELP')
schema=schema.replace('if guard=="mixed": fb.mixed_verify_logits_indices=cu','''fb.mixed_verify_logits_indices=cu
        fb.mixed_verify_is_causal=guard!="uncertified"
        fb.spec_info=None''')
schema=schema.replace('        if guard=="writer_hook":','        if guard=="spec": fb.spec_info=NS()\n        if guard=="writer_hook":')
assert schema.count('fb.mixed_verify_is_causal=')==1
exec(schema,vars(f))
torch.set_num_threads(1)
cases=[]
for label,rows,prefixes,window,positive,guard in [
    ('eager',[4101,8],[0,2513],2047,True,None),
    ('breakable',[4096,8,8,8],[701,8192,0,32],2047,True,None),
    ('breakable',[1973,2319,8,8],[10000,997,4145,2048],2047,True,None),
    ('breakable',[16384,8,8,8],[5376,2047,133,4102],2047,True,None),
    ('breakable',[16384]+[8]*31,[16384]+[2047+23*j for j in range(31)],2047,True,None),
    ('breakable',[4096,8],[0,16383],2047,False,'uncertified'),
    ('breakable',[4096,8],[0,2047],-1,False,None),
    ('breakable',[4096,8],[0,2047],2047,False,'spec'),
    ('breakable',[4096,8],[0,2047],2047,False,'writer_hook'),
    ('breakable',[4096,8],[0,2047],2047,False,'mirror'),
    ('full',[4096,8],[0,2047],2047,False,None),
    ('warmup',[4096,8],[0,2047],2047,False,None),
    ('breakable',[4096,8],[0,2047],2047,False,'capturing'),
]:
    cases.append(f.exercise(label, rows, prefixes, window, positive, guard=guard))
# The independent combined witness exercises the selected V231 FP8 producer and
# a subsequent ordinary request reusing freed SWA slots on the same captured graph.
# The direct comparison above exercises the real model's supported unfused producer.
encoded_query=None
if 'muse_prompt_query_encodings' in f.source(f.MODEL, True):
    witness=HERE.parent/'v235_peer_mixed_sliding_prefill/independent_causal_attention.py'
    encoded_query=json.loads(subprocess.check_output(
        [sys.executable,'-B',str(witness),'--candidate',str(candidate)],text=True))
print(json.dumps(dict(encoded_query_producer=encoded_query,parent='V232',candidate=candidate.name,case_count=len(cases),
    paired_forwards=2*len(cases),physical_cache_bytes=sum(x['physical_cache_bytes'] for x in cases),
    sampled_fp64_attention_scalars=sum(x['attention_output_scalars'] for x in cases),
    cases=cases,scope='Native model/qk transform, Radix, FA3 argument dispatch, metadata, full/SWA translation, original writes and cache bytes; launch storage modeled on CPU, sampled FP64 attention reference.'),indent=2))
