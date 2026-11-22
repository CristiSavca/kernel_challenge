"""Compile pinned FA3 host selection for ragged page-one batches and their split prompt calls."""
from collections import Counter
import importlib.util
import inspect
import json
from pathlib import Path
import runpy

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
src=HERE.parent/'v170_long_prefill_attention_feasibility/check.py'
spec=importlib.util.spec_from_file_location('native_dispatch',src)
native=importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
# Reuse the pinned host fixture with the ABI values of actual ragged calls.
source=inspect.getsource(native.host_dispatch)
for old,new in (
    ('int b,q,k,page,left,variable;', 'int b,q,k,page,left,variable,total;'),
    ('>>left>>variable)', '>>left>>variable>>total)'),
    ('params.total_q=b*q;', 'params.total_q=total;'),
    ('params.cu_seqlens_q=variable;', 'params.cu_seqlens_q=variable;params.cu_seqlens_k=variable&&!paged_KV;'),
    ('expanded = [(*row,1) for row in cases] + [(b,q,k,0,w,0) for b,q,k in cold for w in (-1,2047)]','expanded = cases'),
):
    assert source.count(old)==1
    source=source.replace(old,new)
env=dict(vars(native))
exec(source,env)
wave=runpy.run_path(str(ROOT/'experiments/iterations/20261119-serving-cold-wave.py'))
archive=wave['ARCHIVE']; server=next(archive.glob('*v41__server.log')); clients=next(archive.glob('*v41__load.json'))
load=json.loads(clients.read_bytes())
order=[(r['concurrency'],i) for r in load['rungs'] for i in range(r['reps'])]
cases=set();by_waves=Counter()
for section,(users,rep) in zip(server.read_text().split('Cache flushed successfully!')[1:],order):
    records=[c for c in load['requests'] if (c['load'],c['rep'],c['turn'])==(users,rep,0)]
    max_prompt=max(c['prompt'] for c in records)
    for event in wave['cold_wave'](section,records):
        running,new=event['running'],event['new']
        if event['seqs']!=1 or not running or new+running*8<=16384:
            continue
        assert 256*(max_prompt+new+8*running+256)<=32*1024**2
        by_waves[users]+=1
        # Anonymous requests: evaluate a freshly admitted chunk and three possible longer prefix lengths.
        # For the unsplit call the maximum paged table includes the running histories.
        for p in sorted({0,new,min(3*new,max_prompt-new),max_prompt-new}):
            for page_columns in {max_prompt+8,131072}:
                cases.add((running+1,new,page_columns,1,-1,1,new+running*8))
                cases.add((1,new,new+p,0,-1,1,new))
                cases.add((running,8,page_columns,1,-1,1,running*8))
cases=sorted(cases)
rows,hashes,_,_=env['host_dispatch'](cases,[])
families={}
for (b,q,k,page,_,_,total),result in zip(cases,rows):
    kind='split_dense_prompt' if page==0 else ('native_paged_tail' if q==8 else 'v235_paged_mixed')
    if kind not in families:
        families[kind]=Counter()
    families[kind][tuple(result)]+=1
out={k:[dict(splits=x[0],kv_tma=bool(x[1]),pack_gqa=bool(x[2]),tile_M=x[3],tile_N=x[4],shapes=n)
        for x,n in sorted(v.items())] for k,v in families.items()}
assert all(row['kv_tma'] for row in out['split_dense_prompt'])
assert all(not row['kv_tma'] for row in out['v235_paged_mixed'])
print(json.dumps(dict(pinned_FA3=native.PIN,source_sha256=hashes,cases=len(cases),observed_joint_shapes_by_load=by_waves,
    native_host_dispatch=out,assumptions='Host selection for the real single-prompt arrival dimensions, page size one, FP8 D64, GQA8 and 132 SM with representative prefix extents. Tail rows show the native fallback. Actual device execution, tile service time, graph overhead and end-to-end performance are unmeasured.'),indent=2))

