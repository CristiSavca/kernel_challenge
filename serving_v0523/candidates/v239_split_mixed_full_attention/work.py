"""Bound full-layer prompt packing using the actual first-turn lengths from the same V41 wave."""
from collections import Counter
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[3]
wave=runpy.run_path(str(ROOT/'experiments/iterations/20261119-serving-cold-wave.py'))
archive=wave['ARCHIVE']
server=next(archive.glob('*v41__server.log'))
client=next(archive.glob('*v41__load.json'))
load=json.loads(client.read_bytes())
epochs=server.read_text().split('Cache flushed successfully!')[1:]
replications=[(r['concurrency'],i) for r in load['rungs'] for i in range(r['reps'])]
assert len(epochs)==len(replications)==7 and load['gpus']==8
cap=32*1024*1024
row_bytes=2*2*64
width=8
layers=16
out={}
for section,(users,rep) in zip(epochs,replications):
    clients=[c for c in load['requests'] if (c['load'],c['rep'],c['turn'])==(users,rep,0)]
    lengths=sorted(c['prompt'] for c in clients)
    assert len(lengths)==users
    counts=out.setdefault(users,Counter())
    for e in wave['cold_wave'](section,clients):
        new,running,prompts=e['new'],e['running'],e['seqs']
        live=new+width*running
        if not running or not 16384 < live <= 16640:
            continue
        counts['full_joint_arrivals']+=1
        # A prompt's live prefix plus this chunk cannot exceed its total first-turn length.
        # The archive omits prompt IDs at this point; use the longest possible distinct cohort.
        used_max=sum(lengths[-prompts:])
        size=(used_max+256*prompts+live)*row_bytes
        # When multiple prompts are admitted, even their cache hits and new rows alone can exceed the limit.
        size_min=(e['cached']+new+256*prompts+live)*row_bytes
        if size<=cap:
            counts['prompt_pack_certain_fit']+=1
            counts['certain_prompt_rows']+=new
            counts['certain_prompt_layer_calls']+=layers
            counts['certain_max_upper_bytes']=max(counts['certain_max_upper_bytes'],size)
            counts['certain_paged_tail_request_steps']+=running
            # Bound all-request packing against the cold history of each already-completed prompt.
            # Every chosen running request had an entire first turn in cache before it could verify.
            whole_min=(sum(lengths[:running])+8*running+new+e['cached']+256*(prompts+running)+live)*row_bytes
            counts['certain_fit_but_whole_pack_definitely_exceeds_cap']+=int(whole_min>cap)
        elif size_min>cap:
            counts['prompt_pack_certain_over_budget']+=1
        else:
            counts['unknown_prompt_prefix']+=1
    counts['repetitions']+=1
for row in out.values():
    assert row['full_joint_arrivals']==row['prompt_pack_certain_fit']+row['prompt_pack_certain_over_budget']+row['unknown_prompt_prefix']
print(json.dumps(dict(source=server.name,server_sha256=wave['digest'](server),client_sha256=wave['digest'](client),
        source_candidate='eight-H200 V41 physical scheduler and client trace; hypothetical V239 admission',
        full_attention_layers=layers,packed_and_converted_limit_bytes=cap,KV_bytes_per_logical_row=row_bytes,
        guard_rows_per_packed_request=256,rungs=out,
        interpretation='The bounds use the original initial wave and anonymous request memberships. An eligible attention bucket requires real mixed admission, native layout, graph destination and a linear cohort. Each split performs one additional FA3 launch per full layer and packs the prompt prefix from its real physical pages; no time saving is assumed.'),indent=2))

