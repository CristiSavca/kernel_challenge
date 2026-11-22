"""Bound V234's unchanged guard and staging memory on archived full-chunk opportunities."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[3]
m=runpy.run_path(str(ROOT/'experiments/iterations/20261118-serving-mixed-graph-capacity.py'))
folder=m['ARCHIVE'];log=next(folder.glob('*v41__server.log'))
info=json.loads(next(folder.glob('*server_info.json')).read_bytes())['internal_states'][0]
loads=json.loads(next(folder.glob('*load.json')).read_bytes())
order=[r['concurrency'] for r in loads['rungs'] for _ in range(r['reps'])]
chunk=info['chunked_prefill_size'];width=info['speculative_num_draft_tokens']
out={c:Counter() for c in set(order)};guard=256;left=2047;h=2;d=64
assert width==8 and chunk==16384
for c,(prompts,rows,tails) in m['read_trace'](log.read_text(),order):
    live=rows+width*tails
    if not tails or live<=chunk:continue
    count=prompts+tails
    # The cache can retain no more than left preceding tokens for each query sequence.
    capacity=live+count*(left+guard)
    size=2*(capacity+live)*h*d
    assert size<=32*1024**2
    stat=out[c];stat['opportunities']+=1
    stat['one_prompt']+=int(prompts==1);stat['several_prompts']+=int(prompts>1)
    stat['max_total_sequences']=max(stat['max_total_sequences'],count)
    stat['max_live_rows']=max(stat['max_live_rows'],live)
    stat['max_guarded_dense_and_pending_writer_bytes']=max(stat['max_guarded_dense_and_pending_writer_bytes'],size)
print(json.dumps(dict(source=str(log.relative_to(ROOT)),sha256=sha256(log.read_bytes()).hexdigest(),
    cutoff_bytes=32*1024**2,window=left,guard_rows_per_sequence=guard,fp8_bytes_per_k_or_v_row=h*d,by_load=out,
    absent='V41 captured an ordinary schedule. Prefix lengths default to the maximum visible for this bound; actual mixed readiness, FA3 selection, throughput and peak require hardware.'),indent=2))
