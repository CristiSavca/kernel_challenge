"""Apply hypothetical bounded subsegments / maximal affine runs to the existing allocator replay."""
from collections import Counter
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
parent=HERE/'independent_allocator.py'
source=parent.read_text()
# Import the exact pinned allocator and trace driver from this candidate without emitting its standalone report.
scope={"__file__":str(parent)}
exec(compile(source.partition('client=next(')[0],str(parent),'exec'),scope)
np=scope['np']
def variants(inputs):
    sums=Counter()
    tile=scope['K_TILE'];parts=scope['PARTS']
    counts=[(len(x)+tile-1)//tile for x in inputs]
    per=(sum(counts)+parts-1)//parts
    base=0
    for t,count in zip(inputs,counts):
        a=t.numpy();n=len(a)//tile
        live=np.zeros(count,dtype=bool)
        blocks=a[:n*tile].reshape(n,tile)
        live[:n]=(blocks[:,:-1]+1==blocks[:,1:]).all(axis=1)
        sums['tiles']+=count;sums['individual']+=int(live.sum())
        for part in range(base//per,(base+count-1)//per+1):
            lo=max(base,part*per)-base;hi=min(base+count,(part+1)*per)-base
            ids=live[lo:hi];sums['segments']+=1
            if ids.all():sums['whole_segment']+=len(ids)
            for width in (4,8):
                for off in range(0,len(ids),width):
                    group=ids[off:off+width]
                    if group.all():sums[f'group{width}']+=len(group)
            k=0
            for val in np.r_[ids,False]:
                if val:k+=1
                elif k:
                    if k>=2:sums['run2']+=k
                    if k>=4:sums['run4']+=k;sums['run4_regions']+=1
                    k=0
        base+=count
    return sums
scope['coverage']=variants
out={}
client=next((scope['ROOT']/'experiments/modal_node/20261113T204343Z-bench').glob('*v41__load.json'))
from collections import defaultdict
history=defaultdict(list)
for q in json.loads(client.read_bytes())['requests']:history[q['load'],q['rep']].append(q)
for free in (0,4096,256,32):
    data=defaultdict(Counter)
    for (b,rep),jobs in sorted(history.items()):data[b].update(scope['run'](jobs,free)[0])
    out['fresh' if not free else f'free_runs_{free}']={b:dict(c,coverage={k:c[k]/c['tiles'] for k in
        ('whole_segment','group4','group8','run2','run4','individual')}) for b,c in data.items()}
print(json.dumps(dict(scope="Alternative dispatch geometry on the same synthetic physical allocator histories and measured request sizes. No runtime or model hardware timing.",
    source=scope['sha256'](parent.read_bytes()).hexdigest(),results=out),indent=2))
