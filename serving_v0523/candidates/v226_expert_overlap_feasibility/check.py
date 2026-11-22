"""Bound cross-expert W13/W2 overlap on unchanged V224 tiles. No GPU or artifact downloads required."""
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from statistics import mean
import json
import random

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
ARCHIVE = CAND.parents[1] / 'experiments/iterations/20261104-moe-hardware-profile-recovered.json'
WIDTH = {8: 8, 32: 8, 128: 16, 256: 24}
SMS, CHANNEL_CTAS, UNIT = 132, 32, 294912
TRIALS = 2048
RATIOS = ((1,1), (3,2), (2,1), (5,2), (3,1))

def read(candidate, file):
    return json.loads((next(CAND.glob(candidate)) / file).read_bytes())

def ceildiv(a,b):
    return (a+b-1)//b

def bounds(tiles, a, b):
    """Indivisible constant-duration channel jobs, duration unit gcd(a,b). W2 requires all W13 tiles of its expert-row block."""
    if not tiles: return 0,0
    n = CHANNEL_CTAS * tiles
    baseline=(a+b)*ceildiv(n,SMS)
    all_work=ceildiv((a+b)*n,SMS)
    # The final W13 owner starts no earlier than its own wave minus one.
    last_gate=a*ceildiv(n,SMS)+b
    lower=max(all_work,last_gate)
    assert lower<=baseline
    return baseline,lower

@lru_cache(None)
def schedule(tiles,a,b):
    """Construct an ASAP schedule. Each tile owns independent disjoint W13 and W2 output stripes.
    If W13 and a ready W2 compete for free SMs, prioritize W13. Once W13 is done, mix ready W2 into the last wave.
    """
    if not tiles: return 0
    gate_left=[CHANNEL_CTAS]*tiles
    done=[0]*tiles
    down_left=[CHANNEL_CTAS]*tiles
    running=Counter()
    now=0
    while True:
        for (end,kind,job),n in list(running.items()):
            if end!=now:continue
            if kind=='gate':done[job]+=n
            del running[end,kind,job]
        free=SMS-sum(running.values())
        for kind,duration,remaining in [('gate',a,gate_left),('down',b,down_left)]:
            for job,count in enumerate(remaining):
                if kind=='down' and done[job]!=CHANNEL_CTAS: continue
                n=min(count,free)
                if n:
                    remaining[job]-=n
                    running[now+duration,kind,job]+=n
                    free-=n
                if not free:break
            if not free:break
        if not running:break
        now=min(k[0] for k in running)
    assert all(v==CHANNEL_CTAS for v in done) and not any(gate_left+down_left)
    baseline,lower=bounds(tiles,a,b)
    assert lower<=now<=baseline
    return now

def outcomes(hist, maxrows):
    report=[]
    for a,b in RATIOS:
        bulk=[];low=[];feasible=[]
        for tiles,count in hist.items():
            x,y=bounds(tiles,a,b)
            bulk.extend([x]*count); low.extend([y]*count)
            feasible.extend([schedule(tiles,a,b)]*count)
        row=dict(gate_to_down=f'{a}/{b}',mean_bulk_units=mean(bulk),mean_ideal_lower_units=mean(low),
                 mean_constructive_units=mean(feasible),best_possible_component_ratio=mean(bulk)/mean(low),
                 constructive_component_ratio=mean(bulk)/mean(feasible),
                 no_gain_fraction=sum(x==y for x,y in zip(bulk,low))/len(bulk))
        if (a,b)==(2,1):
            # One bandwidth-equivalent wave quantum, assuming each busy CTA consumes equal bandwidth/SM.
            quantum=SMS*UNIT/4.8e12*1e6
            row.update(optimistic_us_per_layer_at_peak_equal_SM_bandwidth=(mean(bulk)-mean(low))*quantum,
                       constructive_us_per_layer_at_peak_equal_SM_bandwidth=(mean(bulk)-mean(feasible))*quantum)
        report.append(row)
    return dict(trials=sum(hist.values()),critical_row_tiles={str(k):v for k,v in sorted(hist.items())},
                mean_critical_row_tiles=sum(k*v for k,v in hist.items())/sum(hist.values()),
                max_theoretical_activation_roundtrip_bytes_per_rank=maxrows*8*4096*4,
                mean_rank_activation_roundtrip_bytes_uniform=maxrows*4096*4,ratios=report)

def main():
    patch=next(CAND.glob('v224_*/patch.diff')).read_bytes()
    m=read('v224_*','validation.json')
    assert sha256(patch).hexdigest()==m['patch_sha256']
    code=patch.decode()
    section=code[code.index('+def humming_grouped_decode_outputs('):code.index('+def _scale_live_routes(')]
    assert section.index('_gather_grouped_decode[')<section.index('activations = run_gate_up(')<section.index('outputs = inputs')<section.index('run_down(down')
    assert '+    return outputs, row_map, local_ids' in section
    gate=read('v191_*','config.json')['configs'];down=read('v205_*','configs.json')
    machine=read('v213_*','sm90a.json')['cases']
    native=[]
    for width in (8,16,24):
        w13,w2=gate[f'8192_{width}'],down[f'4096_{width}']
        assert w13['block']==[width,256,128] and w2['block']==[width,128,256]
        assert 8192//w13['block'][1]==4096//w2['block'][1]==CHANNEL_CTAS
        for label,conf in [('w13',w13),('w2',w2)]:
            compiled=machine[f'{label}_{width}']['V213']
            # V224 preserves these six machine bodies: prove header/renderer source identity within official git patches.
            assert conf['threads']==640 and compiled['regs']==96 and conf['shared']==compiled['shared_bytes']
            assert 65536//(conf['threads']*compiled['regs'])==1
            native.append(dict(role=label,width=width,threads=conf['threads'],regs=compiled['regs'],
                               shared=conf['shared'],sms=SMS,channel_jobs_per_tile=CHANNEL_CTAS))
    previous=next(CAND.glob('v213_*/patch.diff')).read_bytes()
    protected=['grouped_decode_mma_pair.cuh','live_decode_reduce.cuh','decode_gate_up.cuh',
               'decode_gate_up_launch.cuh','decode_mma_pair.py','decode_gate_up.py','live_decode_reduce.py']
    def file_entry(p,basename):
        for f in p.split(b'diff --git ')[1:]:
            path=f.splitlines()[0].split(b' b/')[-1].decode()
            if path.endswith('/'+basename):return f
        raise AssertionError(basename)
    for f in protected:assert file_entry(previous,f)==file_entry(patch,f), f
    rng=random.Random(226)
    simulated=[]
    for rows,width in WIDTH.items():
        hist=Counter()
        for _ in range(TRIALS):
            counts=[0]*256
            for _ in range(rows):
                for i in rng.sample(range(256),8): counts[i]+=1
            ranks=[sum(ceildiv(c,width) for c in counts[r:r+32]) for r in range(0,256,32)]
            hist[max(ranks)]+=1
        simulated.append(dict(rows=rows,native_width=width,**outcomes(hist,rows)))
    physical=[]
    for case in json.loads(ARCHIVE.read_bytes())['cases']:
        rows=case['M'];counts=case['routing']['local_counts'];width=WIDTH[rows]
        tiles=sum(ceildiv(c,width) for c in counts)
        physical.append(dict(rows=rows,local_live_routes=sum(counts),histogram=counts,tile_count=tiles,
                         duration_of_different_kernel_us=case['conditional_same_traffic_4_8TB_s_calculation']['profile_duration_us'],
                         packed_bank_MiB=tiles*27,activation_roundtrip_MiB=sum(counts)*4096*4/2**20,
                         **outcomes({tiles:1},rows)))
    matrix=[dict(tiles=t,bulk=bounds(t,2,1)[0],semantically_necessary=bounds(t,2,1)[1],
                 constructive=schedule(t,2,1)) for t in range(1,513)]
    timings=read('v175_*','check.json')['per_rung']
    for row,old in zip(simulated,timings):
        assert row['rows']==8*old['users']
        sim=row['ratios'][2]
        delta=sim['constructive_us_per_layer_at_peak_equal_SM_bandwidth']*62/1000
        period=old['approx_v64_period_ms']
        row['historical_V64_period_ms']=period
        row['62_layer_delta_ms_at_equal_SM_peak']=delta
        row['illustrative_V64_period_ratio']=period/(period-delta)
    for a,b in RATIOS:
        all_bounds=[bounds(t,a,b) for t in range(1,513)]
        assert all(x<=y and v<=w for (x,v),(y,w) in zip(all_bounds,all_bounds[1:]))
    print(json.dumps(dict(status='no production patch',parent_patch=m['patch_sha256'],tree=m['tree'],
       unchanged_decode_bodies=protected,occupancy=native,physical_input=physical,
       assumptions='Work-conserving 132 identical SMs, one nonpreemptive 640-thread CTA per SM, all CTAs of each phase same duration; 2:1 weights and 1:1..3:1 sensitivity. No scheduler stalls, no memory contention, no extra launches or synchronization charged. Humming persistent grid treated as balanced; relative times and concurrency are unmeasured.',
       critical_rank_simulation=simulated,exact_or_order_exhausted_shapes_at_2_to_1=sum(r['semantically_necessary']==r['constructive'] for r in matrix),
       checked_shapes=len(matrix),physical_h200_run=False),indent=2))

if __name__=='__main__': main()
