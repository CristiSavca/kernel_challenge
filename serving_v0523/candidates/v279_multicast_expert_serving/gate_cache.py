"""Price W13 source reuse with clustered issuers and a shared finite L2 model."""
from collections import Counter, OrderedDict
from pathlib import Path
import json

here=Path(__file__).resolve().parent
histograms=json.loads((here.parent/"v274_four_block_prompt_multicast/routes.json").read_bytes())["cases"]
counts=histograms[-1]["histograms"][3]


def replay(fan, grid, capacity, k_progress):
    # Each TMA uses one public 176 x 64 input box and one distinct 256 x 64 packed weight + scale slice.
    tile_rows=[(expert,row,n) for expert,n in enumerate(counts) for row in range(0,n,176)]
    ctas=[(expert,row,n,band) for expert,row,n in tile_rows for band in range(32)]
    lru=OrderedDict()
    size=0
    reads,misses=Counter(),Counter()
    def access(key,count):
        nonlocal size
        kind=key[0]
        reads[kind]+=count
        if key in lru:
            lru.move_to_end(key)
            return
        misses[kind]+=count
        if count>capacity:
            return
        while size+count>capacity:
            size-=lru.popitem(last=False)[1]
        lru[key]=count
        size+=count
    assert grid%fan==0
    for begin in range(0,len(ctas),grid):
        wave=ctas[begin:begin+grid]
        for start_k in range(0,64,k_progress):
            for base in range(0,len(wave),fan):
                cluster=wave[base:base+fan]
                e,row,n,band=cluster[0]
                assert len(cluster)==fan and all((a,b,c)==(e,row,n) for a,b,c,_ in cluster)
                for k in range(start_k,min(64,start_k+k_progress)):
                    access(("A",e,row,k),176*64*2)
                    for _,_,_,out in cluster:
                        access(("bank",e,out,k),256*64*9//16)
    return {name:{k:v/2**20 for k,v in data.items()} for name,data in (("request_MiB",reads),("miss_MiB",misses))}


cases=[]
for cap in (0,16,50):
    for step in (1,4,16,64):
        native=replay(1,132,cap*2**20,step)
        for grid in (128,132):
            candidate=replay(4,grid,cap*2**20,step)
            assert native["request_MiB"]=={"A":4356,"bank":1782}
            assert candidate["request_MiB"]=={"A":1089,"bank":1782}
            saved={key:value-candidate["miss_MiB"][key] for key,value in native["miss_MiB"].items()}
            cases.append(dict(capacity_MiB=cap,K_steps_per_issuer=step,mc_grid=grid,
                              native=native,mc4=candidate,avoided_miss_MiB=saved))
print(json.dumps(dict(cases=cases,scope="Synthetic, fully associative LRU per projection. One activation read per cluster, "
    "132 ordinary CTAs, 128/132 clustered CTAs, identical packed slices and grouped N-first order. "
    "Assumes synchronized persistent waves, constant issue pace, complete separate source boxes per M tile, no output allocation, "
    "no L1, and no real cache-set or replacement behavior. These are sensitivity cases, not device measurements."),indent=2))
