"""Model jointly cached BF16 activation row segments, packed weights, scales and optionally output writes."""
import json
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
cases=json.loads((here.parent/"v274_four_block_prompt_multicast/routes.json").read_text())["cases"]
results=[]
for case in cases:
    ranks=[]
    for hist in case["histograms"]:
        draws={}
        for name,cohort,grid,fan in (("pair132",1,132,2),("quad128",1,128,4),("ordered128",4,128,4),("within_wave",0,128,4)):
            scenarios={}
            for allocate in (0,1):
                for count in ("cluster","four_clusters","grid"):
                    peers={"cluster":fan,"four_clusters":fan*4,"grid":grid}[count]
                    nums=list(map(float,subprocess.check_output([sys.argv[1],str(cohort),str(grid),str(fan),str(peers),str(allocate),*map(str,hist)]).split()))
                    scenarios[f"{allocate}_{count}"]={cap:dict(A_MiB=a,B_MiB=b) for cap,a,b in zip((18,27,36,40,50),nums[::2],nums[1::2])}
            draws[name]=scenarios
        ranks.append(draws)
    results.append(dict(tokens=case["tokens"],ranks=ranks))
summary={}
for case in results:
    summary[case["tokens"]]={}
    for cap in (18,27,36,40,50):
        deltas=[]; weight=[]
        for rank in case['ranks']:
            for k,q in rank['ordered128'].items():
                p=rank['pair132'][k][cap]
                deltas.append(sum(q[cap].values())-sum(p.values()))
                weight.append(q[cap]['B_MiB']-p['B_MiB'])
        summary[case["tokens"]][cap]=dict(minimum_extra_miss_MiB=min(deltas),max_extra_miss_MiB=max(deltas),
            max_extra_weight_misses_MiB=max(weight))
print(json.dumps(dict(scope="All ranks, 3 plausible in-wave CTA arrival orders, with/without write allocation. Fully associative LRU, variable size exact-overlap A row segments for full TMA source boxes truncated at overall tensor end, one packed weight+scale sector per 128x64, cold cache. Reported misses exclude compulsory C writes common to all variants. No set conflicts, compression, other clients or timing.",summary=summary,cases=results),indent=2))
