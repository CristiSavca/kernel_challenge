"""Replay B-sector LRU at multiple capacities and intra-wave K-tile arrival orders."""
import json
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
route=json.loads((here.parent/"v274_four_block_prompt_multicast/routes.json").read_text())["cases"]
cache=[1,4,9,12,18,24,27,32,36,40,50,64,90,144,288]
results=[]
for case in route:
    ranks=[]
    for hist in case["histograms"]:
        rank={}
        for name,cohort,grid in (("paired",1,132),("ordinary_quad",1,128),("cohort_quad",4,128),("global_quad",32,128)):
            outcomes={}
            for peers in (1,8,grid):
                row=list(map(float,subprocess.check_output([sys.argv[1],str(cohort),str(grid),str(peers),*map(str,hist)]).split()))
                outcomes[str(peers)]=dict(critical_rows=int(row[0]),weight_MiB=dict(zip(cache,row[1:])))
                assert abs(row[-1]-288)<1e-5
            rank[name]=outcomes
        ranks.append(rank)
    results.append(dict(tokens=case["tokens"],ranks=ranks))
summary=[]
for case in results:
    row={}
    for cap in (9,18,27,36,50):
        ratios=[]
        deltas=[]
        for rank in case["ranks"]:
            for q,p in (("1","1"),("8","8"),("128","132")):
                a=rank["cohort_quad"][q]["weight_MiB"][cap]
                b=rank["paired"][p]["weight_MiB"][cap]
                ratios.append(a/b)
                deltas.append(a-b)
        row[cap]=dict(minimum_extra_MiB=min(deltas),maximum_extra_MiB=max(deltas),largest_ratio=max(ratios))
    summary.append(dict(tokens=case['tokens'],capacities=row))
print(json.dumps(dict(scope="Standalone synthetic expert routes reconstructed from archived H200. B and scale sectors are 4608 bytes; fully associative LRU for the indicated effective B capacity. Arrival order varies from first CTA completing its whole K sweep to all resident CTAs progressing in K lockstep. Effective capacity models eviction by activations, outputs and other CUDA clients; no measured HBM bytes or GPU time.",summary=summary,results=results),indent=2))
