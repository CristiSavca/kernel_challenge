"""Assess allowing one extra equal-duration wave when four-way residency is lower."""
from collections import Counter
from pathlib import Path
import json
import math
import sys

here=Path(__file__).resolve().parent
src=(here/"schedule.py").read_text().split("cases=json.loads",1)[0]
ns=dict(__file__=str(here/"schedule.py"))
exec(compile(src,str(here/"schedule.py"),"exec"),ns)
schedule=ns["schedule"]

def bound(rows,experts,n,pair,quad):
    nb=n//128
    active=min(rows,experts)
    if not rows: return 0,0,0
    lo=(rows+255)//256
    hi=active+(rows-active)//256
    extra=max((t*nb+quad-1)//quad - (t*nb+pair-1)//pair for t in range(lo,hi+1))
    return lo,hi,extra

def cost(hist,g,size):
    work=schedule(hist,size,g)
    return dict(max_jobs=max(map(len,work)),critical_issued_rows=max(sum(job[0][2] for job in stream) for stream in work))

rows=[]
for x in json.loads((here/"routes.json").read_text())["cases"]:
    case=[]
    for h in x["histograms"]:
        vals={q:cost(h,q,4) for q in [128,124,120]}
        base=cost(h,132,2)
        case.append(dict(pair132=base,quads=vals,uniform_job_bounds={q:bound(sum(h),32,4096,132,q) for q in vals}))
    rows.append(dict(tokens=x["tokens"],ranks=case))

# An adversarial legal histogram with one full-width tile collected on every fourth M tile.
hostile=[256,31,31,31]*8
rng=__import__("random").Random(27)
worst=(0,[]); best=(99,[])
for h0 in [hostile]+[[rng.choice((32,64,128,224,256,257,272,300,480,512,520,768,1024)) for _ in range(32)] for i in range(201)]:
    p=cost(h0,132,2); q=cost(h0,128,4)
    ratio=q["critical_issued_rows"]/p["critical_issued_rows"]
    if ratio>worst[0]: worst=ratio,h0
    if ratio<best[0]: best=ratio,h0
print(json.dumps(dict(purpose="Unshipped alternative. Equal-duration job-count bound is exact over the reported possible M-tile interval. Critical issued tensor rows is a proxy, without runtime or cache effects.",
    examples={r:{q:bound(r,32,4096,132,q) for q in (128,124,120)} for r in (8192,12288,16384,20480,24576,32768)},
    reconstructed=rows,
    regression_witness=dict(counts=worst[1],pair132=cost(worst[1],132,2),quad128=cost(worst[1],128,4),critical_row_ratio=worst[0]),
    improvement_witness=dict(counts=best[1],pair132=cost(best[1],132,2),quad128=cost(best[1],128,4),critical_row_ratio=best[0])),indent=2))
